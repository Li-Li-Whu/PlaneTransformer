# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import datetime
import logging
import math
import time
import sys
import os
import numpy as np
from fvcore.nn import FlopCountAnalysis
from sklearn.neighbors import NearestNeighbors

#torch.set_printoptions(threshold=np.inf)
#torch.autograd.set_detect_anomaly(True)


from torch.distributed.distributed_c10d import reduce
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)


# 计算模型的总参数量
def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()  # numel() 返回张量中元素的个数
    return total_params


def check_nan(tensor, message="Check point"):
    if torch.isnan(tensor).any():
        print(f"NaN detected at {message}")
        #raise ValueError(f"NaN detected at {message}")


def check_gradients(model, _name_):
    k = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(_name_)
                print(f"NaN gradient at {name}")   
                k+=1
    if k > 0:
        raise ValueError(f"NaN gradient at {_name_}")




def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr



def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion_pt,
    criterion_pl,
    dataset_config,
    dataset_loader,
    logger,
):

    loss_calculator = True
    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    loss_center_dist_avg = 0
    loss_center_angle_avg = 0
    loss_sem_cls_avg = 0
    loss_point_dist_avg = 0
    loss_point_angle_avg = 0
    loss_mask_ce_avg = 0
    loss_dice_avg = 0

 
    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
              if key != "scan_name":
                   batch_data_label[key] = batch_data_label[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_clouds_normalized": batch_data_label["point_clouds_normalized"],
            # "rg_center_normalized": batch_data_label["rg_center_normalized"],
            # "rg_center_present": batch_data_label["rg_center_present"],
            #"gt_rg_centers_present": batch_data_label["gt_rg_centers_present"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }

        '''
        #param, #FLOP
        all_params = count_parameters(model)
        print(all_params*4/1024/1024)
        flops = FlopCountAnalysis(model, inputs)
        print(flops.total()*4/1024/1024/1024)
        assert 1==0, "2222222222222"
        '''
        if not args.use_center_proposal and not args.mask_pred:
            pt_outputs, pl_outputs = model(inputs)
        else:
            pl_outputs = model(inputs)

        output_tensor = pl_outputs["outputs"]
        #检查forward
        for key, value in output_tensor.items():
            #print(key, value) 
            check_nan(value, "after forward pass")

        # Compute loss(point, plane)
        #print(pl_outputs["outputs"])
        #print(pl_outputs["outputs"]["mask_heatmaps"])
        
        
        loss_pl, loss_dict_pl, assignments = criterion_pl(pl_outputs, batch_data_label)
        if not args.use_center_proposal and not args.mask_pred:
            loss_pt, loss_dict_pt = criterion_pt(pt_outputs, batch_data_label)
        
        #check_nan(loss_dict_pl["loss_center_dist"], "center_dist")
        #check_nan(loss_dict_pl["loss_center_angle"], "center_angle")
        #检查loss 有nan
        for key, value in loss_dict_pl.items():
            check_nan(value, key)
        if not args.use_center_proposal and not args.mask_pred:
            for key, value in loss_dict_pt.items():
                check_nan(value, key)   
        #print("iter_%d_loss: " % curr_iter)
        #print(loss)
        #print("--------------------------------------")
     
        loss_reduced_pl = all_reduce_average(loss_pl)
        loss_dict_reduced_pl = reduce_dict(loss_dict_pl)
        if not args.use_center_proposal and not args.mask_pred:
            loss_reduced_pt = all_reduce_average(loss_pt)
            loss_dict_reduced_pt = reduce_dict(loss_dict_pt)
        #print(loss_reduced)


        if not math.isfinite(loss_reduced_pl.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            print("Loss in not finite. Training will be stopped.")
            sys.exit(1)
        
        #loss = loss.contiguous()
        #torch.autograd.set_detect_anomaly(True)
        #with torch.autograd.detect_anomaly():\
        if not args.use_center_proposal and not args.mask_pred:
            loss = loss_pt + loss_pl
        else:
            loss = loss_pl
        

        loss.backward()
        check_gradients(model, "after backward pass")
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        

        optimizer.step()
        check_gradients(model, "after optimizer step")

        torch.cuda.empty_cache()
        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced_pl.item())
  
        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print("*"*20)
            print("scan_name")
            print(batch_data_label["scan_name"])
            
            print("pl: loss_center_dist")
            print(loss_dict_pl["loss_center_dist"])
            print("pl: loss_center_angle")
            print(loss_dict_pl["loss_center_angle"])
            if args.mask_pred:
                print("pl: loss_mask_ce")
                print(loss_dict_pl["loss_mask_ce"])
                print("pl: loss_dice")
                print(loss_dict_pl["loss_dice"])

            #print("loss_normal")
            #print(loss_dict["loss_normal"])
            print("pl: loss_sem_cls")
            print(loss_dict_pl["loss_sem_cls"])
            print("pl: loss_cardinality")
            print(loss_dict_pl["loss_cardinality"])
            if not args.use_center_proposal and not args.mask_pred:
                print("*"*20)
                print("pt: loss_point_dist")
                print(loss_dict_pt["loss_point_offset_dist"])
                print("pt: loss_point_angle")
                print(loss_dict_pt["loss_point_offset_angle"])

            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            print("*"*20)  
            logger.log_scalars(loss_dict_reduced_pl, curr_iter, prefix="Train_details/")
            
            if not args.use_center_proposal and not args.mask_pred:
                logger.log_scalars(loss_dict_reduced_pt, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")
        
        curr_iter += 1
        loss_center_dist_avg += loss_dict_pl["loss_center_dist"]
        loss_center_angle_avg += loss_dict_pl["loss_center_angle"]
        if args.mask_pred:
            loss_mask_ce_avg += loss_dict_pl["loss_mask_ce"]
            loss_dice_avg += loss_dict_pl["loss_dice"]

        loss_sem_cls_avg += loss_dict_pl["loss_sem_cls"]
        
        if not args.use_center_proposal and not args.mask_pred:
            loss_point_dist_avg += loss_dict_pt["loss_point_offset_dist"]
            loss_point_angle_avg += loss_dict_pt["loss_point_offset_angle"]

        #print("dataset_loader")
        #print(len(dataset_loader))
        
        if is_primary() and curr_iter % len(dataset_loader) == 0:   #args.log_every_epoch
            print("="*20)
            if not args.use_center_proposal and not args.mask_pred:
                print("loss_point_dist_avg")
                print(loss_point_dist_avg / len(dataset_loader))
                print("loss_point_angle_avg")
                print(loss_point_angle_avg / len(dataset_loader))
                print("="*20)
            
            print("loss_center_dist_avg")
            print(loss_center_dist_avg / len(dataset_loader))
            print("loss_center_angle_avg")
            print(loss_center_angle_avg / len(dataset_loader))
            if args.mask_pred:
                print("loss_mask_ce_avg")
                print(loss_mask_ce_avg / len(dataset_loader))
                print("loss_dice_avg")
                print(loss_dice_avg / len(dataset_loader))


            print("loss_sem_cls_avg")
            print(loss_sem_cls_avg / len(dataset_loader))
            
            train_avg_dict = {}
            train_avg_dict["loss_center_dist_avg"] = loss_center_dist_avg / len(dataset_loader)
            train_avg_dict["loss_center_angle_avg"] = loss_center_angle_avg / len(dataset_loader)
            if args.mask_pred:
                train_avg_dict["loss_mask_ce_avg"] = loss_mask_ce_avg / len(dataset_loader)
                train_avg_dict["loss_dice_avg"] = loss_dice_avg / len(dataset_loader)

            train_avg_dict["loss_sem_cls_avg"] = loss_sem_cls_avg / len(dataset_loader)
            if not args.use_center_proposal and not args.mask_pred:
                train_avg_dict["loss_point_dist_avg"] = loss_point_dist_avg / len(dataset_loader)
                train_avg_dict["loss_point_angle_avg"] = loss_point_angle_avg / len(dataset_loader)


            logger.log_scalars(train_avg_dict, curr_epoch, prefix="Train_avg/")
        
        barrier()

    return loss_calculator




@torch.no_grad()
def evaluate_pl(
    args,
    curr_epoch,
    model,
    criterion_pl,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    point_calculator = True
    curr_iter = 0
    #max_iters = 300
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    loss_center_dist_avg = 0
    loss_center_angle_avg = 0
    loss_mask_ce_avg = 0
    loss_dice_avg = 0
    loss_sem_cls_avg = 0


    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    
    k_num_wrong_more = 0
    wrong_more_name = []
    k_num_wrong_less = 0
    wrong_less_name = []

    c_avg = 0
    c_avg_right = 0
    k_label_right = 0
    label_right_name = []
    
    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        #curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            if key != "scan_name":
                batch_data_label[key] = batch_data_label[key].to(net_device)
        
        #optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_clouds_normalized": batch_data_label["point_clouds_normalized"],
            # "rg_center_normalized": batch_data_label["rg_center_normalized"],
            # "rg_center_present": batch_data_label["rg_center_present"],
            #"rg_centers_normalized": batch_data_label["rg_centers_normalized"],
            #"gt_rg_centers_present": batch_data_label["gt_rg_centers_present"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        
        if not args.use_center_proposal and not args.mask_pred:
            _, outputs = model(inputs)
        else:
            outputs = model(inputs)
        loss, loss_dict, assignments = criterion_pl(outputs, batch_data_label)
        loss_str = ""
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_avg.update(loss_reduced.item())
        loss_str = f"Loss {loss_avg.avg:0.2f};"
        
        #loss.requires_grad_(True)
        #print(loss)
  

        print("assignments[assignments].size()")
        print(assignments["assignments"])
        #print(assignments["assignments"].shape())
        print(assignments["assignments"][0][0])
        print(assignments["assignments"][0][1])
        assign_pred = assignments["assignments"][0][0].cpu().numpy()
        assign_tgts = assignments["assignments"][0][1].cpu().numpy()
        print(assign_pred)
        print(assign_tgts)
        length = len(assign_tgts)
        #res_center = np.zeros((length, 3), dtype=np.float32)
        #res_angle = np.zeros((length, 2), dtype=np.float32)


        if args.test_only is True:
            pc_norm = inputs["point_clouds_normalized"].cpu().numpy().squeeze()
            np.savetxt(os.path.join(args.plane_result_dir, "".join(batch_data_label["scan_name"])) + '.txt', pc_norm, fmt='%.6f')

            outputs = outputs["outputs"]
        
            outputs_sem_cls = outputs["sem_cls_logits"][0]
            outputs_center = outputs["center_normalized"][0].cpu().numpy()
            # outputs_mask_heatmaps = outputs["mask_heatmaps"][0].sigmoid().cpu().numpy()
            
            '''
            prop_cls = torch.nn.functional.softmax(outputs_sem_cls, dim=-1)[:, :].max(-1).indices
            prop_val = torch.nn.functional.softmax(outputs_sem_cls, dim=-1)[:, :].max(-1).values
            prop_cls = prop_cls.cpu().numpy()
            prop_val = prop_val.cpu().numpy()
            '''
            prop_cls = torch.sigmoid(outputs_sem_cls)
            prop_cls = prop_cls.cpu().numpy()
            
            #print(prop_cls)
            #print(prop_val)        
            #assert 1==0, '000000000000'  
            center_id = []
            '''
            for i, itm in enumerate(prop_cls):
                if itm == 0:
                    value = prop_val[i]
                    if value > 0.5:
                        center_id.append(i)
            '''
            for i, itm in enumerate(prop_cls):
                value = itm
                if value > 0.5:
                    center_id.append(i)
                
            print(center_id)            
            #assert 1==0, '111111111111'  
            
            if len(center_id) > len(assign_tgts):
                k_num_wrong_more += 1
                wrong_more_name.append(batch_data_label["scan_name"])
            
            if len(center_id) < len(assign_tgts):
                k_num_wrong_less += 1
                wrong_less_name.append(batch_data_label["scan_name"])
            
            center_tgts = batch_data_label["gt_plane_center_normalized"][0].cpu().numpy()
            cdist = 0
            k = 0
            for i, idx in enumerate(center_id):
                for j, itm in enumerate(assign_pred):
                    if idx == itm:
                        k += 1
                        tgt_id = assign_tgts[j]
                        center_t = center_tgts[tgt_id]
                        center_p = outputs_center[idx, :]
                        cdist += (abs(center_p[0] - center_t[0]) + abs(center_p[1] - center_t[1]) + abs(center_p[2] - center_t[2])) / 3.0
            
            if k == 0:
                k += 1
            
            cdist /= k

            if k == len(assign_tgts) and len(center_id) == len(assign_tgts):
                k_label_right += 1
                c_avg_right += cdist
                label_right_name.append(batch_data_label["scan_name"])
            
            c_avg += cdist
            
            print("---------------------------------")
            print("".join(batch_data_label["scan_name"]))
            fname = "".join(batch_data_label["scan_name"])
            with open(os.path.join(args.plane_result_dir, fname) + "_pred_test.txt", "w") as f1:
                for i, id in enumerate(center_id):
                    center_pred = outputs_center[id, :]
                    line1 = str(center_pred[0]) + ' ' + str(center_pred[1]) + ' ' + str(center_pred[2]) + ' ' + str(1.0)
                    f1.write(line1)
                    if i < len(center_id) - 1:
                        f1.write('\n')
            f1.close()
            
            if args.mask_pred:
                outputs_mask_heatmaps = outputs["mask_heatmaps"][0].sigmoid().cpu().numpy()
                if len(center_id) > 0:
                    center_id = np.array(center_id)
                    mask_heatmaps = outputs_mask_heatmaps[:, center_id]
                    # np.savetxt(os.path.join(args.mask_heatmaps_result_dir, fname) + "_heatmaps.txt", mask_heatmaps, fmt='%.7f')
                

                    # gt_mask = np.loadtxt("roofpc3d/scans_test/" + fname + ".txt")[:, :4]
                    gt_mask = np.loadtxt("building3d/scans/" + fname + ".txt")[:, :4]
                    point_cloud = torch.from_numpy(gt_mask[:, :3]).to('cuda')
                
                    
                
                    mask_full = np.zeros((len(mask_heatmaps), 1), dtype=np.float32)
                    for point_id in range(len(mask_heatmaps)):
                        max_score = 0.0
                        max_id = 0.0
                        for instance_id in range(len(center_id)):
                            #ins_score = scores[instance_id] 
                            hea_score = mask_heatmaps[:, instance_id]

                            #mask = pred_masks[:, instance_id].astype("uint8")
                            #if mask[point_id] == 1:
                            score = hea_score[point_id] #* ins_score
                            if score > max_score:
                                max_score = score
                                max_id = instance_id + 1 
                            else:
                                continue

                        mask_full[point_id][0] = float(max_id)
            
                    mask_full -= 1.0
        
                
                    gt_mask = gt_mask.reshape(len(gt_mask), -1)
                    pred_coor_mask_full = np.hstack((gt_mask, mask_full))
                    pred_mask_full_path = "pred_mask_full"
                    np.savetxt(
                        f"{pred_mask_full_path}/{fname}_mask_full.txt",
                        pred_coor_mask_full,
                        fmt="%.5f",
                    )

            print("---------------------------------")


           
            
        # Compute loss

        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:    
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print("*"*20)
            print("loss_center_dist")
            print(loss_dict["loss_center_dist"])
            print("loss_center_angle")
            print(loss_dict["loss_center_angle"])
            if args.mask_pred:
                print("loss_mask_ce")
                print(loss_dict["loss_mask_ce"])
                print("loss_dice")
                print(loss_dict["loss_dice"])
            print("loss_sem_cls")
            print(loss_dict["loss_sem_cls"])

            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
            print("*"*20)
            

            test_dict = {}
            test_dict_details = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            #if args.test_only is not True:
            test_dict["loss"] = loss_avg.avg
            test_dict_details["loss_center_dist"] = loss_dict["loss_center_dist"]
            test_dict_details["loss_center_angle"] = loss_dict["loss_center_angle"]
            if args.mask_pred:
                test_dict_details["loss_mask_ce"] = loss_dict["loss_mask_ce"]
                test_dict_details["loss_dice"] = loss_dict["loss_dice"]
            test_dict_details["loss_sem_cls"] = loss_dict["loss_sem_cls"]

            #if args.test_only is not True:


        curr_iter += 1
        
        #Val/Test 的 loss_avg
        #loss_normal_avg += loss_dict["loss_normal"]
        loss_center_dist_avg += loss_dict["loss_center_dist"]
        loss_center_angle_avg += loss_dict["loss_center_angle"]
        if args.mask_pred:
            loss_mask_ce_avg += loss_dict["loss_mask_ce"]
            loss_dice_avg += loss_dict["loss_dice"]
        loss_sem_cls_avg += loss_dict["loss_sem_cls"]

   
        if is_primary() and curr_iter % len(dataset_loader) == 0:
            print("="*20)
            print("loss_center_dist_avg: %.5f" % (loss_center_dist_avg / len(dataset_loader)))
            print("loss_center_angle_avg: %.5f" % (loss_center_angle_avg / len(dataset_loader)))
            if args.mask_pred:
                print("loss_mask_ce_avg: %.5f" % (loss_mask_ce_avg / len(dataset_loader)))
                print("loss_dice_avg: %.5f" % (loss_dice_avg / len(dataset_loader)))
            print("loss_sem_cls_avg: %.5f" % (loss_sem_cls_avg / len(dataset_loader)))
            
            if args.test_only is True:
                
                k1 = k_num_wrong_more
                k2 = k_num_wrong_less
                with open(args.plane_result_dir + "all_bad_pls_scan.txt", 'w') as f:
                    f.write("k_pred_num_right: \n")
                    f.write(str(len(dataset_loader) - k1 - k2) + "\n")
                    f.write("k_pred_num_wrong: \n")
                    f.write(str(k1+k2) + "\n")
                    f.write("k_num_wrong_more: \n")
                    f.write(str(k1) + "\n")
                    f.write("k_num_wrong_less: \n")
                    f.write(str(k2) + "\n")
                    f.write("============ pred more planes ===========\n")
                    for name in wrong_more_name:
                        f.write(str(name) + "\n")
                    f.write("============ pred less planes ===========\n")
                    for name in wrong_less_name:
                        f.write(str(name) + "\n\n")
                    
                    f.write("============ pred right planes(==assign) ===========\n")
                    for name in label_right_name:
                        f.write(str(name) + "\n\n")

                    f.write("*"*20 + "\n")
                    f.write("pl_center_loss_avg:" + "\n")
                    f.write("%.5f" % (c_avg / len(dataset_loader)) + "\n")
                    f.write("pl_center_loss_avg_right:" + "\n")
                    f.write("%.5f" % (c_avg_right / len(label_right_name) + 1) + "\n")
                    f.write("pl_label_right_num(==assign):" + "\n")
                    f.write(str(k_label_right))
                    f.write("*"*20)
                f.close()
                        
            
            
            
        if is_primary():
            
            if args.test_only is True:
                logger.log_scalars(test_dict, curr_iter, prefix="Test/")
                test_angle_dict = {}
                test_angle_dict["loss_center_dist"] = loss_dict["loss_center_dist"]
                test_angle_dict["loss_center_angle"] = loss_dict["loss_center_angle"]
                test_angle_dict["loss_sem_cls"] = loss_dict["loss_sem_cls"]

                logger.log_scalars(test_angle_dict, curr_iter, prefix="Test_details/")
            
            else:
                logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Val_details/")
                logger.log_scalars(test_dict, curr_iter, prefix="Val/")

        barrier()


    return point_calculator



@torch.no_grad()
def evaluate_pt(
    args,
    curr_epoch,
    model,
    criterion_pt,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):
   
    plane_calculator = True
    curr_iter = 0
    #max_iters = 300
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    loss_point_dist_avg = 0
    loss_point_angle_avg = 0
    
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    #all_params = count_parameters(model)
    #print(all_params*4/1024/1024)
    #assert 1==0, "11111111111111"
    if args.dataset_name == "building3d":
        data_path = "building3d/scans/"
    elif args.dataset_name == "roofpc3d":
        data_path = "roofpc3d/scans_test/"
    else:
        print("Unvalid dataset name!")

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        #curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            if key != "scan_name":
                batch_data_label[key] = batch_data_label[key].to(net_device)
        
        #optimizer.zero_grad()
        inputs = {
            "scan_name": batch_data_label["scan_name"],
            "point_clouds": batch_data_label["point_clouds"],
            "point_clouds_normalized": batch_data_label["point_clouds_normalized"],
            # "rg_center_normalized": batch_data_label["rg_center_normalized"],
            # "rg_center_present": batch_data_label["rg_center_present"],
            
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        
        #flops = FlopCountAnalysis(model, inputs)
        #print(flops.total()*4/1024/1024/1024)
        #assert 1==0, "2222222222222"
        
        outputs, _ = model(inputs)
        loss, loss_dict = criterion_pt(outputs, batch_data_label)
        loss_str = ""
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_avg.update(loss_reduced.item())
        loss_str = f"Loss {loss_avg.avg:0.2f};"
        

        if args.test_only is True:
            
            outputs_points = outputs["point_normalized"][0].cpu().numpy()
            #outputs_xyz = outputs["point_xyz"][0].cpu().numpy()
            #src_xyz = inputs["point_clouds"]  #outputs["src_xyz"][0].cpu().numpy()
            print(inputs["scan_name"][0])
            pt_cloud = np.loadtxt(data_path + inputs["scan_name"][0] + ".txt")
            print(pt_cloud)

            outputs_offset = outputs["point_offset"][0].cpu().numpy()
            outputs_label = outputs["point_label"][0].cpu().numpy()
            
            center_normalized = batch_data_label["gt_plane_center_normalized"][0].cpu().numpy()
            center_label = batch_data_label["gt_plane_sem_cls_label"][0].cpu().numpy()
            ngt = batch_data_label["gt_plane_present"][0].sum(-1).cpu().numpy()
            ngt = int(ngt)
            center_normalized = center_normalized[:ngt, :]
            center_label = center_label[:ngt]

            print("---------------------------------")
            print("".join(batch_data_label["scan_name"]))
            fname = "".join(batch_data_label["scan_name"])
    
            with open(os.path.join(args.point_result_dir, fname) + "_pred_point.txt", "w") as f1:
                for id, p in enumerate(outputs_points):
                    xyz = pt_cloud[id, :]
                    
                    label = outputs_label[id]
                    line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + ' ' + str(label)
                    #line = str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + ' ' + str(label)
                    f1.write(line)
                    if id < len(outputs_points) - 1:
                        f1.write('\n')
            f1.close()
            
            with open(os.path.join(args.point_result_dir, fname) + "_center_norm.txt", "w") as f2:
                for j, item in enumerate(center_normalized):
                    line = str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(center_label[j])
                    f2.write(line)
                    if j < len(center_normalized) - 1:
                        f2.write("\n")
            f2.close()
                        
            

        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:    
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print("*"*20)
            print("loss_point_dist")
            print(loss_dict["loss_point_offset_dist"])
            print("loss_point_angle")
            print(loss_dict["loss_point_offset_angle"])

            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
            print("*"*20)

            test_dict = {}
            test_dict_details = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            #if args.test_only is not True:
            test_dict["loss"] = loss_avg.avg
            test_dict_details["loss_point_dist"] = loss_dict["loss_point_offset_dist"]
            test_dict_details["loss_point_angle"] = loss_dict["loss_point_offset_angle"]

            if args.test_only is True:
                logger.log_scalars(test_dict_details, curr_iter, prefix="Test_details/")
                logger.log_scalars(test_dict, curr_iter, prefix="Test/")
            else:
                logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Val_details/")
                logger.log_scalars(test_dict, curr_iter, prefix="Val/")

        curr_iter += 1
        
        #Val/Test 的 loss_avg
        loss_point_dist_avg += loss_dict["loss_point_offset_dist"]
        loss_point_angle_avg += loss_dict["loss_point_offset_angle"]

            
    if is_primary():
        #if args.test_only is True:
        print("loss_point_dist_avg")
        print(loss_point_dist_avg / len(dataset_loader))
        print("loss_point_angle_avg")
        print(loss_point_angle_avg / len(dataset_loader))      
        
        barrier()

    return plane_calculator



