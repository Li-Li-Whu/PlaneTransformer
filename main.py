# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
import argparse
import os
import sys
import pickle
#os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'
import time

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import train_one_epoch, evaluate_pt, evaluate_pl
from models.model_pstr import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger
from models.postprocess_fnc import points_assignment_function

torch.autograd.set_detect_anomaly(True)

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float) #5e-4
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float) #1e-6
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####

    parser.add_argument(
        "--model_name",
        default="PSTR",
        type=str,
        help="Name of the model",
        choices=["3detr, PSTR"],
    )

    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int) #3
    parser.add_argument("--enc_dim", default=128, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)
    
    ###Query
    parser.add_argument("--query_with_normal", default=False, action="store_true") #3 / 6
    parser.add_argument("--use_center_proposal", default=False, action="store_true")
    parser.add_argument("--mask_pred", default=False, action="store_true")
    parser.add_argument("--not_use_encoder", default=False, action="store_true")


    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int) #8
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.3, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.1, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=512, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--use_semcls", default=False, action="store_true")
    #parser.add_argument("--encoder_only", default=False, action="store_true")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_cls_cost", default=1.0, type=float)
    parser.add_argument("--matcher_center_cost", default=5.0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=1.0, type=float)
    
    parser.add_argument("--matcher_mask_cost", default=5.0, type=float)
    parser.add_argument("--matcher_dice_cost", default=2.0, type=float)

    ### point_wise Loss Weights
    parser.add_argument("--loss_point_dist_weight", default=1.0, type=float)
    parser.add_argument("--loss_point_angle_weight", default=1.0, type=float)


    ### plane_wise Loss Weights
    parser.add_argument("--loss_sem_cls_weight", default=1.0, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    
    parser.add_argument("--loss_center_dist_weight", default=5.0, type=float)
    parser.add_argument("--loss_center_angle_weight", default=1.0, type=float)

    parser.add_argument("--loss_mask_ce_weight", default=5.0, type=float)
    parser.add_argument("--loss_dice_weight", default=2.0, type=float)



    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["building3d", "roofpc3d"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=8, type=int)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=200, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--val_only", default=False, action="store_true")
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--dist_cmd", default="norm_dist", type=str, help="different distance metrics for point assginment", choices=["center_dist", "plane_dist", "sum_dist", "norm_dist"])
    parser.add_argument("--q_k_num", default=50, type=int, help="the number of k nearest neighbors for plane estimation")

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--plane_result_dir", default="result_pred_pl/", type=str)
    parser.add_argument("--point_result_dir", default="result_pred_pt/", type=str)
    parser.add_argument("--plane_seg_result_dir", default="result_plane_seg/", type=str)
    parser.add_argument("--mask_heatmaps_result_dir", default="mask_heatmaps/", type=str)


    parser.add_argument("--log_every", default=1, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=30, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:6006", type=str)

    return parser


def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion_pt,
    criterion_pl,
    dataset_config,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    logger = Logger(args.checkpoint_dir)

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)
        
        aps = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion_pt,
            criterion_pl,
            dataset_config,
            dataloaders["train"],
            logger,
        )
            
        
        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        curr_iter = epoch * len(dataloaders["train"])

        if (
            epoch > 0
            and args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
            )

            
    # always evaluate last checkpoint    
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    
    if args.val_only:
        point_calculator = evaluate_pt(
            args,
            epoch,
            model,
            criterion_pt,
            dataset_config,
            dataloaders["test"],
            logger,
            curr_iter,
        )
    
        time.sleep(3)
        plane_calculator = evaluate_pl(
            args,
            epoch,
            model,
            criterion_pl,
            dataset_config,
            dataloaders["test"],
            logger,
            curr_iter,
        )
        

    

def test_model(args, model, model_no_ddp, criterion_pt, criterion_pl, dataset_config, dataloaders):
    print("2")
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)
    #print("2")
    #optimizer = build_optimizer(args, model_no_ddp)
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])


    logger = Logger(args.checkpoint_dir)
    #criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    #print("2")
    t1 = time.time()
    
    
    if not args.use_center_proposal and not args.mask_pred:
        print('================ shifted point pred ================')
        point_calculator = evaluate_pt(
            args,
            epoch,
            model,
            criterion_pt,
            dataset_config,
            dataloaders["test"],
            logger,
            curr_iter,
        )
    
    
    print('================= plane center pred =================')
    time.sleep(3)
    plane_calculator = evaluate_pl(
        args,
        epoch,
        model,
        criterion_pl,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    
    if not args.use_center_proposal and not args.mask_pred:
        print("==================== points_assignment ======================")
        print(args.point_result_dir)
        print(args.plane_result_dir)
        points_assignment_function(args)
    
    t_delta = time.time() - t1
    print("time cost:  %d min %.3f s" % (int(t_delta // 60), t_delta % 60))



def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    datasets, dataset_config = build_dataset(args)

    model, _ = build_model(args, dataset_config)

    model = model.cuda(local_rank)
    #model = nn.DataParallel(model，devise = [0,1,2])
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=0, find_unused_parameters=True
        )

    criterion_pt, criterion_pl = build_criterion(args, dataset_config)
    criterion_pt = criterion_pt.cuda(local_rank)
    criterion_pl = criterion_pl.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True #True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        #criterion = None  # faster evaluation
        print("1")
        test_model(args, model, model_no_ddp, criterion_pt, criterion_pl, dataset_config, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )

        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            criterion_pt,
            criterion_pl,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
