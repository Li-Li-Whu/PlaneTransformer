# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment

#torch.set_printoptions(threshold=np.inf) 

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def check_nan(tensor, message="Check point"):
    if torch.isnan(tensor).any():
        print(f"NaN detected at {message}")
        raise ValueError(f"NaN detected at {message}")

class Matcher(nn.Module):
    def __init__(self, cost_class, cost_center, cost_mask, cost_dice, mask_pred):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_center = cost_center
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.mask_pred = mask_pred
        #self.cost_objectness = cost_objectness
        

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_plane_sem_cls_label"].shape[1]
    
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs["sem_cls_prob"]
        gt_plane_sem_cls_labels = (
            targets["gt_plane_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )

    
        pred_cls_neg = 1 - outputs["sem_cls_prob"]
        pred_cls_prob = torch.cat((pred_cls_neg, pred_cls_prob), dim=-1)
        #print(pred_cls_prob.size())
        
        class_mat = -torch.gather(pred_cls_prob, 2, gt_plane_sem_cls_labels)
        
        if not self.mask_pred:
        # center cost: batch x nqueries x ngt
            center_mat = outputs["center_dist"].detach()
        #print(center_mat.size())
        
        else:
            center_mat = outputs["center_dist"].detach()
            #mask_cost_map: (batch, n_queries, n_tgt)
            cost_mask = torch.zeros((batchsize, nqueries, ngt), dtype=torch.float32, device=class_mat.device)
            cost_dice = torch.zeros((batchsize, nqueries, ngt), dtype=torch.float32, device=class_mat.device)
            for b in range(batchsize):
                mask_pred = outputs["mask_heatmaps"][b].T
                mask_tgt = targets["mask_tgt"][b]
                cost_mask[b] = batch_sigmoid_ce_loss_jit(mask_pred, mask_tgt)
                cost_dice[b] = batch_dice_loss_jit(mask_pred, mask_tgt)
            
        if not self.mask_pred:
            final_cost = (
                self.cost_class * class_mat 
                + self.cost_center * center_mat 
            )#+ self.cost_objectness * objectness_mat + self.cost_center * center_mat 
        else:
            final_cost = (
                self.cost_class * class_mat 
                + self.cost_center * center_mat
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )#+ self.cost_objectness * objectness_mat + self.cost_center * center_mat 
        
        final_cost = final_cost.detach().cpu().numpy()

        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )

        
        for b in range(batchsize):
            assign = []

            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


#loss_function for predicted planar centers(center_xyz, label_class)
class SetCriterion_plane(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict, mask_pred):
        super().__init__()
        self.mask_pred = mask_pred
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)
        
        if not self.mask_pred:
            self.loss_functions = {
                "loss_sem_cls": self.loss_sem_cls,
                "loss_center_dist": self.loss_center_dist,
                "loss_center_angle": self.loss_center_angle,
                # this isn't used during training and is logged for debugging.
                # thus, this loss does not have a loss_weight associated with it.
                # this loss for logging difference between predicted planar centers' num and targets planar centers' num
                "loss_cardinality": self.loss_cardinality,
            }
        else:
            self.loss_functions = {
                "loss_sem_cls": self.loss_sem_cls,
                "loss_center_dist": self.loss_center_dist,
                "loss_center_angle": self.loss_center_angle,
                "loss_mask": self.loss_mask,
                # this isn't used during training and is logged for debugging.
                # thus, this loss does not have a loss_weight associated with it.
                # this loss for logging difference between predicted planar centers' num and targets planar centers' num
                "loss_cardinality": self.loss_cardinality,
            }


  
    def loss_mask(self, outputs, targets, assignments):
        batch_size = outputs["mask_heatmaps"].shape[0]
        pred_mask = outputs["mask_heatmaps"]
        
        tgt_mask = targets["mask_tgt"]
        loss_mask_ce = []
        loss_dice = []
        for b in range(batch_size):
            pred_id = assignments["assignments"][b][0]
            tgts_id = assignments["assignments"][b][1]
            
            #assert 1==0, '111'
            pt_mask = pred_mask[b][:, pred_id].T
            gt_mask = tgt_mask[b][tgts_id, :]
            num_masks = torch.tensor(pt_mask.shape[0])
            loss_mask_ce.append(sigmoid_ce_loss_jit(pt_mask, gt_mask, num_masks))
            loss_dice.append(dice_loss_jit(pt_mask, gt_mask, num_masks))
        
        return {
            "loss_mask_ce": torch.stack(loss_mask_ce).mean(-1),
            "loss_dice": torch.stack(loss_dice).mean(-1),
        }


    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are positive centers
        # Cardinality is the error between predicted #positive centers and ground truth centers

        #pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "positive" (which is the last class)
        #pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        pred_prob = outputs["sem_cls_prob"]
        pred_objects = (pred_prob > 0.5).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}
    

    def loss_sem_cls(self, outputs, targets, assignments):
        pred_logits = outputs["sem_cls_logits"]
        gt_plane_label = torch.gather(
            targets["gt_plane_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )

        gt_plane_label[assignments["proposal_matched_mask"].int() == 0] = (
            0
        ) #binary: 0 || multi: pred_logits.shape[-1] - 1
        
        #pred_logits = pred_logits.squeeze(-1)
        
        #print(pred_logits.size())
        #print(gt_plane_label.size())
        gt_plane_label = gt_plane_label.float()
        loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            gt_plane_label.unsqueeze(-1),
            pos_weight=torch.tensor(2, device=pred_logits.device),
            reduction="mean",
        )
        #assert 1==0, "111111111"
        '''
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_plane_label,
            self.semcls_percls_weights,
            reduction="mean",
        )
        '''
        return {"loss_sem_cls": loss}

    
    def loss_center_dist(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_planes_replica"] > 0:

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_planes"] > 0:
                center_loss /= targets["num_planes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center_dist": center_loss}


    def loss_center_angle(self, outputs, targets, assignments):
        batch_size = outputs["query_offset"].shape[0]
        angle_diff_bs = 0
        for b in range(batch_size):
            pred_id = assignments["assignments"][b][0]
            tgts_id = assignments["assignments"][b][1]
 
            center_tgts = targets["gt_plane_center_normalized"][b, ...]
            pred_offset = outputs["query_offset"][b, ...]
            pred_xyz = outputs["query_xyz"][b, ...]
        
            angle_diff = 0
            for id, itm in enumerate(pred_id):
                p_xyz = pred_xyz[itm, :3]
                p_off = pred_offset[itm, :3] 
                c_tgt = center_tgts[tgts_id[id], :3]
                c_tgt = c_tgt - p_xyz

                p_off_norm = torch.norm(p_off, p=2, dim=-1)
                p_off = p_off / (p_off_norm + 1e-10)
                c_tgt_norm = torch.norm(c_tgt, p=2, dim=-1)
                c_tgt = c_tgt / (c_tgt_norm + 1e-10)
                angle_diff += -(p_off * c_tgt).sum(-1)
            
            angle_diff /= len(pred_id)
            angle_diff_bs += angle_diff

        angle_diff_bs /= batch_size

        return {"loss_center_angle": angle_diff_bs}



    def single_output_forward(self, outputs, targets, interm = False):
        
        output_center = outputs["center_normalized"]
        center_dist = torch.cdist(
            output_center, targets["gt_plane_center_normalized"], p=1
        )
        outputs["center_dist"] = center_dist

        if interm == False:
           print("*"*20)
           #print(targets["scan_name"])
           print("*"*20)

        
        assignments = self.matcher(outputs, targets)

    
        losses = {}
        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses, assignments


    def forward(self, outputs, targets):
        nactual_gt = targets["gt_plane_present"].sum(axis=1).long()
        num_planes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_planes"] = num_planes
        targets[
            "num_planes_replica"
        ] = nactual_gt.sum().item()  # number of planes on this worker for dist training
        
        
        #print(outputs["outputs"]["mask_heatmaps"])
        #print(outputs["outputs"]["mask_heatmaps"].shape)

        #assert 1==0, '222'
        interm = False
        loss, loss_dict, assignments = self.single_output_forward(outputs["outputs"], targets, interm)


        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm = True
                interm_loss, interm_loss_dict, _ = self.single_output_forward(
                    outputs["aux_outputs"][k], targets, interm
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        #print(loss)
        return loss, loss_dict, assignments


#loss_function for predicted point-wise offset 
class SetCriterion_point(nn.Module):
    def __init__(self, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        
        self.loss_weight_dict = loss_weight_dict

      
        self.loss_functions = {
            "loss_point_offset_dist": self.loss_point_offset_dist,
            "loss_point_offset_angle": self.loss_point_offset_angle,
        }

    
    def loss_point_offset_dist(self, outputs, targets):
        # point_offset_dist is the L1 norm distance between predicted offset and ground truth offset for each point
        
        batch_size = targets["gt_plane_center"].shape[0]
        npoints = outputs["point_normalized"].shape[1]

        tgt_centers = targets["gt_plane_center_normalized"]
        centers_label = targets["gt_center_sem_cls_label"]
        ngt = targets["gt_plane_present"].sum(axis=1).long()

        points_xyz = outputs["point_xyz"]
        points_offset = outputs["point_offset"]
        points_label = outputs["point_label"]
        
        #centers_label = centers_label[..., :ngt]
        

        dist_all_bs = 0
        for b in range(batch_size):
            print("*"*20)
            print("points-loss")
            print(b)
            pts_xyz = points_xyz[b, :, :3]#.squeeze(1)
            pts_offset = points_offset[b, :, :3]
            tgt_center = tgt_centers[b, :, :3]#.squeeze(1)

            pts_label = points_label[b, :]
            cpt_label = centers_label[b, :ngt[b]]

            dist_per_bs = 0
            
            for id, l in enumerate(cpt_label):
                print("*"*10)
                mask = pts_label == l
                pred_p = pts_xyz[mask]
                print(pred_p.shape[0])
                tgts_c = tgt_center[id, :3].repeat(pred_p.shape[0], 1)
                center_v = tgts_c - pred_p
                offset_v = pts_offset[mask]
                norm_diff = offset_v - center_v
                dist_loss = torch.sum(torch.sum(abs(norm_diff), dim=-1))

                print("dist_per_plane_loss_mean")
                print(dist_loss/pred_p.shape[0])
                print("*"*10)
                dist_per_bs += dist_loss #loss_dist
            
            dist_per_bs /= npoints
            print("dist_per_bs_mean")
            print(dist_per_bs)
            #loss_per_bs /= npoints
            dist_all_bs += dist_per_bs

        if batch_size > 1:
            dist_all_bs /= batch_size
            print("="*10)
            print("="*10)
            print("dist_all_bs_mean")
            print(dist_all_bs)
            print("="*10)
            print("="*10)
        
        return {"loss_point_offset_dist": dist_all_bs}

    
    def loss_point_offset_angle(self, outputs, targets):
        # point_offset_angle is the angle between vector A (org_point to shifted point) and vector B (org_point to ground truth planar center)
        #outputs_points (bs x np x ch)
        batch_size, npoints = (
            outputs["point_normalized"].shape[0], 
            outputs["point_normalized"].shape[1],
        )

        tgt_centers = targets["gt_plane_center_normalized"]
        centers_label = targets["gt_center_sem_cls_label"]
        #planes numbers for each batch
        ngt = targets["gt_plane_present"].sum(axis=1).long()

        points_xyz = outputs["point_xyz"]
        points_label = outputs["point_label"]
        points_offset = outputs["point_offset"]
        
        angle_all_bs = 0
        for b in range(batch_size):
            print("*"*20)
            print(b)
            pts_xyz = points_xyz[b, :, :3]
            pts_offset = points_offset[b, :, :3]
            tgt_center = tgt_centers[b, :, :3]
            
            pts_label = points_label[b, :]
            cpt_label = centers_label[b, :ngt[b]]
            
            angle_per_bs = 0
            for id, l in enumerate(cpt_label):
                print("*"*10)
                mask = pts_label == l
                pred_p = pts_xyz[mask]
                print(pred_p.shape[0])
                pt_offset = pts_offset[mask]
                gt_center = tgt_center[id, :3].repeat(pred_p.shape[0], 1)
                gt_center = gt_center - pred_p
                
                offset_norm = torch.norm(pt_offset, p=2, dim=1)
                center_norm = torch.norm(gt_center, p=2, dim=1)

                pt_offset = pt_offset / (offset_norm.unsqueeze(-1) + 1e-10)
                gt_center = gt_center / (center_norm.unsqueeze(-1) + 1e-10)
                
                angle_dist = - (pt_offset * gt_center).sum(-1)

                angle_loss = torch.sum(angle_dist)
                print("angle_per_plane_loss_mean")
                print(angle_loss / angle_dist.shape[0])
                print("*"*10)
                angle_per_bs += angle_loss 
            
            angle_per_bs /= npoints
            print("angle_per_bs_mean")
            print(angle_per_bs)
            angle_all_bs += angle_per_bs
        
        if batch_size > 1:
            angle_all_bs /= batch_size
            print("*"*10)
            print("*"*10)
            print("angle_all_bs_mean")
            print(angle_all_bs)
            print("*"*10)
            print("*"*10)
        
        return {"loss_point_offset_angle": angle_all_bs}



    def forward(self, outputs, targets):
        losses = {}
        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses




def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_center=args.matcher_center_cost,
        cost_mask=args.matcher_mask_cost,
        cost_dice=args.matcher_dice_cost,
        mask_pred=args.mask_pred,
    )
    #center_loss_calculation
    loss_weight_dict_pl = {
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_center_dist_weight": args.loss_center_dist_weight,
        "loss_center_angle_weight": args.loss_center_angle_weight,
        # "loss_mask_ce_weight": args.loss_mask_ce_weight,
        # "loss_dice_weight": args.loss_dice_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
    }
    #"loss_center_dist_weight": args.loss_center_dist_weight,
    #"loss_center_angle_weight": args.loss_center_angle_weight,

    criterion_pl = SetCriterion_plane(matcher, dataset_config, loss_weight_dict_pl, args.mask_pred)
    
    #point_loss_calculaiton
    loss_weight_dict_pt = {
        "loss_point_offset_dist_weight": args.loss_point_dist_weight,
        "loss_point_offset_angle_weight": args.loss_point_angle_weight, 
    }
    criterion_pt = SetCriterion_point(dataset_config, loss_weight_dict_pt)
    
    return criterion_pt, criterion_pl
