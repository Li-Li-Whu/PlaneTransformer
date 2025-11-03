#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


'''
roofpc3d
'''
#training with single gpu/multi gpus
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name roofpc3d --ngpus 1 --nqueries 16 --enc_nlayers 2 --dec_nlayers 8 --max_epoch 180 --matcher_cls_cost 1.0 --matcher_center_cost 5.0 --matcher_objectness_cost 1.0 --loss_no_object_weight 0.20 --loss_sem_cls_weight 1.0 --loss_center_dist_weight 5.0 --loss_center_angle_weight 1.0 --loss_point_dist_weight 1.0 --loss_point_angle_weight 1.0 --batchsize_per_gpu 10 --save_separate_checkpoint_every_epoch 30 --checkpoint_dir outputs/roofpc3d_q_16_dec_8
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset_name roofpc3d --ngpus 3 --nqueries 16 --enc_nlayers 2 --dec_nlayers 8 --max_epoch 180 --matcher_cls_cost 1.0 --matcher_center_cost 5.0 --matcher_objectness_cost 1.0 --loss_no_object_weight 0.20 --loss_sem_cls_weight 1.0 --loss_center_dist_weight 5.0 --loss_center_angle_weight 1.0 --loss_point_dist_weight 1.0 --loss_point_angle_weight 1.0 --batchsize_per_gpu 10 --save_separate_checkpoint_every_epoch 30 --checkpoint_dir outputs/roofpc3d_q_16_dec_8
#testing
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name roofpc3d --ngpus 1 --nqueries 16 --enc_nlayers 2 --dec_nlayers 8 --matcher_cls_cost 0.5 --matcher_center_cost 1.0 --loss_no_object_weight 1.0 --loss_sem_cls_weight 1.0 --loss_center_dist_weight 1.0 --loss_center_angle_weight 1.0 --loss_point_dist_weight 1.0 --loss_point_angle_weight 1.0 --batchsize_per_gpu 1 --checkpoint_dir outputs/roofpc3d_q_16_dec_8 --test_ckpt checkpoints/checkpoint.pth --test_only


'''
building3d
'''
#training with single gpu/multi gpus
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name building3d --ngpus 1 --nqueries 32 --enc_nlayers 2 --dec_nlayers 8 --max_epoch 180 --matcher_cls_cost 1.0 --matcher_center_cost 5.0 --matcher_objectness_cost 1.0 --loss_no_object_weight 0.20 --loss_sem_cls_weight 1.0 --loss_center_dist_weight 5.0 --loss_center_angle_weight 1.0 --loss_point_dist_weight 1.0 --loss_point_angle_weight 1.0 --batchsize_per_gpu 10 --save_separate_checkpoint_every_epoch 30 --checkpoint_dir outputs/building3d_q_32_dec_8
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset_name building3d --ngpus 3 --nqueries 32 --enc_nlayers 2 --dec_nlayers 8 --max_epoch 180 --matcher_cls_cost 1.0 --matcher_center_cost 5.0 --matcher_objectness_cost 1.0 --loss_no_object_weight 0.20 --loss_sem_cls_weight 1.0 --loss_center_dist_weight 5.0 --loss_center_angle_weight 1.0 --loss_point_dist_weight 1.0 --loss_point_angle_weight 1.0 --batchsize_per_gpu 10 --save_separate_checkpoint_every_epoch 30 --checkpoint_dir outputs/building3d_q_32_dec_8
#testing
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name building3d --ngpus 1 --nqueries 32 --enc_nlayers 2 --dec_nlayers 8 --matcher_cls_cost 1.0 --matcher_center_cost 5.0 --matcher_objectness_cost 1.0 --loss_no_object_weight 0.20 --loss_sem_cls_weight 1.0 --loss_center_dist_weight 5.0 --loss_center_angle_weight 1.0 --loss_point_dist_weight 1.0 --loss_point_angle_weight 1.0 --batchsize_per_gpu 1 --checkpoint_dir outputs/building3d_q_32_dec_8 --test_ckpt checkpoints/checkpoint.pth --test_only



tensorboard --logdir='outputs/roofpc3d_q_32_dec_8' --port=6016 
tensorboard --logdir='outputs/building3d_q_32_dec_8' --port=6026





