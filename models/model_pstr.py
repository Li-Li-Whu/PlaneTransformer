# Copyright (c) Facebook, Inc. and its affiliates.
import os
import math
from functools import partial
import time

import numpy as np
import torch
import torch.nn as nn

from utils.pc_util import scale_points, shift_scale_points

from models.basic_class import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (TransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoderLayer)
                                

from models.pointnet2 import *

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids



class PlaneProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding planes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz):
        center_normalized = query_xyz + center_offset
        return center_normalized   #, center_unnormalized


    def compute_objectness_and_cls_prob(self, cls_logits):
        '''
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob
        '''
        
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls
        cls_prob = torch.nn.functional.sigmoid(cls_logits)   
        objectness_prob = cls_prob[..., :] 
        return cls_prob[..., :], objectness_prob
        
    
   

class Model_PSTR(nn.Module):
    """
    Main PSTR model. Consists of the following learnable sub-models
    - Pointnet2_SA_MSG: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N'x D matrix of N' point features.

    - Pointnet2_FP: Input is a N'xD matrix of N' point features
                Output is a N x D_out matrix of N point features.

    - query computation: samples a set of B coordinates from the N' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD plane features
                Takes N'xD features from the encoder and BxD query features.
    - mlp_heads: Predicts possible plane parameters and classes from the BxD plane features
    """

    def __init__(
        self,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256, #512+512
        decoder_dim=256,
        position_embedding="fourier", 
        in_channel=3,
        mlp_dropout=0.3,
        num_queries=64,
        use_normal = False,
        query_with_normal = False,
        use_center_proposal = False,
        mask_pred = False,
        not_use_encoder = False,

    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.use_normal = use_normal
        self.query_with_normal = query_with_normal
        self.use_center_proposal = use_center_proposal
        self.mask_pred = mask_pred
        self.not_use_encoder = not_use_encoder
        
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel,[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256],[128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 256, 512]])
        
        self.encoder = encoder

        self.fp4 = PointNetFeaturePropagation(512+512 + 256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])


        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        
        if not self.use_center_proposal: # and not self.mask_pred:
            self.conv_o1 = nn.Conv1d(128, 64, 1)
            self.bn_o1 = nn.BatchNorm1d(64)
            self.conv_o2 = nn.Conv1d(64, 32, 1)
            self.bn_o2 = nn.BatchNorm1d(32)
            self.offset_branch = nn.Conv1d(32, 3, 1)
        
        self.init_weights()
        
        self.decoder = decoder

        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=128,
            hidden_dims=[256, 256],
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, query_with_normal=query_with_normal, normalize=False
        )


        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

    
        self.plane_processor = PlaneProcessor(dataset_config)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[int(decoder_dim/2), int(decoder_dim/4)],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the plane instance center(positive or negative)
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls) # planar center class    #output_dim=dataset_config.num_semcls
        # geometry of the plane insatnce center(center_xyz)
        center_head = mlp_func(output_dim=3)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head)
        ]

        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims): # rg_centers, rg_centers_num,  
        # if not self.use_center_proposal:
        # #Useing only furthest point sampling for center queries
        encoder_xyz = encoder_xyz.contiguous()
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # else:
        #     bs = rg_centers.shape[0]
        #     q_num = self.num_queries
        #     query_xyz = torch.zeros((bs, q_num, 3), dtype=torch.float32, device=encoder_xyz.device)
        #     for b in range(bs):
        #         rg_num = int(rg_centers_num[b, :].sum(-1))
        #         #print(rg_num)
        #         if rg_num <= self.num_queries:
        #             query_xyz[b, 0:rg_num, :] = rg_centers[b, 0:rg_num, :]
        #         else:
        #             query_xyz[b, 0:q_num, :] = rg_centers[b, 0:q_num, :]

            
            #else:

        #print(query_xyz.shape)       
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)  

        fea = xyz
        l0_fea = fea.permute(0, 2, 1)
        l0_xyz = l0_fea

        l1_xyz, l1_fea = self.sa1(l0_xyz, l0_fea)
        l2_xyz, l2_fea = self.sa2(l1_xyz, l1_fea)
        l3_xyz, l3_fea = self.sa3(l2_xyz, l2_fea)
        l4_xyz, l4_fea = self.sa4(l3_xyz, l3_fea)
        #l4_xyz: batch x 3 x npoints
        #l4_fea: batch x channel x npoints


        l3_fea = self.fp4(l3_xyz, l4_xyz, l3_fea, l4_fea)
        l2_fea = self.fp3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.fp2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.fp1(l0_xyz, l1_xyz, None, l1_fea)

        
        #encoder need: npoints x batch x channel
        l0_fea = l0_fea.permute(2, 0, 1)
        enc_xyz, enc_features, enc_ind = self.encoder(
            l0_fea, xyz=l0_xyz
        )
        #print(enc_xyz.shape)
        #print(l0_xyz.shape)
        #print(enc_features.shape)
        #print(l0_fea.shape)
        
        if self.not_use_encoder:
            enc_xyz = l0_xyz
            enc_feratures = l0_fea 
        else:
            enc_xyz, enc_features, enc_ind = self.encoder(
                l0_fea, xyz=l0_xyz
            )
        
        
        if not self.use_center_proposal and not self.mask_pred:
            enc_fea = enc_features.permute(1, 2, 0)
            interm_fea = self.drop1(self.bn1(self.conv1(enc_fea)))
            #interm_fea = self.drop1(self.bn1(self.conv1(l0_fea)))
            offset_fea = F.relu(self.bn_o2(self.conv_o2(F.relu(self.bn_o1(self.conv_o1(interm_fea))))))
            pred_offset = self.offset_branch(offset_fea).permute(0, 2, 1)

            return enc_xyz, enc_features, pred_offset 
        
        else:
            return enc_xyz, enc_features
    
    def get_point_offset_prediction(self, src_point, pred_offset, point_src):
        """
        Parameters:
            src_point: batch x npoints x 3 tensor of src XYZ coords
            pred_offset: batch x npoints x 3
        """
        point_offset = pred_offset

        point_normalized = src_point[..., :3] + pred_offset
        
        point_xyz = src_point[..., :3]
        point_label = src_point[..., -1] 
        point_prediction = {
            "point_normalized": point_normalized.contiguous(),
            "point_offset": point_offset,
            "point_xyz": point_xyz,
            "point_label": point_label,
            "src_xyz": point_src,
        }

        return point_prediction

    def get_plane_predictions(self, query_xyz, point_cloud_dims, plane_features, enc_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            plane_features: num_layers x num_queries x batch x channel
        """
        # plane_features change to (num_layers x batch) x channel x num_queries
        plane_features = plane_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            plane_features.shape[0],
            plane_features.shape[1],
            plane_features.shape[2],
            plane_features.shape[3],
        )
        plane_features_ = plane_features.reshape(num_layers * batch, channel, num_queries)

        
        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](plane_features_).transpose(1, 2)
        center_offset = (
            (self.mlp_heads["center_head"](plane_features_).sigmoid().transpose(1, 2) - 0.5) * 2
        )


        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)

        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
  

        outputs = []
        enc_features = enc_features.permute(1, 0, 2)
        query_features = plane_features.permute(0, 1, 3, 2)
        for l in range(num_layers):
            
            center_normalized = self.plane_processor.compute_predicted_center(
                center_offset[l], query_xyz[:, :, :3]
            )
        
            qry_features = query_features[l]
        
            
            masks_heatmaps= []
            for i in range(batch):
                masks_heatmaps.append(enc_features[i] @ qry_features[i].T)
            
            masks_heatmaps = torch.cat(masks_heatmaps)
            #rint(masks_pred.shape)
            #rint(masks_pred)
            masks_heatmaps = masks_heatmaps.reshape(batch, enc_features.shape[1], num_queries)
            
            #assert 1==0, '111'
            #rint(masks_pred.shape)
            #rint(masks_pred)
            #assert 1==0, 'plane_features'
            
            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                semcls_prob, object_prob = self.plane_processor.compute_objectness_and_cls_prob(cls_logits[l])
            
            if not self.mask_pred:
                plane_prediction = {
                    "sem_cls_logits": cls_logits[l], 
                    "center_normalized": center_normalized.contiguous(),
                
                    "query_xyz": query_xyz[:, :, :].contiguous(),
                    "query_offset": center_offset[l],
                
                    "sem_cls_prob": semcls_prob,
                    #"objectness_prob": object_prob,  
                }
            else:
                plane_prediction = {
                    "sem_cls_logits": cls_logits[l], 
                    "mask_heatmaps": masks_heatmaps.contiguous(),
                    "center_normalized": center_normalized.contiguous(),
                
                    "query_xyz": query_xyz[:, :, :].contiguous(),
                    "query_offset": center_offset[l],
                
                    "sem_cls_prob": semcls_prob,
                    #"objectness_prob": object_prob,  
                }

               
                
            outputs.append(plane_prediction)
        
        final_outputs = outputs[-1]
        aux_outputs = outputs[:-1]
        

        return {
            "outputs": final_outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs): 
        point_src = inputs["point_clouds"]
        point_clouds = inputs["point_clouds_normalized"]
        # rg_centers = inputs["rg_center_normalized"]
        # rg_centers_num = inputs["rg_center_present"]

        pc = point_clouds[:, :, :3]
        
        if not self.use_center_proposal and not self.mask_pred:
            enc_xyz, enc_features, pred_offset = self.run_encoder(pc)
        else:
            enc_xyz, enc_features = self.run_encoder(pc)
        
        
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        #print("encoder features: npoints x batch x channel")
        #print(enc_features.size())
        # encoder features: npoints x batch x channel
        # encoder xyz:  batch x 3 x npoints

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        
        enc_xyz = enc_xyz.permute(0, 2, 1)
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims) #rg_centers[..., :3], rg_centers_num

        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)
        #print("enc_pos.size()")
        #print(enc_pos.size())
        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)

        tgt = torch.zeros_like(query_embed)
        plane_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]
        
        plane_predictions = self.get_plane_predictions(
            query_xyz, point_cloud_dims, plane_features, enc_features
        )

        if not self.use_center_proposal and not self.mask_pred:
            point_predictions = self.get_point_offset_prediction(
                point_clouds[:, :, :], pred_offset, point_src[..., :3],
            )
            
            return point_predictions, plane_predictions

        else:
            return plane_predictions





def build_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.enc_dim,
        nhead=args.enc_nhead,
        dim_feedforward=args.enc_ffn_dim,
        dropout=args.enc_dropout,
        activation=args.enc_activation,
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=args.enc_nlayers
    )
    
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder


def build_PSTR(args, dataset_config):
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model_PSTR(
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        use_normal=args.use_normal,
        query_with_normal=args.query_with_normal,
        use_center_proposal=args.use_center_proposal,
        mask_pred = args.mask_pred,
        not_use_encoder = args.not_use_encoder,
    )

    output_processor = PlaneProcessor(dataset_config)

    return model, output_processor


MODEL_FUNCS = {
    "PSTR": build_PSTR,
}


def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor