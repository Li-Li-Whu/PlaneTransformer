import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import time

def square_distance(src, dst):
    """
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist

def farthest_point_sample(xyz, npoint):
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

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1,...DN]
    Return:
        new_points:, indexed points data, [B, D1,...DN, C] 
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    # 检查 idx 是否越界
    # print("idx max:", idx.max().item(), "idx min:", idx.min().item())  # 确保 idx 在 [0, N-1] 范围内
    assert idx.min() >= 0, f"发现负索引！idx.min() = {idx.min()}"
    assert idx.max() < points.shape[1], f"索引越界！idx.max() = {idx.max()}, points.shape[1] = {points.shape[1]}"

    
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    # print(f"B:{B}")
    # print(f"device: {device}")
    # print(f" batch_indices.shape: {batch_indices.shape}")
    # print(batch_indices)
    # print(f"idx.sahpe: {idx.shape}")
    # print(idx)

    # assert 1==0, "1111"

    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    
    radius = torch.tensor(radius, device=device)
  
    N = torch.tensor(N, device=device)
  
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # print("before:")
    # print(group_idx)
    for b in range(B):
        ind = group_idx[b, :, 0] == N
        group_idx[b, ind, :] = sqrdists[b, ind, :].sort(dim=-1)[1][:, :nsample]
            # group_idx_b[sqrdists[b, :, :] > (radius*2) ** 2] = N

            # group_idx_b = torch.arange(N, dtype=torch.long).to(device).view(1, N).repeat([S, 1])
            # group_idx_b[sqrdists[b, :, :] > (radius*2) ** 2] = N
            # group_idx[b, :, :] = group_idx_b.sort(dim=-1)[0][:, :nsample]

    # print("after:")
    # print(group_idx)
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])

    mask = group_idx == N
    group_idx[mask] = group_first[mask]


    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: Number of point for FPS
        radius: Radius of ball query
        nsample: Number of point for each ball query
        xyz: Old feature of points position data, [B, N, C]
        points: New feature of points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points  = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        ''' 
        PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius_list: list of float32 -- search radius in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp_list: list of list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, sum_k{mlp[k][-1]}) TF tensor
        '''
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  #(B, S, C)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  #(B, S, K)
            grouped_xyz = index_points(xyz, group_idx)  # (B, N, C) -> (B, S, K, C )
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)  #(B, N, D) -> (B, S, K ,D)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) #
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat



class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # print(f"***********xyz1.shape: {xyz1.shape}")
            # print(xyz1)
            # print(f"***********xyz2.shape: {xyz2.shape}")
            # print(xyz2)
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points

'''
class PointNet2(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel,[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256],[128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 256, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512 + 256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        self.conv_o1 = nn.Conv1d(128, 64, 1)
        self.bn_o1 = nn.BatchNorm1d(64)
        self.conv_o2 = nn.Conv1d(64, 32, 1)
        self.bn_o2 = nn.BatchNorm1d(32)
        self.offset_branch = nn.Conv1d(32, 3, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        fea = xyz
        l0_fea = fea.permute(0, 2, 1)
        l0_xyz = l0_fea

        l1_xyz, l1_fea = self.sa1(l0_xyz, l0_fea)
        l2_xyz, l2_fea = self.sa2(l1_xyz, l1_fea)
        l3_xyz, l3_fea = self.sa3(l2_xyz, l2_fea)
        l4_xyz, l4_fea = self.sa4(l3_xyz, l3_fea)

        l3_fea = self.fp4(l3_xyz, l4_xyz, l3_fea, l4_fea)
        l2_fea = self.fp3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.fp2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.fp1(l0_xyz, l1_xyz, None, l1_fea)

        x = self.drop1(self.bn1(self.conv1(l0_fea)))
        o = F.relu(self.bn_o2(self.conv_o2(F.relu(self.bn_o1(self.conv_o1(x))))))
        pred_offset = self.offset_branch(o).permute(0, 2, 1)  #(B, N, C)


        return pred_offset   
'''

# # torchstat
# model = PointNet2()
# stat = stat(model, (8192, 3))
# print(stat)

# #thop
# model = PointNet2()
# input = torch.randn(1, 2048, 3)
# macs, params = profile(model, inputs=(input, ))
# print("MACs: {}".format(macs))
# print("Params: {}".format(params))








