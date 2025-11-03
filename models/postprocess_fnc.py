import os
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


import numpy as np
import torch
import torch.nn as nn

import glob


#def query_ball_from_center(pred_xyz, pred_center, query_radius):
def query_ball_from_center(pred_xyz, pred_center, query_K):
    #KNN
    knn = NearestNeighbors(n_neighbors=query_K)
    knn.fit(pred_xyz)
    distances, indices = knn.kneighbors(pred_center)
    point_around_center = indices
    #print(indices.shape[0])
    #print(indices)
    '''
    #query_ball_with_radius
    point_around_center = []
    for id, ct in enumerate(pred_center):
        dist = np.linalg.norm(pred_xyz - ct, axis=1)
        indice = np.where(dist < query_radius)
        #print(indice)
        point_around_center.append(indice)
    '''
    return point_around_center


def plane_estimation(point_around_center, tgt_xyz, ax1):
    planes_coefficients = []
    qiyi = False
    for id, indice in enumerate(point_around_center):
        #print(len(indice[0]))
        #if len(indice[0]) <= 3:
        if len(indice) <= 3:
            continue
        else:
            #indice = indice[0]
            #print(indice[0])
            pts = tgt_xyz[indice, :3]
            #print("*")
            #print(pts)
            x2, y2, z2 = [], [], []
            for i, p in enumerate(pts):
                x2.append(p[0])
                y2.append(p[1])
                z2.append(p[2])

            point_num = len(x2)
            #print('points number:', point_num)
            #print('*' * 20)
            # print(x2, y2, z2)

            # 创建系数矩阵A
            A = np.zeros((3, 3))
            for i in range(0, point_num):

                A[0, 0] = A[0, 0] + x2[i] ** 2
                A[0, 1] = A[0, 1] + x2[i] * y2[i]
                A[0, 2] = A[0, 2] + x2[i]
                A[1, 0] = A[0, 1]
                A[1, 1] = A[1, 1] + y2[i] ** 2
                A[1, 2] = A[1, 2] + y2[i]
                A[2, 0] = A[0, 2]
                A[2, 1] = A[1, 2]
                A[2, 2] = point_num

            # 创建b
            b = np.zeros((3, 1))
            for i in range(0, point_num):
                b[0, 0] = b[0, 0] + x2[i] * z2[i]
                b[1, 0] = b[1, 0] + y2[i] * z2[i]
                b[2, 0] = b[2, 0] + z2[i]

            # 求解X
            #print(A)
            det_A = np.linalg.det(A)
            if det_A == 0:
                print("矩阵是奇异的，行列式为零")
                A_inv = np.linalg.pinv(A)
                qiyi = True
            else:
                A_inv = np.linalg.inv(A)
            X = np.dot(A_inv, b)
            #print(X[0, 0], X[1, 0], X[2, 0])
            pl_coeff = np.squeeze(X)
            planes_coefficients.append(pl_coeff)

            x_p = np.linspace(min(x2), max(x2), 5)  # 从min(x2), max(x2)按等差数列生成10个数
            y_p = np.linspace(min(y2), max(y2), 5)
            # print('x_p, y_p:', x_p)
            x_p, y_p = np.meshgrid(x_p, y_p)
            C = -1
            z_p = (pl_coeff[0] * x_p + pl_coeff[1] * y_p + pl_coeff[2]) / (-C)  # / 1
            #print('---------')
            #print('z_p:', z_p)
            #print(z_p)
            ax1.plot_wireframe(x_p, y_p, z_p, rstride=1, cstride=1)

    return planes_coefficients, qiyi


def point_assignment(planes_coefficients, pred_point, pred_center, src_xyz_normalization, dist_cmd):
    src_xyz_norm = src_xyz_normalization[:, :3]
    label = []
    for i, p in enumerate(src_xyz_norm):
        #print(p)
        pt_pred = pred_point[i, :3]
        #print(planes_coefficients)
        if dist_cmd == "norm_dist":
            plane_dist = np.zeros(len(planes_coefficients), dtype=np.float32)
            center_dist = np.zeros(len(planes_coefficients), dtype=np.float32)

            for idx, pl in enumerate(planes_coefficients):
                ct = pred_center[idx, :3]
                A = pl[0]
                B = pl[1]
                C = -1.0
                D = pl[2]
                # print(pl)
                pl_dist = np.abs(A * p[0] + B * p[1] + C * p[2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
                ct_dist = np.linalg.norm(pt_pred - ct)
                plane_dist[idx] = pl_dist
                center_dist[idx] = ct_dist

            pl_d_norm = plane_dist / np.linalg.norm(plane_dist)
            ct_d_norm = center_dist / np.linalg.norm(center_dist)

            indice = np.argsort(pl_d_norm + ct_d_norm)
            min_idx = indice[0]

        else:
            min_dist = 9999.9
            min_idx = -1
            for idx, pl in enumerate(planes_coefficients):
                ct = pred_center[idx, :3]
                A = pl[0]
                B = pl[1]
                C = -1.0
                D = pl[2]
                # print(pl)
                pl_dist = np.abs(A * p[0] + B * p[1] + C * p[2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
                ct_dist = np.linalg.norm(pt_pred - ct)
                if dist_cmd == "sum_dist":
                    distance = pl_dist + ct_dist
                elif dist_cmd == "center_dist":
                    distance = ct_dist
                elif dist_cmd == "plane_dist":
                    distance = pl_dist
                else:
                    print("invalid dist! please check again!")
                    assert 1==0
                if distance < min_dist:
                    min_dist = distance
                    min_idx = idx

        label.append(min_idx)
    label = np.array(label).reshape(-1, 1)
    points_assign = np.concatenate((src_xyz_normalization, label), axis=1)

    return points_assign



def centroid_normalization(point_cloud):
    # calculate mean center point
    centroid = np.mean(point_cloud, axis=0)
    point_offset = point_cloud - centroid

    # calculate max distance from centroid
    max_dist = np.max(np.sqrt(np.sum(point_offset ** 2, axis=1)))

    # normalization
    point_norm = point_offset / max_dist


    return point_norm


def readTxt(textfile):
    x2, y2, z2, l2 = [], [], [], []
    label = []
    with open(textfile, 'r') as f:
        for line in f.readlines():
            line.replace('\n', '')
            k = 0
            for itm in label:
                if float(line.split(' ')[-1]) == itm:
                    k = 1
            if k == 0:
                label.append(float(line.split(' ')[-1]))

            x2.append(float(line.split(' ')[0]))
            y2.append(float(line.split(' ')[1]))
            z2.append(float(line.split(' ')[2]))
            l2.append(float(line.split(' ')[-1]))
        print(label)
        print(len(label))

    f.close()
    return x2, y2, z2, l2, label



def points_assignment_function(args):
    pred_pl_path = args.plane_result_dir
    pred_pt_path = args.point_result_dir

    files_all = glob.glob(os.path.join(pred_pt_path, "*_pred_point.txt"))
    empty_file = []
    qiyi_file = []
    dist_cmd = args.dist_cmd #"plane_dist"
    q_k_num = args.q_k_num #70
    for file in files_all:
        #print(file)
        scan_name = file.split('.')[0].split('/')[-1].split('_')[0]
        print(scan_name)
        pl_path = pred_pl_path + scan_name + '_pred_test.txt'
        pt_path = pred_pt_path + scan_name + '_pred_point.txt'

        #pt_tgt = np.loadtxt(file)
        pl_pred = np.loadtxt(pl_path)
        pt_pred = np.loadtxt(pt_path)
        pt_tgt = pt_pred[:, 3:7]

        if pl_pred.shape[0] == 0:
            empty_file.append(scan_name)
            continue

        if pl_pred.ndim == 1:
            pl_pred = np.expand_dims(pl_pred, axis=0)

        pt_tgt_norm = centroid_normalization(pt_tgt[:, :3])
        pt_tgt_norm = np.concatenate((pt_tgt_norm[:, :], pt_tgt[:, 3, None]), axis=1)


        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        Colors = ['r', 'g', 'b', 'm', 'y', 'c', 'k', 'w']
        ax1.scatter3D(pt_tgt_norm[:, 0], pt_tgt_norm[:, 1], pt_tgt_norm[:, 2], c='y', marker='.')


        point_around_center = query_ball_from_center(pt_pred[:, :3], pl_pred[:, :3], q_k_num)


        planes_coefficients, qiyi = plane_estimation(point_around_center, pt_tgt_norm[:, :3], ax1)

        if qiyi:
            qiyi_file.append(scan_name)


        points_assign = point_assignment(planes_coefficients, pt_pred[:, :3], pl_pred[:, :3], pt_tgt_norm, dist_cmd)

        result_name = os.path.join(args.plane_seg_result_dir, scan_name + '.txt')
        with open(result_name, 'w') as f:
            for id, p in enumerate(points_assign):
                line = str(p[0]) +  ' ' + str(p[1]) +  ' ' + str(p[2]) +  ' ' + str(p[3]) + ' ' + str(p[4])
                f.write(line)
                if id < len(points_assign) - 1:
                    f.write('\n')
        f.close()

        #plt.show()
        plt.close(fig1)


    print("*************************")
    print("empty_file:")
    for file in empty_file:
        print(file)
    print("*************************")
    print("qiyi_file:")
    for file in qiyi_file:
        print(file)
    print("finished!")