import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
import glob
import os
import time

#import shutil



def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """Input is NxC, output is num_samplexC"""
    if replace is None:
        replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

# 从txt文档读取点
def readTxt(textfile):

    x2, y2, z2, l2 = [], [], [], []
    label = []
    with open(textfile, 'r') as f:
        for line in f.readlines():
            line.replace('\n', '')
            k = 0
            for itm in label:
                if float(line.split(' ')[3]) == itm:
                    k = 1
            if k == 0:
                label.append(float(line.split(' ')[3]))
            
            x2.append(float(line.split(' ')[0]))
            y2.append(float(line.split(' ')[1]))
            z2.append(float(line.split(' ')[2]))     
            l2.append(float(line.split(' ')[3]))
        print(label)
        print(len(label))

    f.close()
    return x2, y2, z2, l2, label

DATA_PATH = 'building3d/scans/'
DATA_PATH_1 = 'roofpc3d/scans/'
DATA_PATH_2 = 'roofpc3d/scans_test/'
files_all = sorted(glob.glob(os.path.join(DATA_PATH, '*.txt')))
files_all_1 = sorted(glob.glob(os.path.join(DATA_PATH_1, '*.txt')))
files_all_2 = sorted(glob.glob(os.path.join(DATA_PATH_2, '*.txt')))




for filename in files_all:
    
    scan_name = filename.split('.')[0].split('/')[-1].split('_')[0]   #txt_path.split('.')[0].split('/')[-1]   #.split('_')[0]
    print(scan_name)
    #plane_angle_name = DATA_PATH + scan_name + '_planes_angle.txt'
    # plane_ct_name = DATA_PATH + str(scan_name) + '_ct.txt'
    res_name_p = DATA_PATH + str(scan_name) + '.txt'
    res_name_c = DATA_PATH + str(scan_name) + '_planes.txt'
    '''
    pointcloud = np.loadtxt(res_name_p)
    pc = random_sampling(pointcloud, 2048)
    with open(res_name_p, 'w') as f:
        for id, p in enumerate(pc):
            line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[3])
            f.write(line)
            if id < len(pc) - 1:
                f.write('\n')
    f.close()
    '''
    x, y, z, l, label = readTxt(res_name_p)
    
    
    with open(res_name_p, 'w') as f:
        for i in range(len(x)):
            line = str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(l[i])
            f.write(line)
            if i < len(x) - 1:
                f.write('\n')
    f.close()
    

    plane_centers = []
    for id, itm in enumerate(label):
        x2, y2, z2 = [], [], []
        it_label = itm
        for idx, itmer in enumerate(l):
            if it_label == itmer:
                x2.append(x[idx])
                y2.append(y[idx])
                z2.append(z[idx])
        print(len(x2))

        if len(x2) == len(y2) and len(x2) == len(z2):
            point_num = len(x2)
            #print('points number:', point_num)
            #print('*' * 20)
            # print(x2, y2, z2)

            x_m = sum(x2) / len(x2)
            y_m = sum(y2) / len(y2)
            z_m = sum(z2) / len(z2)

            center = (x_m, y_m, z_m)
            plane_centers.append(center)

            # plt.show()
    plane_centers = np.array(plane_centers)
    print(plane_centers.shape)
    print(plane_centers)

    with open(res_name_c, 'w') as f:
        for i, c in enumerate(plane_centers):
            line = str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ' ' + str(1.0) + ' ' + str(label[i])
            f.write(line)
            if i < len(label) - 1:
                f.write('\n')
    f.close()
    
    
    #if os.path.exists(txt_path):
    #    os.remove(txt_path)
    # if os.path.exists(plane_ct_name):
    #    os.remove(plane_ct_name)
  

for filename in files_all_1:
    
    scan_name = filename.split('.')[0].split('/')[-1].split('_')[0]   #txt_path.split('.')[0].split('/')[-1]   #.split('_')[0]
    print(scan_name)
    #plane_angle_name = DATA_PATH + scan_name + '_planes_angle.txt'
    # plane_ct_name = DATA_PATH_1 + str(scan_name) + '_ct.txt'
    res_name_p = DATA_PATH_1 + str(scan_name) + '.txt'
    res_name_c = DATA_PATH_1 + str(scan_name) + '_planes.txt'
    
    x, y, z, l, label = readTxt(res_name_p)
    
    
    with open(res_name_p, 'w') as f:
        for i in range(len(x)):
            line = str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(l[i])
            f.write(line)
            if i < len(x) - 1:
                f.write('\n')
    f.close()
    

    plane_centers = []
    for id, itm in enumerate(label):
        x2, y2, z2 = [], [], []
        it_label = itm
        for idx, itmer in enumerate(l):
            if it_label == itmer:
                x2.append(x[idx])
                y2.append(y[idx])
                z2.append(z[idx])
        print(len(x2))

        if len(x2) == len(y2) and len(x2) == len(z2):
            point_num = len(x2)
            #print('points number:', point_num)
            #print('*' * 20)
            # print(x2, y2, z2)

            x_m = sum(x2) / len(x2)
            y_m = sum(y2) / len(y2)
            z_m = sum(z2) / len(z2)

            center = (x_m, y_m, z_m)
            plane_centers.append(center)

            # plt.show()
    plane_centers = np.array(plane_centers)
    print(plane_centers.shape)
    print(plane_centers)

    with open(res_name_c, 'w') as f:
        for i, c in enumerate(plane_centers):
            line = str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ' ' + str(1.0) + ' ' + str(label[i])
            f.write(line)
            if i < len(label) - 1:
                f.write('\n')
    f.close()
    
    
    #if os.path.exists(txt_path):
    #    os.remove(txt_path)
    # if os.path.exists(plane_ct_name):
    #    os.remove(plane_ct_name)



for filename in files_all_2:
    
    scan_name = filename.split('.')[0].split('/')[-1].split('_')[0]   #txt_path.split('.')[0].split('/')[-1]   #.split('_')[0]
    print(scan_name)
    #plane_angle_name = DATA_PATH + scan_name + '_planes_angle.txt'
    # plane_ct_name = DATA_PATH_2 + str(scan_name) + '_ct.txt'
    res_name_p = DATA_PATH_2 + str(scan_name) + '.txt'
    res_name_c = DATA_PATH_2 + str(scan_name) + '_planes.txt'

    x, y, z, l, label = readTxt(res_name_p)
    
    
    with open(res_name_p, 'w') as f:
        for i in range(len(x)):
            line = str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(l[i])
            f.write(line)
            if i < len(x) - 1:
                f.write('\n')
    f.close()
    

    plane_centers = []
    for id, itm in enumerate(label):
        x2, y2, z2 = [], [], []
        it_label = itm
        for idx, itmer in enumerate(l):
            if it_label == itmer:
                x2.append(x[idx])
                y2.append(y[idx])
                z2.append(z[idx])
        print(len(x2))

        if len(x2) == len(y2) and len(x2) == len(z2):
            point_num = len(x2)
            #print('points number:', point_num)
            #print('*' * 20)
            # print(x2, y2, z2)

            x_m = sum(x2) / len(x2)
            y_m = sum(y2) / len(y2)
            z_m = sum(z2) / len(z2)

            center = (x_m, y_m, z_m)
            plane_centers.append(center)

            # plt.show()
    plane_centers = np.array(plane_centers)
    print(plane_centers.shape)
    print(plane_centers)

    with open(res_name_c, 'w') as f:
        for i, c in enumerate(plane_centers):
            line = str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ' ' + str(1.0) + ' ' + str(label[i])
            f.write(line)
            if i < len(label) - 1:
                f.write('\n')
    f.close()
    
    
    #if os.path.exists(txt_path):
    #    os.remove(txt_path)
    # if os.path.exists(plane_ct_name):
    #    os.remove(plane_ct_name)

