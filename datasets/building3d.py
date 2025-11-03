import os
import os
import sys

import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.pc_util import scale_points, shift_scale_points, random_sampling
#from models.region_growing import init_center_proposal


DATASET_ROOT_DIR = "building3d/scans" #"scannet/scans"  ## Replace with path to dataset
DATASET_METADATA_DIR = "building3d/meta_data" ## Replace with path to dataset

class Building3dDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 1
        #self.num_angle_bin = 12
        self.max_num_obj = 25
        self.max_num_rg_center = 40



class Building3dDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        meta_data_dir=None,
        num_points=2048,
        use_normal=False,
        use_semcls=False,
        mask_pred=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        assert split_set in ["train", "test"]
        self.split_set = split_set
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        
        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:6]
                    for x in os.listdir(self.data_path)
                    if x.endswith(".txt")
                ]
            )
        )
        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(meta_data_dir, f"building3d_{split_set}.txt")
            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)

            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_normal = use_normal
        self.use_semcls = use_semcls
        self.mask_pred = mask_pred

        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

    def __len__(self):
        return len(self.scan_names)

    def centroid_normalization(self, point_cloud, plane_center):
        #calculate mean center point
        centroid = np.mean(point_cloud, axis=0)
        point_offset = point_cloud - centroid
        pl_center_offset = plane_center - centroid
        # rg_center_offset = rg_center - centroid
        #rg_center_offset = rg_center - centroid

        #calculate max distance from centroid
        max_dist = np.max(np.sqrt(np.sum(point_offset ** 2, axis=1)))

        #normalization
        point_norm = point_offset / max_dist
        pl_center_norm = pl_center_offset / max_dist
        # rg_center_norm = rg_center_offset / max_dist
        #rg_center_norm = rg_center_offset / max_dist
        
        return point_norm, pl_center_norm


    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        #mesh_vertices = np.loadtxt(os.path.join(self.data_path, scan_name) + "_norm.txt")
        mesh_vertices = np.loadtxt(os.path.join(self.data_path, scan_name) + ".txt")

        instance_planes = np.loadtxt(os.path.join(self.data_path, scan_name) + "_planes.txt")
        # instance_rg_centers = np.loadtxt(os.path.join(self.data_path, scan_name) + "_ct.txt")
        # if len(instance_rg_centers) > self.dataset_config.max_num_rg_center:
        #     instance_rg_centers = instance_rg_centers[0:self.dataset_config.max_num_rg_center, :]
        # elif instance_rg_centers.ndim == 1:
        #     instance_rg_centers = instance_rg_centers[None, ...]


        #region_growing_centers = np.loadtxt(os.path.join(self.data_path, scan_name) + "_ct.txt")
        #print(region_growing_centers.shape)
        #print(region_growing_centers.ndim)
        '''
        if region_growing_centers.ndim == 1:
           region_growing_centers = region_growing_centers[None, ...]
        '''
        if instance_planes.ndim == 1:
            instance_planes = np.expand_dims(instance_planes, axis=0)
        if not self.use_normal:
            point_cloud = mesh_vertices[:, :]  
            #pcl_normal = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, :]
            #point_cloud[:, 3:] = mesh_vertices[:, 4:7]
            #pcl_normal = point_cloud[:, 3:6]

        if self.use_semcls:
            point_cloud = point_cloud[:, :]
            #pcl_normal = point_cloud[:, 3:6]
        '''
        # ------------------------------- Sampling ----------------------------
        point_clouds = random_sampling(point_cloud, self.num_points)
        '''

        if self.split_set == "train":
            replace = len(point_cloud) < 2048 #2048 #4096 #8182 #16364
            choices = np.random.choice(len(point_cloud), 2048, replace=replace)
            point_cloud = point_cloud[choices]

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        MAX_NUM_RG = self.dataset_config.max_num_rg_center
        #MAX_NUM_CTP = self.dataset_config.max_num_ctp

        #target_rg_centers = np.zeros((MAX_NUM_CTP, 3), dtype=np.float32)
        #target_rg_centers_mask = np.zeros((MAX_NUM_CTP), dtype=np.float32)
        target_planes_center = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        target_planes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)

        # target_rg_center= np.zeros((MAX_NUM_RG, 7), dtype=np.float32)
        # target_rg_mask= np.zeros((MAX_NUM_RG), dtype=np.float32)


        target_mask = np.zeros((MAX_NUM_OBJ, len(point_cloud)), dtype=np.float32)

        plane_id = np.unique(point_cloud[:, 3])
        for i, id in enumerate(plane_id):
            indice = point_cloud[:, 3]==id
            if self.mask_pred:
                center = np.mean(point_cloud[indice, :3], axis=0)
                instance_planes[i, :3] = center[:3]
                instance_planes[i, -1] = id
            target_mask[i, indice] = 1.0

        #target_rg_centers_mask[0 : region_growing_centers.shape[0]] = 1
        #target_rg_centers[0 : region_growing_centers.shape[0], :] = region_growing_centers[:, 0:3]
        target_planes_mask[0 : instance_planes.shape[0]] = 1
        target_planes_center[0 : instance_planes.shape[0], :] = instance_planes[:, 0:3]
        # target_rg_mask[0 : instance_rg_centers.shape[0]] = 1
        # target_rg_center[0 : instance_rg_centers.shape[0], :] = instance_rg_centers[:, 0:7]
        
        
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]
        
        
        pc_points = point_cloud[:, :3]
        pl_centers = target_planes_center.astype(np.float32)[:, :3]
        # rg_centers = target_rg_center.astype(np.float32)[:, :3]
        ( 
            point_cloud_normalized, 
            planes_centers_normalized, 
        ) = self.centroid_normalization(pc_points, pl_centers)
        
        point_cloud_dims_min = point_cloud_normalized.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud_normalized.max(axis=0)[:3]

        point_cloud_normalized = np.concatenate((point_cloud_normalized, point_cloud[:, -1, None]), axis=1)
        planes_centers_normalized = planes_centers_normalized * target_planes_mask[..., None]
        #rg_centers_normalized = rg_centers_normalized * target_rg_centers_mask[..., None]
        # target_rg_center[:, :3] = rg_centers_normalized[:, :3]
        # rg_centers_normalized = target_rg_center


    
        ret_dict = {}
        ret_dict["scan_name"] = scan_name
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["point_clouds_normalized"] = point_cloud_normalized.astype(np.float32)

        ret_dict["gt_plane_center"] = target_planes_center.astype(np.float32)
        ret_dict["gt_plane_center_normalized"] =  planes_centers_normalized.astype(
            np.float32
        )

        # ret_dict["rg_center_normalized"] = rg_centers_normalized.astype(np.float32)
        # ret_dict["rg_center_present"] = target_rg_mask.astype(np.float32)
        #ret_dict["rg_centers_normalized"] = rg_centers_normalized.astype(np.float32)
        

        target_centers_semcls = np.zeros((MAX_NUM_OBJ))
        target_centers_semcls[0 : instance_planes.shape[0]] = instance_planes[:, -1] #0-9, 9:no_plane
        target_planes_semcls = np.zeros((MAX_NUM_OBJ))
        target_planes_semcls[0 : instance_planes.shape[0]] = instance_planes[:, 3] #0.0

        
        ret_dict["gt_plane_sem_cls_label"] = target_planes_semcls.astype(np.int64)
        ret_dict["gt_center_sem_cls_label"] = target_centers_semcls.astype(np.int64)

        ret_dict["gt_plane_present"] = target_planes_mask.astype(np.float32)

        ret_dict["mask_tgt"] = target_mask.astype(np.float32)
        #ret_dict["gt_rg_centers_present"] = target_rg_centers_mask.astype(np.float32)


        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        return ret_dict