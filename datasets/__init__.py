# Copyright (c) Facebook, Inc. and its affiliates.
from .roofpc3d import Roofpc3dDetectionDataset, Roofpc3dDatasetConfig
from .building3d import Building3dDetectionDataset, Building3dDatasetConfig
# from .roofNTNU import RoofNTNUDetectionDataset, RoofNTNUDatasetConfig


DATASET_FUNCTIONS = {
    "roofpc3d": [Roofpc3dDetectionDataset, Roofpc3dDatasetConfig],
    "building3d": [Building3dDetectionDataset,Building3dDatasetConfig],
    # "roofNTNU": [RoofNTNUDetectionDataset, RoofNTNUDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    dataset_dict = {
        "train": dataset_builder(
            dataset_config, 
            split_set="train", 
            root_dir=args.dataset_root_dir, 
            meta_data_dir=args.meta_data_dir, 
            use_normal=args.use_normal,
            use_semcls=args.use_semcls,
            mask_pred=args.mask_pred,
            augment=True,
        ),
        "test": dataset_builder(
            dataset_config, 
            split_set="test", 
            root_dir=args.dataset_root_dir, 
            use_normal=args.use_normal,
            use_semcls=args.use_semcls,
            mask_pred=args.mask_pred,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config
    



