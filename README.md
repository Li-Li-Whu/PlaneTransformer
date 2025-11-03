# PlaneTransformer
This is the PyToch implementation of the following manuscript:
> A transformer-based roof plane segmentation approach for airborne LiDAR point clouds
>
> Siyuan You, Guozheng Xu, Pengwei Zhou, Yubing Wei, Jian Yao, Li Li
>
This manuscript has been accepted by TGRS Journal.

## Datasets
To evaluate the performance of the propsed PlaneTransformer, we train and test the model on a synthetic and a real-world dataset, which can be derived from [DeepRoofPlane](https://github.com/Li-Li-Whu/DeepRoofPlane.). 

 
## Usage
The environment requires torch1.12.1+cu113 with python=3.9. The requirements list is shown as follow:
- scikit-learn
- pyyaml
- libboost
- plyfile 
- open3d 
- scipy 
- matplotlib 
- trimesh
- tensorboardx 

If the above requirements are incomplete, you can install the missed dependencies according to the compilation errors.

## Data preparation
Before traning the model, you need to prepare the data files as follows:
1) Generate the train.txt and test.txt.
The `building3d_train.txt` and `building3d_test.txt` are provided in `./building3d/meta_data/`, 
The `roofpc3d_train.txt` and `roofpc3d_test.txt` are provided in `./roofpc3d/meta_data/`.
2) copy the `[scans_id].txt` to `scans` floder below the corresponding datasets folder.
3) Generate the Groud Truth of the planar centers by running the command:
```shell script
cd TransformerPlane
python cal_plane_center.py
``` 
The data files are organized as follows.
```
PlaneTransformer
├── building3d
│   ├── meta_data
│   │   ├── building3d_train.txt
|   |   ├── building3d_test.txt
|   ├── scans
|   |   ├── [scans_id].txt & [scans_id]_planes.txt
├── roofpc3d
│   ├── meta_data
│   │   ├── roofpc3d_train.txt
|   |   ├── roofpc3d_test.txt
|   ├── scans
|   |   ├── [scans_id].txt & [scans_id]_planes.txt
|   ├── scans_test
|   |   ├── [scans_id].txt & [scans_id]_planes.txt
```


## Train and test
Then, you can run main.py for training and testing the model. Notably, the commands with parameter settings required by the model are stored in the `./scripts/scripts.sh` file. The model checkpoints will be stored in the `./outputs` folder.
An example for training command:
```shell script
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name roofpc3d --ngpus 1 --nqueries 16 --enc_nlayers 2 --dec_nlayers 8 --max_epoch 180 --batchsize_per_gpu 10 --save_separate_checkpoint_every_epoch 30 --checkpoint_dir outputs/roofpc3d_q_16_dec_8
``` 
An example for testing command (--test_ckpt $test_model_dir$):
```shell script
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name roofpc3d --ngpus 1 --nqueries 16 --enc_nlayers 2 --dec_nlayers 8  --batchsize_per_gpu 1 --checkpoint_dir outputs/roofpc3d_q_16_dec_8 --test_ckpt checkpoints/checkpoint.pth --test_only
```
* If you want post-processing with different distance metrics for point assignment, add "--dist_cmd $norm_dist||center_dist||plane_dist$" to testing command.
* If you want post-processing with different k neareat numbers for plane estimation, add "--q_k_num $int$" to testing command.



## Results
* The model outputs: pred_planar_centers, pred_point_offsets, plane_segments will be saved in `./result_pred_pl`, `./result_pred_pt` and `./result_plane_seg`.
Several examples are provided in these folders.

* For the quantative evaluation of the experimental results, you can check ./evaluation_res.

<!-- ## Pretrained Model
We provide pretrained models trained on Roofpc3d and Building3d dataset, respectively. Download them [here](https://pan.baidu.com/). -->


## Citation

If you find our work useful for your research, please consider citing our paper.
> A transformer-based roof plane segmentation approach for airborne LiDAR point clouds
>
> Siyuan You, Guozheng Xu, Pengwei Zhou, Yubing Wei, Jian Yao, Li Li
>

```shell script
@article{you2025transformer,
  title={A Transformer-based Roof Plane Segmentation Approach for Airborne LiDAR Point Clouds},
  author={You, Siyuan and Xu, Guozheng and Zhou, Pengwei and Wei, Yubing and Yao, Jian and Li, Li},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
``` 


In addition, if you use the Roofpc3d and Building3d dataset, please also consider citing the following paper:

```shell script
@article{li2024boundary,
  title={A boundary-aware point clustering approach in Euclidean and embedding spaces for roof plane segmentation},
  author={Li, Li and Li, Qingqing and Xu, Guozheng and Zhou, Pengwei and Tu, Jingmin and Li, Jie and Li, Mingming and Yao, Jian},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={218},
  pages={518--530},
  year={2024},
  publisher={Elsevier}
}
``` 

## Contact:
Li Li (li.li@whu.edu.cn)