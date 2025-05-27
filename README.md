# DEN: Depth Enhancement Network for 3-D Object Detection With the Fusion of mmWave Radar and Vision in Autonomous Driving

## Abstract
In the realm of autonomous driving, precise and robust 3-D perception is paramount. Multimodal fusion for 3-D object detection is crucial for improving accuracy, generalization, and robustness in autonomous driving. In this article, we introduce the depth enhancement network (DEN), an innovative camera-radar fusion framework that generates an accurate depth estimation for 3-D object detection. To overcome the limitations caused by the lack of spatial information in an image, DEN estimates image depth using accurate radar points. Furthermore, to extract more comprehensive and fine-grained scene depth information, we present an innovative label optimization strategy (LOS) that enhances label density and quality. DEN achieves an 18.78% reduction in mean absolute error (MAE) and a 12.8% decrease in root mean-square error (RMSE) for depth estimation. Additionally, it improves 3-D object detection accuracy by 0.8% compared to the baseline model. Under low visibility conditions, DEN demonstrates a 6.7% reduction in MAE and a 9.6% reduction in RMSE compared to the baseline. These improvements demonstrated its robustness and enhanced performance under challenging conditions.


## Getting Started

### Installation
```shell
# clone repo
git clone https://github.com/Wangwx-code/DEN

cd DEN

# setup conda environment
conda env create --file DEN.yaml
conda activate DEN

# install dependencies
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
mim install mmcv==1.6.0
mim install mmsegmentation==0.28.0
mim install mmdet==2.25.2

cd mmdetection3d
pip install -v -e .
cd ..

python setup.py develop  # GPU required
```

### Data preparation
**Step 0.** Download [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).

**Step 1.** Symlink the dataset folder to `./data/nuScenes/`.
```
ln -s [nuscenes root] ./data/nuScenes/
```

**Step 2.** Create annotation file. 
This will generate `nuscenes_infos_{train,val}.pkl`.
```
python scripts/gen_info.py
```

**Step 3.** Generate ground truth depth.  
*Note: this process requires LiDAR keyframes.*
```
python scripts/gen_depth_gt.py
```

**Step 4.** Generate radar point cloud in perspective view. 
You can download pre-generated radar point cloud [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/youngseok_kim_kaist_ac_kr/EcEoswDVWu9GpGV5NSwGme4BvIjOm-sGusZdCQRyMdVUtw?e=OpZoQ4).  
*Note: this process requires radar blobs (in addition to keyframe) to utilize sweeps.*  
```
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_pv.py  # transform to camera coords
```

**Step 5.** Generate enhanced lidar point cloud. 
```
python scripts/gen_lidar_enhance.py
```


The folder structure will be as follows:
```
DEN
├── data
│   ├── nuScenes
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
|   |   ├── depth_gt
|   |   ├── radar_bev_filter  # temporary folder, safe to delete
|   |   ├── radar_pv_filter
|   |   ├── v1.0-trainval
```

### Training and Evaluation
**Training**
```
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**  
*Note: use `-b 1 --gpus 1` to measure inference time.*
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```

## Model Zoo
All models use 4 keyframes and are trained without CBGS.  
All latency numbers are measured with batch size 1, GPU warm-up, and FP16 precision.

| Method | Input | Backbone | Image Size | NDS↑ | mAP↑ | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ | FPS↑ |
|--------|-------|----------|------------|------|------|-------|-------|-------|-------|-------|------|
| BEVDet | C | R50 | 256×704 | 39.2 | 31.2 | 0.691 | 0.272 | 0.523 | 0.909 | 0.247 | - |
| CenterFusion  | C+R | DLA34 | 448×800 | 45.3 | 33.2 | 0.649 | ​**0.263**​ | 0.535 | 0.540 | ​**0.142**​ | - |
| CRAFT  | C+R | DLA34 | 448×800 | 51.7 | 41.1 | 0.494 | 0.276 | 0.454 | 0.486 | 0.176 | - |
| CRN  | C+R | R18 | 256×704 | 54.0 | 44.7 | ​**0.524**​ | 0.286 | 0.567 | 0.278 | 0.181 | 18.80 |
| PETR  | C | R101 | 900×1600 | 44.2 | 37.0 | 0.711 | 0.267 | 0.383 | 0.865 | 0.201 | - |
| MVFusion  | C+R | R101 | 900×1600 | 45.5 | 38.0 | 0.675 | 0.258 | 0.372 | 0.833 | 0.196 | - |
| BEVFormer  | C | R101 | 900×1600 | 51.7 | 41.6 | 0.673 | 0.274 | 0.372 | 0.394 | 0.198 | - |
| BEVDepth  | C | R101 | 512×1408 | 53.5 | 41.2 | 0.565 | 0.266 | ​**0.358**​ | 0.331 | 0.190 | - |
| ​**LOS**​ | C+R | R18 | 256×704 | 54.3 | 45.3 | 0.527 | 0.285 | 0.564 | 0.283 | 0.178 | 18.80 |
| ​**DFM**​ | C+R | R18 | 256×704 | 54.5 | 45.4 | ​**0.524**​ | 0.286 | 0.556 | 0.279 | 0.178 | 18.41 |
| ​**DEN**​ | C+R | R18 | 256×704 | ​**54.8**​ | ​**45.5**​ | 0.525 | 0.284 | 0.534 | ​**0.272**​ | 0.178 | 18.41 |

## Features
- [ ] BEV segmentation checkpoints 
- [ ] BEV segmentation code 
- [x] 3D detection checkpoints 
- [x] 3D detection code 
- [x] Code release 


## Acknowledgement
This project is based on excellent open source projects:
- [CRN](https://github.com/youngskkim/CRN)


## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{wang2025den,
    title={DEN: Depth Enhancement Network for 3-D Object Detection With the Fusion of mmWave Radar and Vision in Autonomous Driving},
    author={Wang, Wenxiang and Han, Jianping and Jiang, Zhongmin and Zhou, Zhiyuan and Wu, Yingxiao},
    booktitle={IEEE Internet of Things Journal},
    volume={12},
    number={10},
    pages={14420--14430},
    year={2025},
    doi={10.1109/JIOT.2025.3525899}
}
```
