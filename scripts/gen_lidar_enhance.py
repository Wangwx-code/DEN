import os
from multiprocessing import Pool

import mmcv
import numpy as np
from tqdm import tqdm

from utils.lidar_enhance_util import lidar_interpolation


SPLIT = 'v1.0-trainval'
DATA_PATH = 'data/nuScenes'
OUT_PATH = 'lidar_enhanced'
info_paths = ['data/nuScenes/nuscenes_infos_train.pkl', 'data/nuScenes/nuscenes_infos_val.pkl']

# SPLIT = 'v1.0-test'
# DATA_PATH = 'data/nuScenes/v1.0-test'
# OUT_PATH = 'radar_bev_filter_test'
# info_paths = ['data/nuScenes/nuscenes_infos_test.pkl']

RADAR_CHAN = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
              'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

lidar_key="LIDAR_TOP"

def worker(info):
    lidar_path = info['lidar_infos'][lidar_key]['filename']
    points = np.fromfile(os.path.join(DATA_PATH, lidar_path),
                            dtype=np.float32,
                            count=-1).reshape(-1, 5)[..., :3]
    points = lidar_interpolation(points, num_enhance=-1, ele_range=(30, -5), resolution=0.08, downsample_step=4)

    file_name = os.path.split(info['lidar_infos']['LIDAR_TOP']['filename'])[-1]
    points.astype(np.float32).flatten().tofile(os.path.join(DATA_PATH, OUT_PATH, file_name))

if __name__ == '__main__':
    po = Pool(24)
    mmcv.mkdir_or_exist(os.path.join(DATA_PATH, OUT_PATH))
    for info_path in info_paths:
        infos = mmcv.load(info_path)
        for info in infos:
            po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()