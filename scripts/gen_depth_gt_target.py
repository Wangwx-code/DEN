import os
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from scripts.utils.lidar_enhance_util import lidar_interpolation


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


data_root = 'data/nuScenes'
INFO_PATHS = ['data/nuScenes/nuscenes_infos_train.pkl',
              'data/nuScenes/nuscenes_infos_val.pkl']

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]

def is_points_in_cubes(points, cube_positions, cube_rotations, cube_sizes):
    # 将点的坐标转换到长方体的局部坐标系中
    points_in_cube_frame = points[:, None, :] - cube_positions

    # 创建旋转矩阵
    rotation_matrices = np.array([
        Quaternion(rotation).inverse.rotation_matrix for rotation in cube_rotations
    ])

    # 应用旋转矩阵的逆矩阵
    points_in_cube_frame = np.einsum('ijk,pik->pij', rotation_matrices, points_in_cube_frame)

    # 判断点是否在长方体内部
    return np.any(np.all(np.abs(points_in_cube_frame) <= cube_sizes[None, :, :] / 2, axis=-1), axis=-1)

def worker(info):
    exist = True
    for i, cam_key in enumerate(cam_keys):
        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        target_path = os.path.join(data_root, 'depth_gt_target', f'{file_name}.bin')
        if not os.path.exists(target_path):
            exist = False
            # print(file_name)
    if exist:
        return
    lidar_path = info['lidar_infos'][lidar_key]['filename']
    points = np.fromfile(os.path.join(data_root, lidar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]
    points_enhanced = lidar_interpolation(points, num_enhance=-1, resolution=0.1)
    points_enhanced = np.pad(points_enhanced, ((0, 0), (0, 1)))
    points = np.concatenate((points, points_enhanced), axis=0)
    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(points.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    if len(info['ann_infos']) != 0:
        cube_translations, cube_rotations, cube_size = [], [], []
        for ann_info in info['ann_infos']:
            cube_translations.append(ann_info['translation'])
            cube_rotations.append(ann_info['rotation'])
            box_size = ann_info['size']
            # box_size[0], box_size[1] = box_size[1], box_size[0]
            cube_size.append([box_size[1], box_size[0], box_size[2]])
        cube_translations, cube_rotations, cube_size = np.array(cube_translations), np.array(cube_rotations), np.array(cube_size)
        target_mask = is_points_in_cubes(pc.points[:3, :].T, cube_translations, cube_rotations, cube_size)
        pc.points = pc.points[:, target_mask]
    else:
        pc.points = np.zeros((4, 0), dtype=pc.points.dtype)

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        img = mmcv.imread(os.path.join(data_root, info['cam_infos'][cam_key]['filename']))
        pts_img, depth = map_pointcloud_to_image(pc.points.copy(), img, cam_calibrated_sensor, cam_ego_pose)
        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        result = np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1).astype(np.float32)
        print("{}\t{}".format(file_name, result.shape))
        result.flatten().tofile(os.path.join(data_root, 'depth_gt_target', f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")
 
def print_error(value):
    print("error: ", value)

if __name__ == '__main__':
    po = Pool(24)
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt_target'))
    for info_path in INFO_PATHS:
        infos = mmcv.load(info_path)
        for info in infos:
            po.apply_async(func=worker, args=(info, ), error_callback=print_error)
    po.close()
    po.join()
