import numpy as np
from pyquaternion import Quaternion

def is_points_in_cubes(points, cube_positions, cube_rotations, cube_sizes):
    # 将点的坐标转换到长方体的局部坐标系中
    points_in_cube_frame = points[:, None, :] - cube_positions

    # 创建旋转矩阵
    rotation_matrices = np.array([
        Quaternion(rotation).rotation_matrix for rotation in cube_rotations
    ])

    # 应用旋转矩阵的逆矩阵
    points_in_cube_frame = np.einsum('ijk,pik->pij', rotation_matrices, points_in_cube_frame)

    # 判断点是否在长方体内部
    return np.any(np.all(np.abs(points_in_cube_frame) <= cube_sizes[None, :, :] / 2, axis=-1), axis=-1)