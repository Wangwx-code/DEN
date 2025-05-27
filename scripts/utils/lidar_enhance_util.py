import numpy as np
import os
import os.path as osp
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix

from scipy.ndimage import convolve

def average_of_neighbors(matrix):
    cube = np.pad(matrix, ((1, 1), (0, 0)))
    # 定义卷积核
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
    # 计算邻居的总和
    neighbor_sum = convolve(cube, kernel, mode='wrap')[1: -1, :]
    
    # 计算邻居的数量（排除0值）
    neighbor_count = convolve(np.where(cube != 0, 1, 0), kernel, mode='wrap')[1: -1, :]
    
    # 计算平均值（排除0值）
    # average = np.where(neighbor_count != 0, neighbor_sum / neighbor_count, 0)
    average = np.nan_to_num(neighbor_sum / neighbor_count)
    
    return np.where(matrix > 0, matrix, average)

def window_pooling(inputs, mode='average'):
    assert len(inputs.shape) == 2
    # TODO: max min
    if mode == 'average':
        return average_of_neighbors(inputs)
    else:
        raise Exception('mode值不能为: {}'.format(mode))

def move(cube, direction):
    if direction == 'up':
        return np.pad(cube[1:, :], ((0, 1), (0, 0)), 'constant')
    if direction == 'down':
        return np.pad(cube[:-1, :], ((1, 0), (0, 0)),'constant')
    if direction == 'left':
        return np.pad(cube[:, 1:], ((0, 0), (0, 1)),'constant')
    if direction == 'right':
        return np.pad(cube[:, :-1], ((0, 0), (1, 0)),'constant')
    

def cube_pooling(inputs, mode='max'):
    a = move(inputs, 'left')
    b = move(inputs, 'right')
    c = move(inputs, 'up')
    d = move(inputs, 'down')
    e = move(c, 'left')
    f = move(c, 'right')
    g = move(d, 'left')
    h = move(d, 'right')
    # i = move(c, 'up')
    # j = move(d, 'down')
    # k = move(i, 'left')
    # l = move(i, 'right')
    # m = move(j, 'left')
    # n = move(j, 'right')
    cube = np.stack([a, b, c, d, e, f, g, h], axis=-1)
    masked_A = np.ma.masked_equal(cube, 0)
    if mode == 'max':
        masked_A = np.max(masked_A, axis=-1)
    elif mode == 'min':
        masked_A = np.min(masked_A, axis=-1)
    elif mode == 'average':
        masked_A = np.average(masked_A, axis=-1)
    else:
        raise Exception('mode值不能为: {}'.format(mode))
    return np.where(inputs > 0, inputs, masked_A)

def stick_pooling(inputs, mode='max'):
    c = move(inputs, 'up')
    d = move(inputs, 'down')
    cube = np.stack([c, d], axis=-1)
    masked_A = np.ma.masked_equal(cube, 0)
    if mode == 'max':
        masked_A = np.max(masked_A, axis=-1)
    elif mode == 'min':
        masked_A = np.min(masked_A, axis=-1)
    elif mode == 'average':
        masked_A = np.average(masked_A, axis=-1)
    else:
        raise Exception('mode值不能为: {}'.format(mode))
    return np.where(inputs > 0, inputs, masked_A)


class MyNuScenesExplorer(NuScenesExplorer):
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
    
    def render_sample_data(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None,
                           verbose: bool = True,
                           show_panoptic: bool = False,
                           lidar_num_enhance=-2,
                            ele_range=(30, -10),
                            resolution=0.1,
                            downsample_step=3) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        if show_lidarseg:
            show_panoptic = False
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                if show_lidarseg or show_panoptic:
                    gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                    assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                    # Ensure that lidar pointcloud is from a keyframe.
                    assert sd_record['is_key_frame'], \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                    assert nsweeps == 1, \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                        'be set to 1.'

                    # Load a single lidar point cloud.
                    pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                     nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            if lidar_num_enhance > -2:
                points = lidar_interpolation(points.T, lidar_num_enhance, ele_range=ele_range, resolution=resolution, downsample_step=downsample_step).T
            dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            if sensor_modality == 'lidar' and (show_lidarseg or show_panoptic):
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                semantic_table = getattr(self.nusc, gt_from)
                # Load labels for pointcloud.
                if lidarseg_preds_bin_path:
                    sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
                    lidarseg_labels_filename = lidarseg_preds_bin_path
                    assert os.path.exists(lidarseg_labels_filename), \
                        'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                        'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
                else:
                    if len(semantic_table) > 0:
                        # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                        lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                            self.nusc.get(gt_from, sample_data_token)['filename'])
                    else:
                        lidarseg_labels_filename = None

                if lidarseg_labels_filename:
                    # Paint each label in the pointcloud with a RGBA value.
                    if show_lidarseg or show_panoptic:
                        if show_lidarseg:
                            colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                        self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)
                        else:
                            colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                              self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)

                        if show_lidarseg_legend:

                            # If user does not specify a filter, then set the filter to contain the classes present in
                            # the pointcloud after it has been projected onto the image; this will allow displaying the
                            # legend only for classes which are present in the image (instead of all the classes).
                            if filter_lidarseg_labels is None:
                                if show_lidarseg:
                                    # Since the labels are stored as class indices, we get the RGB colors from the
                                    # colormap in an array where the position of the RGB color corresponds to the index
                                    # of the class it represents.
                                    color_legend = colormap_to_colors(self.nusc.colormap,
                                                                      self.nusc.lidarseg_name2idx_mapping)
                                    filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                                else:
                                    # Only show legends for stuff categories for panoptic.
                                    filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

                            if filter_lidarseg_labels and show_panoptic:
                                # Only show legends for filtered stuff categories for panoptic.
                                stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
                                filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

                            create_lidarseg_legend(filter_lidarseg_labels,
                                                   self.nusc.lidarseg_idx2name_mapping,
                                                   self.nusc.colormap,
                                                   loc='upper left',
                                                   ncol=1,
                                                   bbox_to_anchor=(1.05, 1.0))
                else:
                    print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                          'from the ego vehicle instead.'.format(self.nusc.version))

            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            # ax.imshow(points)
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()


def cartesian_to_polar(x, y, z):
    temp = x ** 2 + y ** 2
    r = np.sqrt(temp + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(temp))
    return r, theta, phi

def polar_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def point_to_sphere(points, resolution=0.1):
    r, theta, phi = cartesian_to_polar(points[:, 0], points[:, 1], points[:, 2])

    theta, phi = theta * 180 / np.pi, phi * 180 / np.pi
    up, down = np.max(phi), np.min(phi)

    width = int(360 // resolution)
    height = int((up - down) // resolution) + 1

    shape = (height, width)
    sphere = np.zeros(shape)
    i = ((up - phi) // resolution).astype(np.int16)
    j = (theta // resolution).astype(np.int16)
    sphere[i, j] = r

    return sphere, up

def sphere_to_point(sphere):
    pass

def lidar_interpolation(points, num_enhance, resolution=0.1, downsample_step=3):
    """
    points: shape()
    """
    point_map, up = point_to_sphere(points, resolution=resolution)
    if num_enhance == -1:
        while np.any(point_map == 0):
            point_map = window_pooling(point_map, "average")
    elif num_enhance >= 0:
        for _ in range(num_enhance):
            point_map = window_pooling(point_map, "average")
    else:
        raise Exception('num_hance值不能为: {}'.format(num_enhance))
    if downsample_step != 1:
        mask = np.full(point_map.shape, True)
        mask[::downsample_step, ::downsample_step] = False
        point_map[mask] = 0
    i, j = np.nonzero(point_map)
    r = point_map[i, j]
    phi = (up - (i * resolution)) * np.pi / 180
    theta = j * resolution * np.pi / 180
    points = np.stack(polar_to_cartesian(r, theta, phi), axis=-1)
    return points