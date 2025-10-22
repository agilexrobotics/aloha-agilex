import math

import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import random
import torchvision.transforms as transforms
import cv2
import IPython
import pickle
from tqdm import tqdm
import glob
e = IPython.embed


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def depth_to_color_projection(depth_image, color_intrinsics, depth_intrinsics, extrinsics):
    # 获取深度图像的宽度和高度
    depth_height, depth_width = depth_image.shape[:2]

    # 创建网格坐标
    u, v = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
    u = u.flatten()
    v = v.flatten()
    depth_values = depth_image.flatten()

    # 将像素坐标转换为齐次坐标
    depth_points = np.vstack((u, v, np.ones_like(u)))

    # 将深度图像中的点转换到深度相机坐标系
    X_depth = np.linalg.inv(depth_intrinsics) @ depth_points

    # 将深度相机坐标系中的点转换到彩色相机坐标系
    X_color = extrinsics @ np.vstack((X_depth, np.ones((1, X_depth.shape[1]))))

    # 将彩色相机坐标系中的点投影到彩色图像平面
    x_color = (color_intrinsics[0, 0] * (X_color[0, :] / X_color[2, :]) + color_intrinsics[0, 2]).round().astype(int)
    y_color = (color_intrinsics[1, 1] * (X_color[1, :] / X_color[2, :]) + color_intrinsics[1, 2]).round().astype(int)

    # 创建对齐后的深度图像
    aligned_depth = np.zeros_like(depth_image)

    # 将投影后的点存储到对齐后的深度图像中
    valid_indices = (x_color >= 0) & (x_color < depth_image.shape[1]) & (y_color >= 0) & (y_color < depth_image.shape[0])
    aligned_depth[y_color[valid_indices], x_color[valid_indices]] = depth_values[valid_indices]

    return aligned_depth


def color_depth_to_point_cloud(color_image, depth_image, color_intrinsic, depth_intrinsic, color_extrinsic, depth_extrinsic):
    if not np.array_equal(color_extrinsic, depth_extrinsic):
        depth_image = depth_to_color_projection(depth_image, color_intrinsic, depth_intrinsic, np.dot(np.linalg.inv(color_extrinsic), depth_extrinsic))
        # 相机内参矩阵
        fx, fy = color_intrinsic[0][0], color_intrinsic[1][1]
        cx, cy = color_intrinsic[0][2], color_intrinsic[1][2]
    else:
        # 相机内参矩阵
        fx, fy = depth_intrinsic[0][0], depth_intrinsic[1][1]
        cx, cy = depth_intrinsic[0][2], depth_intrinsic[1][2]
    # 获取图像的宽度和高度
    height, width = depth_image.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    z = depth_image.astype(np.float32) / 1000.0  # 将深度图像转换为米

    # 计算 3D 坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 提取 color 颜色值
    b = color_image[..., 0].astype(np.float32)
    g = color_image[..., 1].astype(np.float32)
    r = color_image[..., 2].astype(np.float32)

    # 合并为点云
    point_cloud = np.stack((x, y, z, r, g, b), axis=-1)

    # 跳过深度为零的点
    valid_mask = z > 0.0
    point_cloud = point_cloud[valid_mask]

    return point_cloud


def flatten_list(target):
    return [item for sublist in target for item in sublist]


def find_all_hdf5(dataset_dir, use_ario=True):
    hdf5_files = []
    for f in os.listdir(dataset_dir):
        if use_ario:
            if f.endswith(".hdf5"):
                hdf5_files.append(os.path.join(dataset_dir, f))
            if os.path.isdir(os.path.join(dataset_dir, f)):
                hdf5_files.extend(glob.glob(os.path.join(dataset_dir, f, "*.hdf5")))
        else:
            hdf5_files.append(os.path.join(dataset_dir, f))
    # for root, dirs, files in os.walk(dataset_dir):
    #     for filename in fnmatch.filter(files, "aloha.hdf5"):
    #         hdf5_files.append(os.path.join(root, filename, "aloha.hdf5"))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def batch_sampler(batch_size, episode_len_list, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_list])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_list), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


def batch_sampler_each(batch_size, episode_len_list, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = []
    sum_dataset_episode_len_l = []
    for i, episode_len in enumerate(episode_len_list):
        begin = 0 if i == 0 else sum_dataset_len_l[-1][-1]
        sum_dataset_len = [begin]
        sum_dataset_len = sum_dataset_len + (np.cumsum(episode_len)+begin).tolist()
        sum_dataset_len_l.append(sum_dataset_len)

        begin = 0 if i == 0 else sum_dataset_episode_len_l[-1][-1]
        sum_dataset_episode_len = [i + begin + 1 for i in range(len(episode_len_list[i]))]
        sum_dataset_episode_len_l.append(sum_dataset_episode_len)

    while True:
        record = []
        batch = []
        while True:
            dataset_idx = np.random.choice(len(episode_len_list), p=sample_probs)
            episode_idx = np.random.choice(len(episode_len_list[dataset_idx]))
            if sum_dataset_episode_len_l[dataset_idx][episode_idx] in record:
                continue
            else:
                record.append(sum_dataset_episode_len_l[dataset_idx][episode_idx])
            step_idx = np.random.choice(episode_len_list[dataset_idx][episode_idx])
            batch.append(sum_dataset_len_l[dataset_idx][episode_idx]+step_idx)
            if len(batch) >= batch_size or len(batch) >= sum_dataset_episode_len_l[-1][-1]:
                break
        yield batch


def get_camera_color_statistics(path_list, scale, offset):
    r_sum = 0
    g_sum = 0
    b_sum = 0
    r_sum_sq = 0
    g_sum_sq = 0
    b_sum_sq = 0
    r_max = 0
    g_max = 0
    b_max = 0
    r_min = 255
    g_min = 255
    b_min = 255
    num_pixels = 0

    for s in tqdm(path_list, desc='calc_color_stats'):
        if type(s) is str:
            img = cv2.imread(s, cv2.IMREAD_UNCHANGED).astype(np.float64)
        else:
            img = s
        img = img * scale + offset

        # 更新均值和方差的累加值
        r_sum += np.sum(img[:, :, 0])
        g_sum += np.sum(img[:, :, 1])
        b_sum += np.sum(img[:, :, 2])
        r_sum_sq += np.sum(img[:, :, 0] ** 2)
        g_sum_sq += np.sum(img[:, :, 1] ** 2)
        b_sum_sq += np.sum(img[:, :, 2] ** 2)

        # 更新最大值和最小值
        r_max = max(r_max, np.max(img[:, :, 0]))
        g_max = max(g_max, np.max(img[:, :, 1]))
        b_max = max(b_max, np.max(img[:, :, 2]))
        r_min = min(r_min, np.min(img[:, :, 0]))
        g_min = min(g_min, np.min(img[:, :, 1]))
        b_min = min(b_min, np.min(img[:, :, 2]))

        num_pixels += img.shape[0] * img.shape[1]

    # 计算均值
    r_mean = r_sum / num_pixels
    g_mean = g_sum / num_pixels
    b_mean = b_sum / num_pixels

    # 计算方差
    r_std = math.sqrt((r_sum_sq / num_pixels) - (r_mean ** 2))
    g_std = math.sqrt((g_sum_sq / num_pixels) - (g_mean ** 2))
    b_std = math.sqrt((b_sum_sq / num_pixels) - (b_mean ** 2))

    return (
        np.array([r_mean, g_mean, b_mean], np.float32),
        np.array([r_std, g_std, b_std], np.float32),
        np.array([r_max, g_max, b_max], np.float32),
        np.array([r_min, g_min, b_min], np.float32)
    )

# def get_camera_color_statistics(path_list, scale, offset):
#     r_max = 0
#     g_max = 0
#     b_max = 0
#     r_min = 255
#     g_min = 255
#     b_min = 255
#     r_channel = 0
#     g_channel = 0
#     b_channel = 0
#     num = 0
#     for s in tqdm(path_list, desc='calc_color_stats-1'):
#         if type(s) is str:
#             img = cv2.imread(s, cv2.IMREAD_UNCHANGED)
#         else:
#             img = s
#         img = img * scale + offset
#         r_channel = r_channel + np.sum(img[:, :, 0])
#         g_channel = g_channel + np.sum(img[:, :, 1])
#         b_channel = b_channel + np.sum(img[:, :, 2])
#         num += img.shape[0] * img.shape[1]
#         r_max_tmp = np.max(img[:, :, 0])
#         r_max = r_max_tmp if r_max_tmp > r_max else r_max
#         r_min_tmp = np.min(img[:, :, 0])
#         r_min = r_min_tmp if r_min_tmp < r_min else r_min
#         g_max_tmp = np.max(img[:, :, 1])
#         g_max = g_max_tmp if g_max_tmp > g_max else g_max
#         g_min_tmp = np.min(img[:, :, 1])
#         g_min = g_min_tmp if g_min_tmp < g_min else g_min
#         b_max_tmp = np.max(img[:, :, 2])
#         b_max = b_max_tmp if b_max_tmp > b_max else b_max
#         b_min_tmp = np.min(img[:, :, 2])
#         b_min = b_min_tmp if b_min_tmp < b_min else b_min
#
#     r_mean = r_channel / num
#     g_mean = g_channel / num
#     b_mean = b_channel / num
#     r_channel = 0
#     g_channel = 0
#     b_channel = 0
#     for s in tqdm(path_list, desc='calc_color_stats-2'):
#         if type(s) is str:
#             img = cv2.imread(s, cv2.IMREAD_UNCHANGED)
#         else:
#             img = s
#         img = img * scale + offset
#         r_channel = r_channel + np.sum((img[:, :, 0] - r_mean) ** 2)
#         g_channel = g_channel + np.sum((img[:, :, 1] - g_mean) ** 2)
#         b_channel = b_channel + np.sum((img[:, :, 2] - b_mean) ** 2)
#     r_std = math.sqrt(r_channel / num)
#     g_std = math.sqrt(g_channel / num)
#     b_std = math.sqrt(b_channel / num)
#     return (np.array([r_mean, g_mean, b_mean], np.float32),
#             np.array([r_std, g_std, b_std], np.float32),
#             np.array([r_max, g_max, b_max], np.float32),
#             np.array([r_min, g_min, b_min], np.float32))


def get_camera_depth_statistics(path_list, scale, offset):
    depth_sum = 0
    depth_sum_sq = 0
    depth_max = 0
    depth_min = 65535
    num_pixels = 0

    for s in tqdm(path_list, desc='calc_depth_stats'):
        if type(s) is str:
            img = cv2.imread(s, cv2.IMREAD_UNCHANGED).astype(np.float64)
        else:
            img = s.astype(np.float64)
        img = img * scale + offset

        depth_sum += np.sum(img)
        depth_sum_sq += np.sum(img ** 2)
        num_pixels += img.shape[0] * img.shape[1]

        depth_max = max(depth_max, np.max(img))
        depth_min = min(depth_min, np.min(img))

    depth_mean = depth_sum / num_pixels
    depth_std = np.sqrt(depth_sum_sq / num_pixels - depth_mean ** 2)
    return (
        np.array(depth_mean, np.float32),
        np.array(depth_std, np.float32),
        np.array(depth_max, np.float32),
        np.array(depth_min, np.float32)
    )

# def get_camera_depth_statistics(path_list, scale, offset):
#     depth_max = 0
#     depth_min = 65535
#     depth_channel = 0
#     num = 0
#     for s in tqdm(path_list, desc='calc_depth_stats-1'):
#         if type(s) is str:
#             img = cv2.imread(s, cv2.IMREAD_UNCHANGED)
#         else:
#             img = s
#         img = img * scale + offset
#         depth_channel = depth_channel + np.sum(img)
#         num += img.shape[0] * img.shape[1]
#         depth_max_tmp = np.max(img[:, :])
#         depth_max = depth_max_tmp if depth_max_tmp > depth_max else depth_max
#         depth_min_tmp = np.min(img[:, :])
#         depth_min = depth_min_tmp if depth_min_tmp < depth_min else depth_min
#
#     depth_mean = depth_channel / num
#     depth_channel = 0
#     for s in tqdm(path_list, desc='calc_depth_stats-2'):
#         if type(s) is str:
#             img = cv2.imread(s, cv2.IMREAD_UNCHANGED)
#         else:
#             img = s
#         img = img * scale + offset
#         depth_channel = depth_channel + np.sum((img[:, :] - depth_mean) ** 2)
#     depth_std = math.sqrt(depth_channel / num)
#     print(depth_max, depth_min, depth_mean, depth_std)
#     return (np.array(depth_mean, np.float32),
#             np.array(depth_std, np.float32),
#             np.array(depth_max, np.float32),
#             np.array(depth_min, np.float32))


def get_camera_point_cloud_statistics(path_list, scale, offset):
    x_sum = 0
    y_sum = 0
    z_sum = 0
    r_sum = 0
    g_sum = 0
    b_sum = 0
    x_sum_sq = 0
    y_sum_sq = 0
    z_sum_sq = 0
    r_sum_sq = 0
    g_sum_sq = 0
    b_sum_sq = 0
    x_max = -np.inf
    y_max = -np.inf
    z_max = -np.inf
    r_max = -np.inf
    g_max = -np.inf
    b_max = -np.inf
    x_min = np.inf
    y_min = np.inf
    z_min = np.inf
    r_min = np.inf
    g_min = np.inf
    b_min = np.inf
    num_points = 0

    for s in tqdm(path_list, desc='calc_point_cloud_stats'):
        if type(s) is str:
            pc = np.load(s).astype(np.float64)
        else:
            pc = s
        pc = pc * scale + offset

        # 更新均值和方差的累加值
        x_sum += np.sum(pc[:, 0])
        y_sum += np.sum(pc[:, 1])
        z_sum += np.sum(pc[:, 2])
        r_sum += np.sum(pc[:, 3])
        g_sum += np.sum(pc[:, 4])
        b_sum += np.sum(pc[:, 5])
        x_sum_sq += np.sum(pc[:, 0] ** 2)
        y_sum_sq += np.sum(pc[:, 1] ** 2)
        z_sum_sq += np.sum(pc[:, 2] ** 2)
        r_sum_sq += np.sum(pc[:, 3] ** 2)
        g_sum_sq += np.sum(pc[:, 4] ** 2)
        b_sum_sq += np.sum(pc[:, 5] ** 2)

        # 更新最大值和最小值
        x_max = max(x_max, np.max(pc[:, 0]))
        y_max = max(y_max, np.max(pc[:, 1]))
        z_max = max(z_max, np.max(pc[:, 2]))
        r_max = max(r_max, np.max(pc[:, 3]))
        g_max = max(g_max, np.max(pc[:, 4]))
        b_max = max(b_max, np.max(pc[:, 5]))
        x_min = min(x_min, np.min(pc[:, 0]))
        y_min = min(y_min, np.min(pc[:, 1]))
        z_min = min(z_min, np.min(pc[:, 2]))
        r_min = min(r_min, np.min(pc[:, 3]))
        g_min = min(g_min, np.min(pc[:, 4]))
        b_min = min(b_min, np.min(pc[:, 5]))

        num_points += pc.shape[0]

    # 计算均值
    x_mean = x_sum / num_points
    y_mean = y_sum / num_points
    z_mean = z_sum / num_points
    r_mean = r_sum / num_points
    g_mean = g_sum / num_points
    b_mean = b_sum / num_points

    # 计算方差
    x_std = math.sqrt(x_sum_sq / num_points - x_mean ** 2)
    y_std = math.sqrt(y_sum_sq / num_points - y_mean ** 2)
    z_std = math.sqrt(z_sum_sq / num_points - z_mean ** 2)
    r_std = math.sqrt(r_sum_sq / num_points - r_mean ** 2)
    g_std = math.sqrt(g_sum_sq / num_points - g_mean ** 2)
    b_std = math.sqrt(b_sum_sq / num_points - b_mean ** 2)

    return (
        np.array([x_mean, y_mean, z_mean, r_mean, g_mean, b_mean], np.float32),
        np.array([x_std, y_std, z_std, r_std, g_std, b_std], np.float32),
        np.array([x_max, y_max, z_max, r_max, g_max, b_max], np.float32),
        np.array([x_min, y_min, z_min, r_min, g_min, b_min], np.float32)
    )

# def get_camera_point_cloud_statistics(path_list, scale, offset):
#     x_max = 0
#     y_max = 0
#     z_max = 0
#     x_min = 255
#     y_min = 255
#     z_min = 255
#     x_channel = 0
#     y_channel = 0
#     z_channel = 0
#     r_max = 0
#     g_max = 0
#     b_max = 0
#     r_min = 255
#     g_min = 255
#     b_min = 255
#     r_channel = 0
#     g_channel = 0
#     b_channel = 0
#     num = 0
#     for s in tqdm(path_list, desc='calc_point_cloud_stats-1'):
#         if type(s) is str:
#             pc = np.load(s)
#         else:
#             pc = s
#         pc = pc * scale + offset
#         x_channel = x_channel + np.sum(pc[:, 0])
#         y_channel = y_channel + np.sum(pc[:, 1])
#         z_channel = z_channel + np.sum(pc[:, 2])
#         r_channel = r_channel + np.sum(pc[:, 3])
#         g_channel = g_channel + np.sum(pc[:, 4])
#         b_channel = b_channel + np.sum(pc[:, 5])
#         num += pc.shape[0]
#         x_max_tmp = np.max(pc[:, 0])
#         x_max = x_max_tmp if x_max_tmp > x_max else x_max
#         x_min_tmp = np.min(pc[:, 0])
#         x_min = x_min_tmp if x_min_tmp < x_min else x_min
#         y_max_tmp = np.max(pc[:, 1])
#         y_max = y_max_tmp if y_max_tmp > y_max else y_max
#         y_min_tmp = np.min(pc[:, 1])
#         y_min = y_min_tmp if y_min_tmp < y_min else y_min
#         z_max_tmp = np.max(pc[:, 2])
#         z_max = z_max_tmp if z_max_tmp > z_max else z_max
#         z_min_tmp = np.min(pc[:, 2])
#         z_min = z_min_tmp if z_min_tmp < z_min else z_min
#
#         r_max_tmp = np.max(pc[:, 3])
#         r_max = r_max_tmp if r_max_tmp > r_max else r_max
#         r_min_tmp = np.min(pc[:, 3])
#         r_min = r_min_tmp if r_min_tmp < r_min else r_min
#         g_max_tmp = np.max(pc[:, 4])
#         g_max = g_max_tmp if g_max_tmp > g_max else g_max
#         g_min_tmp = np.min(pc[:, 4])
#         g_min = g_min_tmp if g_min_tmp < g_min else g_min
#         b_max_tmp = np.max(pc[:, 5])
#         b_max = b_max_tmp if b_max_tmp > b_max else b_max
#         b_min_tmp = np.min(pc[:, 5])
#         b_min = b_min_tmp if b_min_tmp < b_min else b_min
#
#     x_mean = x_channel / num
#     y_mean = y_channel / num
#     z_mean = z_channel / num
#     r_mean = r_channel / num
#     g_mean = g_channel / num
#     b_mean = b_channel / num
#     x_channel = 0
#     y_channel = 0
#     z_channel = 0
#     r_channel = 0
#     g_channel = 0
#     b_channel = 0
#     for s in tqdm(path_list, desc='calc_point_cloud_stats-2'):
#         if type(s) is str:
#             pc = np.load(s)
#         else:
#             pc = s
#         pc = pc * scale + offset
#         x_channel = x_channel + np.sum((pc[:, 0] - x_mean) ** 2)
#         y_channel = y_channel + np.sum((pc[:, 1] - y_mean) ** 2)
#         z_channel = z_channel + np.sum((pc[:, 2] - z_mean) ** 2)
#         r_channel = r_channel + np.sum((pc[:, 3] - r_mean) ** 2)
#         g_channel = g_channel + np.sum((pc[:, 4] - g_mean) ** 2)
#         b_channel = b_channel + np.sum((pc[:, 5] - b_mean) ** 2)
#     x_std = math.sqrt(x_channel / num)
#     y_std = math.sqrt(y_channel / num)
#     z_std = math.sqrt(z_channel / num)
#     r_std = math.sqrt(r_channel / num)
#     g_std = math.sqrt(g_channel / num)
#     b_std = math.sqrt(b_channel / num)
#     return (np.array([x_mean, y_mean, z_mean, r_mean, g_mean, b_mean], np.float32),
#             np.array([x_std, y_std, z_std, r_std, g_std, b_std], np.float32),
#             np.array([x_max, y_max, z_max, r_max, g_max, b_max], np.float32),
#             np.array([x_min, y_min, z_min, r_min, g_min, b_min], np.float32))


def get_qpos_statistics(qpos, scale, offset):
    qpos = qpos * scale + offset
    qpos_mean = qpos.mean(dim=[0]).float()
    qpos_std = qpos.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
    qpos_min = qpos.min(dim=0).values.float()
    qpos_max = qpos.max(dim=0).values.float()
    return qpos_mean, qpos_std, qpos_max, qpos_min


def calc_scale_offset(max_arr, min_arr, output_max, output_min):
    input_range = max_arr - min_arr
    ignore_dim = input_range < 0.0001
    input_range[ignore_dim] = output_max[ignore_dim] - output_min[ignore_dim]
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * min_arr
    offset[ignore_dim] = (output_max[ignore_dim] + output_min[ignore_dim]) / 2 - min_arr[ignore_dim]
    return scale, offset


# ----------------------------------------------------------------------------------------------------------------------


def get_all_pose_incre_data(all_pose_data, chunk_size, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        all_pose_incre_data = []
        for k in tqdm(range(len(all_pose_data)), desc='calc_all_pose_incre_data'):
            for i in range(len(all_pose_data[k]) - 1):
                begin_matrix = create_transformation_matrix(all_pose_data[k][i][0], all_pose_data[k][i][1], all_pose_data[k][i][2],
                                                            all_pose_data[k][i][3], all_pose_data[k][i][4], all_pose_data[k][i][5])
                end_matrix = create_transformation_matrix(all_pose_data[k][i+1][0], all_pose_data[k][i+1][1], all_pose_data[k][i+1][2],
                                                          all_pose_data[k][i+1][3], all_pose_data[k][i+1][4], all_pose_data[k][i+1][5])
                result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
                xyzrpy = matrix_to_xyzrpy(result_matrix)
                all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][i+1][6]])
        return all_pose_incre_data
    elif arm_end_pose_incre_mode == 1:
        all_pose_incre_data = []
        for k in tqdm(range(len(all_pose_data)), desc='calc_all_pose_incre_data'):
            for i in range(len(all_pose_data[k]) - 1):
                begin_matrix = create_transformation_matrix(all_pose_data[k][i][0], all_pose_data[k][i][1], all_pose_data[k][i][2],
                                                            all_pose_data[k][i][3], all_pose_data[k][i][4], all_pose_data[k][i][5])
                for j in range(i+1, len(all_pose_data[k])):
                    if j - (i+1) >= chunk_size:
                        break
                    end_matrix = create_transformation_matrix(all_pose_data[k][j][0], all_pose_data[k][j][1], all_pose_data[k][j][2],
                                                              all_pose_data[k][j][3], all_pose_data[k][j][4], all_pose_data[k][j][5])
                    result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
                    xyzrpy = matrix_to_xyzrpy(result_matrix)
                    all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6]])
                    # all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6] - all_pose_data[k][i][6]])
        return all_pose_incre_data
    elif arm_end_pose_incre_mode == 2:
        all_pose_incre_data = []
        for k in tqdm(range(len(all_pose_data)), desc='calc_all_pose_incre_data'):
            for i in range(len(all_pose_data[k]) - chunk_size):
                all_pose_incre_data_chunk = []
                begin_matrix = create_transformation_matrix(all_pose_data[k][i][0], all_pose_data[k][i][1], all_pose_data[k][i][2],
                                                            all_pose_data[k][i][3], all_pose_data[k][i][4], all_pose_data[k][i][5])
                for j in range(i+1, i+chunk_size+1):
                    end_matrix = create_transformation_matrix(all_pose_data[k][j][0], all_pose_data[k][j][1], all_pose_data[k][j][2],
                                                              all_pose_data[k][j][3], all_pose_data[k][j][4], all_pose_data[k][j][5])
                    result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
                    xyzrpy = matrix_to_xyzrpy(result_matrix)
                    all_pose_incre_data_chunk.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6]])
                    # all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6] - all_pose_data[k][i][6]])
                all_pose_incre_data.append(all_pose_incre_data_chunk)
        return all_pose_incre_data


def get_all_pose_incre_datas(all_pose_datas, num, dim, chunk_size, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        all_pose_incre_datas = []
        for i in range(num):
            all_pose_data = []
            for k in range(len(all_pose_datas)):
                all_pose_data.append(all_pose_datas[k][:, i * dim:(i + 1) * dim])
            all_pose_incre_datas.append(get_all_pose_incre_data(all_pose_data, chunk_size, arm_end_pose_incre_mode))
        return np.concatenate(all_pose_incre_datas, axis=-1)
    elif arm_end_pose_incre_mode == 1:
        all_pose_incre_datas = []
        for i in range(num):
            all_pose_data = []
            for k in range(len(all_pose_datas)):
                all_pose_data.append(all_pose_datas[k][:, i * dim:(i + 1) * dim])
            all_pose_incre_datas.append(get_all_pose_incre_data(all_pose_data, chunk_size, arm_end_pose_incre_mode))
        return np.concatenate(all_pose_incre_datas, axis=-1)
    elif arm_end_pose_incre_mode == 2:
        all_pose_incre_datas = []
        for i in range(num):
            all_pose_data = []
            for k in range(len(all_pose_datas)):
                all_pose_data.append(all_pose_datas[k][:, i * dim:(i + 1) * dim])
            all_pose_incre_datas.append(get_all_pose_incre_data(all_pose_data, chunk_size, arm_end_pose_incre_mode))
        arr = np.concatenate(all_pose_incre_datas, axis=-1)
        return arr.reshape((arr.shape[0], -1))


def calc_pose_incre(base_pose, pose_data, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        pose_incre_data = []
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        end_matrix = create_transformation_matrix(pose_data[0][0], pose_data[0][1], pose_data[0][2],
                                                  pose_data[0][3], pose_data[0][4], pose_data[0][5])
        result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
        xyzrpy = matrix_to_xyzrpy(result_matrix)
        pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[0][6]])
        for i in range(1, len(pose_data)):
            begin_matrix = create_transformation_matrix(pose_data[i-1][0], pose_data[i-1][1], pose_data[i-1][2],
                                                        pose_data[i-1][3], pose_data[i-1][4], pose_data[i-1][5])
            end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
                                                      pose_data[i][3], pose_data[i][4], pose_data[i][5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
        return np.array(pose_incre_data)
    elif arm_end_pose_incre_mode == 1:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(pose_data)):
            end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
                                                      pose_data[i][3], pose_data[i][4], pose_data[i][5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6] - base_pose[6]])
        return np.array(pose_incre_data)
    elif arm_end_pose_incre_mode == 2:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(pose_data)):
            end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
                                                      pose_data[i][3], pose_data[i][4], pose_data[i][5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6] - base_pose[6]])
        return np.array(pose_incre_data)


def calc_pose_incres(base_poses, pose_datas, num, dim, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        pose_incres = []
        for i in range(num):
            pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(pose_incres, axis=-1)
    elif arm_end_pose_incre_mode == 1:
        pose_incres = []
        for i in range(num):
            pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(pose_incres, axis=-1)
    elif arm_end_pose_incre_mode == 2:
        pose_incres = []
        for i in range(num):
            pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(pose_incres, axis=-1)


def decode_pose_by_incre(base_pose, incre_data, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        pose_incre_data = []
        pose_incre_data.append([base_pose[0], base_pose[1], base_pose[2], base_pose[3], base_pose[4], base_pose[5], base_pose[6]])
        for i in range(len(incre_data)):
            begin_matrix = create_transformation_matrix(pose_incre_data[-1][0], pose_incre_data[-1][1], pose_incre_data[-1][2],
                                                        pose_incre_data[-1][3], pose_incre_data[-1][4], pose_incre_data[-1][5])
            incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
                                                        incre_data[i][3], incre_data[i][4], incre_data[i][5])
            result_matrix = np.dot(begin_matrix, incre_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
        return np.array(pose_incre_data[1:])
    elif arm_end_pose_incre_mode == 1:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(incre_data)):
            incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
                                                        incre_data[i][3], incre_data[i][4], incre_data[i][5])
            result_matrix = np.dot(begin_matrix, incre_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
        return np.array(pose_incre_data)
    elif arm_end_pose_incre_mode == 2:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(incre_data)):
            incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
                                                        incre_data[i][3], incre_data[i][4], incre_data[i][5])
            result_matrix = np.dot(begin_matrix, incre_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
        return np.array(pose_incre_data)


def decode_pose_by_incres(base_poses, incre_datas, num, dim, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        poses = []
        for i in range(num):
            poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(poses, axis=-1)
    elif arm_end_pose_incre_mode == 1:
        poses = []
        for i in range(num):
            poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(poses, axis=-1)
    elif arm_end_pose_incre_mode == 2:
        poses = []
        for i in range(num):
            poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(poses, axis=-1)


# ----------------------------------------------------------------------------------------------------------------------


# def get_all_pose_incre_data(all_pose_data, chunk_size):
#     all_pose_incre_data = []
#     for k in tqdm(range(len(all_pose_data)), desc='calc_all_pose_incre_data'):
#         for i in range(len(all_pose_data[k]) - 1):
#             begin_matrix = create_transformation_matrix(all_pose_data[k][i][0], all_pose_data[k][i][1], all_pose_data[k][i][2],
#                                                         all_pose_data[k][i][3], all_pose_data[k][i][4], all_pose_data[k][i][5])
#             for j in range(i+1, len(all_pose_data[k])):
#                 if j - (i+1) >= chunk_size:
#                     break
#                 end_matrix = create_transformation_matrix(all_pose_data[k][j][0], all_pose_data[k][j][1], all_pose_data[k][j][2],
#                                                           all_pose_data[k][j][3], all_pose_data[k][j][4], all_pose_data[k][j][5])
#                 result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
#                 xyzrpy = matrix_to_xyzrpy(result_matrix)
#                 all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6]])
#                 # all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6] - all_pose_data[k][i][6]])
#     return all_pose_incre_data
#
#
# def get_all_pose_incre_datas(all_pose_datas, num, dim, chunk_size):
#     all_pose_incre_datas = []
#     for i in range(num):
#         all_pose_data = []
#         for k in range(len(all_pose_datas)):
#             all_pose_data.append(all_pose_datas[k][:, i * dim:(i + 1) * dim])
#         all_pose_incre_datas.append(get_all_pose_incre_data(all_pose_data, chunk_size))
#     return np.concatenate(all_pose_incre_datas, axis=-1)
#
#
# def calc_pose_incre(base_pose, pose_data):
#     begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
#                                                 base_pose[3], base_pose[4], base_pose[5])
#     pose_incre_data = []
#     for i in range(len(pose_data)):
#         end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
#                                                   pose_data[i][3], pose_data[i][4], pose_data[i][5])
#         result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
#         xyzrpy = matrix_to_xyzrpy(result_matrix)
#         pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
#         # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6] - base_pose[6]])
#     return np.array(pose_incre_data)
#
#
# def calc_pose_incres(base_poses, pose_datas, num, dim):
#     pose_incres = []
#     for i in range(num):
#         pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim]))
#     return np.concatenate(pose_incres, axis=-1)
#
#
# def decode_pose_by_incre(base_pose, incre_data):
#     begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
#                                                 base_pose[3], base_pose[4], base_pose[5])
#     pose_incre_data = []
#     for i in range(len(incre_data)):
#         incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
#                                                     incre_data[i][3], incre_data[i][4], incre_data[i][5])
#         result_matrix = np.dot(begin_matrix, incre_matrix)
#         xyzrpy = matrix_to_xyzrpy(result_matrix)
#         pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
#         # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
#     return np.array(pose_incre_data)
#
#
# def decode_pose_by_incres(base_poses, incre_datas, num, dim):
#     poses = []
#     for i in range(num):
#         poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim]))
#     return np.concatenate(poses, axis=-1)


# ----------------------------------------------------------------------------------------------------------------------


# def get_all_pose_incre_data(all_pose_data, chunk_size):
#     all_pose_incre_data = []
#     for k in tqdm(range(len(all_pose_data)), desc='calc_all_pose_incre_data'):
#         for i in range(len(all_pose_data[k]) - chunk_size):
#             all_pose_incre_data_chunk = []
#             begin_matrix = create_transformation_matrix(all_pose_data[k][i][0], all_pose_data[k][i][1], all_pose_data[k][i][2],
#                                                         all_pose_data[k][i][3], all_pose_data[k][i][4], all_pose_data[k][i][5])
#             for j in range(i+1, i+chunk_size+1):
#                 end_matrix = create_transformation_matrix(all_pose_data[k][j][0], all_pose_data[k][j][1], all_pose_data[k][j][2],
#                                                           all_pose_data[k][j][3], all_pose_data[k][j][4], all_pose_data[k][j][5])
#                 result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
#                 xyzrpy = matrix_to_xyzrpy(result_matrix)
#                 all_pose_incre_data_chunk.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6]])
#                 # all_pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], all_pose_data[k][j][6] - all_pose_data[k][i][6]])
#             all_pose_incre_data.append(all_pose_incre_data_chunk)
#     return all_pose_incre_data
#
#
# def get_all_pose_incre_datas(all_pose_datas, num, dim, chunk_size):
#     all_pose_incre_datas = []
#     for i in range(num):
#         all_pose_data = []
#         for k in range(len(all_pose_datas)):
#             all_pose_data.append(all_pose_datas[k][:, i * dim:(i + 1) * dim])
#         all_pose_incre_datas.append(get_all_pose_incre_data(all_pose_data, chunk_size))
#     arr = np.concatenate(all_pose_incre_datas, axis=-1)
#     return arr.reshape((arr.shape[0], -1))
#
#
# def calc_pose_incre(base_pose, pose_data):
#     begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
#                                                 base_pose[3], base_pose[4], base_pose[5])
#     pose_incre_data = []
#     for i in range(len(pose_data)):
#         end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
#                                                   pose_data[i][3], pose_data[i][4], pose_data[i][5])
#         result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
#         xyzrpy = matrix_to_xyzrpy(result_matrix)
#         pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
#         # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6] - base_pose[6]])
#     return np.array(pose_incre_data)
#
#
# def calc_pose_incres(base_poses, pose_datas, num, dim):
#     pose_incres = []
#     for i in range(num):
#         pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim]))
#     return np.concatenate(pose_incres, axis=-1)
#
#
# def decode_pose_by_incre(base_pose, incre_data):
#     begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
#                                                 base_pose[3], base_pose[4], base_pose[5])
#     pose_incre_data = []
#     for i in range(len(incre_data)):
#         incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
#                                                     incre_data[i][3], incre_data[i][4], incre_data[i][5])
#         result_matrix = np.dot(begin_matrix, incre_matrix)
#         xyzrpy = matrix_to_xyzrpy(result_matrix)
#         pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
#         # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
#     return np.array(pose_incre_data)
#
#
# def decode_pose_by_incres(base_poses, incre_datas, num, dim):
#     poses = []
#     for i in range(num):
#         poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim]))
#     return np.concatenate(poses, axis=-1)


# ----------------------------------------------------------------------------------------------------------------------


def calc_relative_pose(poses, num, dim):
    results = []
    for i in range(len(poses)):
        begin_matrix = create_transformation_matrix(poses[i][0], poses[i][1], poses[i][2], poses[i][3], poses[i][4], poses[i][5])
        result = []
        for j in range(0, num):
            end_matrix = create_transformation_matrix(poses[i][j * dim + 0], poses[i][j * dim + 1], poses[i][j * dim + 2],
                                                      poses[i][j * dim + 3], poses[i][j * dim + 4], poses[i][j * dim + 5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            result += [xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], poses[i][j * dim + 6]]
        results.append(result)
    return np.array(results)


def get_norm_stats(args, dataset_path_list):
    if not args.use_ario:
        stats = {}
        all_qpos_joint_state_data = []
        all_action_joint_state_data = []
        all_episode_len = []
        for dataset_path in dataset_path_list:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                action = root['/action'][()]
                if "left" not in args.arm_joint_state_names:
                    qpos = qpos[:, args.arm_joint_state_dim:]
                    action = action[:, args.arm_joint_state_dim:]
                if "right" not in args.arm_joint_state_names:
                    qpos = qpos[:, :args.arm_joint_state_dim]
                    action = action[:, :args.arm_joint_state_dim]
                if args.use_robot_base:
                    qpos = np.concatenate((qpos, root['/base_action'][()]), axis=1)
                    action = np.concatenate((action, root['/base_action'][()]), axis=1)
                all_episode_len.append(qpos.shape[0])
            all_qpos_joint_state_data.append(torch.from_numpy(qpos))
            all_action_joint_state_data.append(torch.from_numpy(action))
        if args.ckpt_stats_dir != "":
            with open(args.ckpt_stats_dir, 'rb') as f:
                stats = pickle.load(f)
        if args.use_arm_joint_state and ("joint_state_norm_mode" not in stats or stats["joint_state_norm_mode"] != args.qpos_norm_mode):
            all_qpos_joint_state_data = torch.cat(all_qpos_joint_state_data, dim=0)
            qpos_joint_state_mean, qpos_joint_state_std, qpos_joint_state_max, qpos_joint_state_min = get_qpos_statistics(all_qpos_joint_state_data, 1, 0)
            qpos_joint_state_scale, qpos_joint_state_offset = calc_scale_offset(qpos_joint_state_max, qpos_joint_state_min,
                                                                                torch.from_numpy(np.array([1 for _ in range(qpos_joint_state_max.shape[0])], dtype=np.float32)),
                                                                                torch.from_numpy(np.array([-1 for _ in range(qpos_joint_state_max.shape[0])], dtype=np.float32)))
            if args.qpos_norm_mode == 3:
                qpos_joint_state_mean, qpos_joint_state_std, qpos_joint_state_max, qpos_joint_state_min = get_qpos_statistics(all_qpos_joint_state_data, qpos_joint_state_scale, qpos_joint_state_offset)
            all_action_joint_state_data = torch.cat(all_action_joint_state_data, dim=0)
            action_joint_state_mean, action_joint_state_std, action_joint_state_max, action_joint_state_min = get_qpos_statistics(all_action_joint_state_data, 1, 0)
            action_joint_state_scale, action_joint_state_offset = calc_scale_offset(action_joint_state_max, action_joint_state_min,
                                                                                    torch.from_numpy(np.array([1 for _ in range(action_joint_state_max.shape[0])], dtype=np.float32)),
                                                                                    torch.from_numpy(np.array([-1 for _ in range(action_joint_state_max.shape[0])], dtype=np.float32)))
            if args.qpos_norm_mode == 3:
                action_joint_state_mean, action_joint_state_std, action_joint_state_max, action_joint_state_min = get_qpos_statistics(all_action_joint_state_data, action_joint_state_scale, action_joint_state_offset)
            stats["qpos_joint_state_mean"] = qpos_joint_state_mean.numpy()
            stats["qpos_joint_state_std"] = qpos_joint_state_std.numpy()
            stats["qpos_joint_state_min"] = qpos_joint_state_min.numpy()
            stats["qpos_joint_state_max"] = qpos_joint_state_max.numpy()
            stats["qpos_joint_state_scale"] = qpos_joint_state_scale.numpy()
            stats["qpos_joint_state_offset"] = qpos_joint_state_offset.numpy()
            stats["action_joint_state_mean"] = action_joint_state_mean.numpy()
            stats["action_joint_state_std"] = action_joint_state_std.numpy()
            stats["action_joint_state_min"] = action_joint_state_min.numpy()
            stats["action_joint_state_max"] = action_joint_state_max.numpy()
            stats["action_joint_state_scale"] = action_joint_state_scale.numpy()
            stats["action_joint_state_offset"] = action_joint_state_offset.numpy()
            stats["joint_state_norm_mode"] = args.qpos_norm_mode
        return stats, all_episode_len

    stats = {}
    all_qpos_joint_state_data = []
    all_action_joint_state_data = []
    all_qpos_end_pose_data = []
    all_action_end_pose_data = []
    all_qpos_robot_base_data = []
    all_action_robot_base_data = []
    all_episode_len = []
    all_camera_color_path = {}
    all_camera_depth_path = {}
    all_camera_point_cloud_path = {}
    for cam_name in args.camera_color_names:
        all_camera_color_path[cam_name] = []
    for cam_name in args.camera_depth_names:
        all_camera_depth_path[cam_name] = []
    for cam_name in args.camera_point_cloud_names:
        all_camera_point_cloud_path[cam_name] = []
    eps = 0.0001
    is_file_index = True
    for dataset_path in dataset_path_list:
        num = 0
        try:
            with h5py.File(dataset_path, 'r') as root:
                if args.use_arm_joint_state:
                    qpos_joint_state = np.concatenate([root[f'/arm/jointStatePosition/puppet{arm_name.capitalize()}'][()] for arm_name in args.arm_joint_state_names], axis=1)
                    action_joint_state = np.concatenate([root[f'/arm/jointStatePosition/master{arm_name.capitalize()}'][()] for arm_name in args.arm_joint_state_names], axis=1)
                    num = qpos_joint_state.shape[0]
                if args.use_arm_end_pose:
                    if args.use_arm_end_pose_incre:
                        qpos_end_pose = np.concatenate(
                            [np.concatenate([root[f'/localization/pose/{arm_name}'][()], root[f'/gripper/encoderAngle/{arm_name}'][()].reshape(-1, 1)], axis=1)
                             for arm_name in args.arm_end_pose_names], axis=1)
                        action_end_pose = qpos_end_pose
                    else:
                        qpos_end_pose = np.concatenate([root[f'/arm/endPose/puppet{arm_name.capitalize()}'][()] for arm_name in args.arm_end_pose_names], axis=1)
                        action_end_pose = np.concatenate([root[f'/arm/endPose/master{arm_name.capitalize()}'][()] for arm_name in args.arm_end_pose_names], axis=1)
                    num = qpos_end_pose.shape[0]
                if args.use_robot_base:
                    qpos_robot_base = root['/robotBase/vel/wheel'][()]
                    action_robot_base = root['/robotBase/vel/wheel'][()]
                    num = qpos_robot_base.shape[0]
                for cam_name in args.camera_color_names:
                    if args.use_camera_color:
                        if root[f'/camera/color/{cam_name}'][()].ndim == 1:
                            all_camera_color_path[cam_name] += [os.path.join(os.path.dirname(dataset_path), path.decode('utf-8')) for path in root[f'/camera/color/{cam_name}'][()].tolist()]
                        else:
                            all_camera_color_path[cam_name].append(root[f'/camera/color/{cam_name}'][()])
                            is_file_index = False
                for cam_name in args.camera_depth_names:
                    if args.use_camera_depth:
                        if root[f'/camera/depth/{cam_name}'][()].ndim == 1:
                            all_camera_depth_path[cam_name] += [os.path.join(os.path.dirname(dataset_path), path.decode('utf-8')) for path in root[f'/camera/depth/{cam_name}'][()].tolist()]
                        else:
                            all_camera_depth_path[cam_name].append(root[f'/camera/depth/{cam_name}'][()])
                            is_file_index = False
                for cam_name in args.camera_point_cloud_names:
                    if args.use_camera_point_cloud:
                        if root[f'/camera/pointCloud/{cam_name}'][()].ndim == 1:
                            all_camera_point_cloud_path[cam_name] += [os.path.join(os.path.dirname(dataset_path), path.decode('utf-8')) for path in root[f'/camera/pointCloud/{cam_name}'][()].tolist()]
                        else:
                            all_camera_point_cloud_path[cam_name].append(root[f'/camera/pointCloud/{cam_name}'][()])
                            is_file_index = False
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        if args.use_arm_joint_state:
            all_qpos_joint_state_data.append(torch.from_numpy(qpos_joint_state))
            all_action_joint_state_data.append(torch.from_numpy(action_joint_state))
        if args.use_arm_end_pose:
            all_qpos_end_pose_data.append(torch.from_numpy(qpos_end_pose))
            all_action_end_pose_data.append(torch.from_numpy(action_end_pose))
        if args.use_robot_base:
            all_qpos_robot_base_data.append(torch.from_numpy(qpos_robot_base))
            all_action_robot_base_data.append(torch.from_numpy(action_robot_base))
        all_episode_len.append(num)
    if args.ckpt_stats_dir != "":
        with open(args.ckpt_stats_dir, 'rb') as f:
            stats = pickle.load(f)
    if not is_file_index:
        for cam_name in args.camera_color_names:
            if len(all_camera_color_path[cam_name]) > 0:
                all_camera_color_path[cam_name] = np.concatenate(all_camera_color_path[cam_name], axis=0)
        for cam_name in args.camera_depth_names:
            if len(all_camera_depth_path[cam_name]) > 0:
                all_camera_depth_path[cam_name] = np.concatenate(all_camera_depth_path[cam_name], axis=0)
        for cam_name in args.camera_point_cloud_names:
            if len(all_camera_point_cloud_path[cam_name]) > 0:
                all_camera_point_cloud_path[cam_name] = np.concatenate(all_camera_point_cloud_path[cam_name], axis=0)

    # for cam_name in args.camera_color_names:
    #     camera_color_mean, camera_color_std, camera_color_max, camera_color_min \
    #         = get_camera_color_statistics(all_camera_color_path[cam_name], 1, 0)
    #     camera_color_scale, camera_color_offset = calc_scale_offset(np.array([camera_color_max]),
    #                                                                 np.array([camera_color_min]),
    #                                                                 np.array([1, 1, 1], dtype=np.float32),
    #                                                                 np.array([0, 0, 0], dtype=np.float32))

    if args.use_camera_depth and ("camera_depth_norm_mode" not in stats or stats["camera_depth_norm_mode"] != args.camera_depth_norm_mode):
        for cam_name in args.camera_depth_names:
            camera_depth_mean = 0.5
            camera_depth_std = 0.5
            camera_depth_max = 65535
            camera_depth_min = 65535
            camera_depth_scale = 1.0 / 65535
            camera_depth_offset = 0
            if args.camera_depth_norm_mode != 0:
                camera_depth_mean, camera_depth_std, camera_depth_max, camera_depth_min \
                    = get_camera_depth_statistics(all_camera_depth_path[cam_name], 1, 0)
                camera_depth_scale, camera_depth_offset = calc_scale_offset(np.array([camera_depth_max]),
                                                                            np.array([camera_depth_min]),
                                                                            np.array([1], dtype=np.float32),
                                                                            np.array([0], dtype=np.float32))
                if args.camera_depth_norm_mode == 3:
                    camera_depth_mean, camera_depth_std, camera_depth_max, camera_depth_min \
                        = get_camera_depth_statistics(all_camera_depth_path[cam_name], camera_depth_scale, camera_depth_offset)
            stats[f"camera_depth_{cam_name}_mean"] = camera_depth_mean
            stats[f"camera_depth_{cam_name}_std"] = camera_depth_std
            stats[f"camera_depth_{cam_name}_max"] = camera_depth_max
            stats[f"camera_depth_{cam_name}_min"] = camera_depth_min
            stats[f"camera_depth_{cam_name}_scale"] = camera_depth_scale
            stats[f"camera_depth_{cam_name}_offset"] = camera_depth_offset
        stats["camera_depth_norm_mode"] = args.camera_depth_norm_mode

    if args.use_camera_point_cloud and ("camera_point_cloud_norm_mode" not in stats or stats["camera_point_cloud_norm_mode"] != args.camera_point_cloud_norm_mode):
        for cam_name in args.camera_point_cloud_names:
            camera_point_cloud_mean = 0
            camera_point_cloud_std = 1
            camera_point_cloud_max = 0
            camera_point_cloud_min = 0
            camera_point_cloud_scale = 1
            camera_point_cloud_offset = 0
            if args.camera_point_cloud_norm_mode != 0:
                camera_point_cloud_mean, camera_point_cloud_std, camera_point_cloud_max, camera_point_cloud_min \
                    = get_camera_point_cloud_statistics(all_camera_point_cloud_path[cam_name], 1, 0)
                camera_point_cloud_scale, camera_point_cloud_offset = calc_scale_offset(camera_point_cloud_max,
                                                                                        camera_point_cloud_min,
                                                                                        np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                                                                                        np.array([-1, -1, -1, 0, 0, 0], dtype=np.float32))
                if args.camera_point_cloud_norm_mode == 3:
                    camera_point_cloud_mean, camera_point_cloud_std, camera_point_cloud_max, camera_point_cloud_min \
                        = get_camera_point_cloud_statistics(all_camera_point_cloud_path[cam_name], camera_point_cloud_scale, camera_point_cloud_offset)

            stats[f"camera_point_cloud_{cam_name}_mean"] = camera_point_cloud_mean
            stats[f"camera_point_cloud_{cam_name}_std"] = camera_point_cloud_std
            stats[f"camera_point_cloud_{cam_name}_max"] = camera_point_cloud_max
            stats[f"camera_point_cloud_{cam_name}_min"] = camera_point_cloud_min
            stats[f"camera_point_cloud_{cam_name}_scale"] = camera_point_cloud_scale
            stats[f"camera_point_cloud_{cam_name}_offset"] = camera_point_cloud_offset
        stats["camera_point_cloud_norm_mode"] = args.camera_point_cloud_norm_mode

    if args.use_arm_joint_state and ("joint_state_norm_mode" not in stats or stats["joint_state_norm_mode"] != args.qpos_norm_mode):
        all_qpos_joint_state_data = torch.cat(all_qpos_joint_state_data, dim=0)
        qpos_joint_state_mean, qpos_joint_state_std, qpos_joint_state_max, qpos_joint_state_min = get_qpos_statistics(all_qpos_joint_state_data, 1, 0)
        qpos_joint_state_scale, qpos_joint_state_offset = calc_scale_offset(qpos_joint_state_max, qpos_joint_state_min,
                                                                            torch.from_numpy(np.array([1 for _ in range(qpos_joint_state_max.shape[0])], dtype=np.float32)),
                                                                            torch.from_numpy(np.array([-1 for _ in range(qpos_joint_state_max.shape[0])], dtype=np.float32)))
        if args.qpos_norm_mode == 0:
            qpos_joint_state_scale = torch.from_numpy(np.ones(qpos_joint_state_scale.shape)).to(qpos_joint_state_scale.device)
            qpos_joint_state_offset = torch.from_numpy(np.zeros(qpos_joint_state_offset.shape)).to(qpos_joint_state_offset.device)
        if args.qpos_norm_mode == 3:
            qpos_joint_state_mean, qpos_joint_state_std, qpos_joint_state_max, qpos_joint_state_min = get_qpos_statistics(all_qpos_joint_state_data, qpos_joint_state_scale, qpos_joint_state_offset)
        all_action_joint_state_data = torch.cat(all_action_joint_state_data, dim=0)
        action_joint_state_mean, action_joint_state_std, action_joint_state_max, action_joint_state_min = get_qpos_statistics(all_action_joint_state_data, 1, 0)
        action_joint_state_scale, action_joint_state_offset = calc_scale_offset(action_joint_state_max, action_joint_state_min,
                                                                                torch.from_numpy(np.array([1 for _ in range(action_joint_state_max.shape[0])], dtype=np.float32)),
                                                                                torch.from_numpy(np.array([-1 for _ in range(action_joint_state_max.shape[0])], dtype=np.float32)))
        if args.qpos_norm_mode == 0:
            action_joint_state_scale = torch.from_numpy(np.ones(action_joint_state_scale.shape)).to(action_joint_state_scale.device)
            action_joint_state_offset = torch.from_numpy(np.zeros(action_joint_state_offset.shape)).to(action_joint_state_offset.device)
        if args.qpos_norm_mode == 3:
            action_joint_state_mean, action_joint_state_std, action_joint_state_max, action_joint_state_min = get_qpos_statistics(all_action_joint_state_data, action_joint_state_scale, action_joint_state_offset)
        stats["qpos_joint_state_mean"] = qpos_joint_state_mean.numpy()
        stats["qpos_joint_state_std"] = qpos_joint_state_std.numpy()
        stats["qpos_joint_state_min"] = qpos_joint_state_min.numpy()
        stats["qpos_joint_state_max"] = qpos_joint_state_max.numpy()
        stats["qpos_joint_state_scale"] = qpos_joint_state_scale.numpy()
        stats["qpos_joint_state_offset"] = qpos_joint_state_offset.numpy()
        stats["action_joint_state_mean"] = action_joint_state_mean.numpy()
        stats["action_joint_state_std"] = action_joint_state_std.numpy()
        stats["action_joint_state_min"] = action_joint_state_min.numpy()
        stats["action_joint_state_max"] = action_joint_state_max.numpy()
        stats["action_joint_state_scale"] = action_joint_state_scale.numpy()
        stats["action_joint_state_offset"] = action_joint_state_offset.numpy()
        stats["joint_state_norm_mode"] = args.qpos_norm_mode

    if args.use_arm_end_pose and ("end_pose_norm_mode" not in stats or stats["end_pose_norm_mode"] != args.qpos_norm_mode):
        if args.use_arm_end_pose_incre:
            all_qpos_end_pose_data = torch.tensor(calc_relative_pose(torch.cat(all_qpos_end_pose_data, dim=0), num=len(args.arm_end_pose_names), dim=args.arm_end_pose_dim))
        else:
            all_qpos_end_pose_data = torch.cat(all_qpos_end_pose_data, dim=0)
        qpos_end_pose_mean, qpos_end_pose_std, qpos_end_pose_max, qpos_end_pose_min = get_qpos_statistics(all_qpos_end_pose_data, 1, 0)
        qpos_end_pose_scale, qpos_end_pose_offset = calc_scale_offset(qpos_end_pose_max, qpos_end_pose_min,
                                                                      torch.from_numpy(np.array([1 for _ in range(qpos_end_pose_max.shape[0])], dtype=np.float32)),
                                                                      torch.from_numpy(np.array([-1 for _ in range(qpos_end_pose_max.shape[0])], dtype=np.float32)))
        if args.qpos_norm_mode == 0:
            qpos_end_pose_scale = torch.from_numpy(np.ones(qpos_end_pose_scale.shape)).to(qpos_end_pose_scale.device)
            qpos_end_pose_offset = torch.from_numpy(np.zeros(qpos_end_pose_offset.shape)).to(qpos_end_pose_offset.device)
        if args.qpos_norm_mode == 3:
            qpos_end_pose_mean, qpos_end_pose_std, qpos_end_pose_max, qpos_end_pose_min = get_qpos_statistics(all_qpos_end_pose_data, qpos_end_pose_scale, qpos_end_pose_offset)

        if args.use_arm_end_pose_incre:
            all_action_end_pose_data = torch.tensor(get_all_pose_incre_datas(all_action_end_pose_data, num=len(args.arm_end_pose_names), dim=args.arm_end_pose_dim, chunk_size=args.chunk_size, arm_end_pose_incre_mode=args.arm_end_pose_incre_mode))
        else:
            all_action_end_pose_data = torch.cat(all_action_end_pose_data, dim=0)
        action_end_pose_mean, action_end_pose_std, action_end_pose_max, action_end_pose_min = get_qpos_statistics(all_action_end_pose_data, 1, 0)
        action_end_pose_scale, action_end_pose_offset = calc_scale_offset(action_end_pose_max, action_end_pose_min,
                                                                          torch.from_numpy(np.array([1 for _ in range(action_end_pose_max.shape[0])], dtype=np.float32)),
                                                                          torch.from_numpy(np.array([-1 for _ in range(action_end_pose_max.shape[0])], dtype=np.float32)))
        if args.qpos_norm_mode == 0:
            action_end_pose_scale = torch.from_numpy(np.ones(action_end_pose_scale.shape)).to(action_end_pose_scale.device)
            action_end_pose_offset = torch.from_numpy(np.zeros(action_end_pose_offset.shape)).to(action_end_pose_offset.device)
        if args.qpos_norm_mode == 3:
            action_end_pose_mean, action_end_pose_std, action_end_pose_max, action_end_pose_min = get_qpos_statistics(all_action_end_pose_data, action_end_pose_scale, action_end_pose_offset)
        stats["qpos_end_pose_mean"] = qpos_end_pose_mean.numpy()
        stats["qpos_end_pose_std"] = qpos_end_pose_std.numpy()
        stats["qpos_end_pose_min"] = qpos_end_pose_min.numpy()
        stats["qpos_end_pose_max"] = qpos_end_pose_max.numpy()
        stats["qpos_end_pose_scale"] = qpos_end_pose_scale.numpy()
        stats["qpos_end_pose_offset"] = qpos_end_pose_offset.numpy()
        stats["action_end_pose_mean"] = action_end_pose_mean.numpy()
        stats["action_end_pose_std"] = action_end_pose_std.numpy()
        stats["action_end_pose_min"] = action_end_pose_min.numpy()
        stats["action_end_pose_max"] = action_end_pose_max.numpy()
        stats["action_end_pose_scale"] = action_end_pose_scale.numpy()
        stats["action_end_pose_offset"] = action_end_pose_offset.numpy()
        stats["end_pose_norm_mode"] = args.qpos_norm_mode

    if args.use_robot_base and ("robot_base_norm_mode" not in stats or stats["robot_base_norm_mode"] != args.qpos_norm_mode):
        all_qpos_robot_base_data = torch.cat(all_qpos_robot_base_data, dim=0)
        qpos_robot_base_mean, qpos_robot_base_std, qpos_robot_base_max, qpos_robot_base_min = get_qpos_statistics(all_qpos_robot_base_data, 1, 0)
        qpos_robot_base_scale, qpos_robot_base_offset = calc_scale_offset(qpos_robot_base_max, qpos_robot_base_min,
                                                                          torch.from_numpy(np.array([1 for _ in range(qpos_robot_base_max.shape[0])], dtype=np.float32)),
                                                                          torch.from_numpy(np.array([-1 for _ in range(qpos_robot_base_max.shape[0])], dtype=np.float32)))
        if args.qpos_norm_mode == 0:
            qpos_robot_base_scale = torch.from_numpy(np.ones(qpos_robot_base_scale.shape)).to(qpos_robot_base_scale.device)
            qpos_robot_base_offset = torch.from_numpy(np.zeros(qpos_robot_base_offset.shape)).to(qpos_robot_base_offset.device)
        if args.qpos_norm_mode == 3:
            qpos_robot_base_mean, qpos_robot_base_std, qpos_robot_base_max, qpos_robot_base_min = get_qpos_statistics(all_qpos_robot_base_data, qpos_robot_base_scale, qpos_robot_base_offset)

        all_action_robot_base_data = torch.cat(all_action_robot_base_data, dim=0)
        action_robot_base_mean, action_robot_base_std, action_robot_base_max, action_robot_base_min = get_qpos_statistics(all_action_robot_base_data, 1, 0)
        action_robot_base_scale, action_robot_base_offset = calc_scale_offset(action_robot_base_max, action_robot_base_min,
                                                                              torch.from_numpy(np.array([1 for _ in range(action_robot_base_max.shape[0])], dtype=np.float32)),
                                                                              torch.from_numpy(np.array([-1 for _ in range(action_robot_base_max.shape[0])], dtype=np.float32)))
        if args.qpos_norm_mode == 0:
            action_robot_base_scale = torch.from_numpy(np.ones(action_robot_base_scale.shape)).to(action_robot_base_scale.device)
            action_robot_base_offset = torch.from_numpy(np.zeros(action_robot_base_offset.shape)).to(action_robot_base_offset.device)
        if args.qpos_norm_mode == 3:
            action_robot_base_mean, action_robot_base_std, action_robot_base_max, action_robot_base_min = get_qpos_statistics(all_action_robot_base_data, action_robot_base_scale, action_robot_base_offset)
        stats["qpos_robot_base_mean"] = qpos_robot_base_mean.numpy()
        stats["qpos_robot_base_std"] = qpos_robot_base_std.numpy()
        stats["qpos_robot_base_min"] = qpos_robot_base_min.numpy()
        stats["qpos_robot_base_max"] = qpos_robot_base_max.numpy()
        stats["qpos_robot_base_scale"] = qpos_robot_base_scale.numpy()
        stats["qpos_robot_base_offset"] = qpos_robot_base_offset.numpy()
        stats["action_robot_base_mean"] = action_robot_base_mean.numpy()
        stats["action_robot_base_std"] = action_robot_base_std.numpy()
        stats["action_robot_base_min"] = action_robot_base_min.numpy()
        stats["action_robot_base_max"] = action_robot_base_max.numpy()
        stats["action_robot_base_scale"] = action_robot_base_scale.numpy()
        stats["action_robot_base_offset"] = action_robot_base_offset.numpy()
        stats["robot_base_norm_mode"] = args.qpos_norm_mode
    return stats, all_episode_len


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, args, dataset_index_list, dataset_path_list, episode_ids, episode_len, norm_stats):
        super(EpisodicDataset).__init__()
        self.dataset_index_list = dataset_index_list
        self.dataset_path_list = dataset_path_list
        self.episode_ids = episode_ids
        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.norm_stats = norm_stats
        self.args = args
        self.transformations = None
        self.__getitem__(0)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_index = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_index

    # def __len__(self):
    #     return sum(self.episode_len)

    def qpos_normalizer(self, qpos, action, next_action, key):
        if self.args.qpos_norm_mode == 0:
            return qpos, action, next_action
        qpos_scale = self.norm_stats["qpos_" + key + "_scale"]
        qpos_offset = self.norm_stats["qpos_" + key + "_offset"]
        qpos_mean = self.norm_stats["qpos_" + key + "_mean"]
        qpos_std = self.norm_stats["qpos_" + key + "_std"]
        action_scale = self.norm_stats["action_" + key + "_scale"] if self.args.use_dataset_action else self.norm_stats["qpos_" + key + "_scale"]
        action_offset = self.norm_stats["action_" + key + "_offset"] if self.args.use_dataset_action else self.norm_stats["qpos_" + key + "_offset"]
        action_mean = self.norm_stats["action_" + key + "_mean"] if self.args.use_dataset_action else self.norm_stats["qpos_" + key + "_mean"]
        action_std = self.norm_stats["action_" + key + "_std"] if self.args.use_dataset_action else self.norm_stats["qpos_" + key + "_std"]
        if self.args.qpos_norm_mode == 1:
            qpos = qpos * qpos_scale + qpos_offset
            action = action * action_scale + action_offset
            if self.args.next_action_num:
                next_action = next_action * action_scale + action_offset
        elif self.args.qpos_norm_mode == 2:
            qpos = (qpos - qpos_mean) / qpos_std
            action = (action - action_mean) / action_std
            if self.args.next_action_num:
                next_action = (next_action - action_mean) / action_std
        else:
            qpos = ((qpos * qpos_scale + qpos_offset) - qpos_mean) / qpos_std
            action = ((action * action_scale + action_offset) - action_mean) / action_std
            if self.args.next_action_num:
                next_action = ((next_action * action_scale + action_offset) - action_mean) / action_std
        return qpos, action, next_action

    def __getitem__(self, index):
        if not self.args.use_ario:
            episode_id, start_index = self._locate_transition(index)
            dataset_index = self.dataset_index_list[episode_id]
            dataset_path = self.dataset_path_list[episode_id]
            with h5py.File(dataset_path, 'r') as root:
                max_action_len = root['/action'].shape[0]  # max_episode
                if "left" not in self.args.arm_joint_state_names:
                    qpos = [root['/observations/qpos'][:, self.args.arm_joint_state_dim:][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                            for i in range(self.args.obs_history_num)]
                    if self.args.use_future:
                        qpos += [root['/observations/qpos'][:, self.args.arm_joint_state_dim:][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                 for i in range(self.args.obs_history_num)]
                    actions = root['/action'][:, self.args.arm_joint_state_dim:]
                if "right" not in self.args.arm_joint_state_names:
                    qpos = [root['/observations/qpos'][:, :self.args.arm_joint_state_dim][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                            for i in range(self.args.obs_history_num)]
                    if self.args.use_future:
                        qpos += [root['/observations/qpos'][:, :self.args.arm_joint_state_dim][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                 for i in range(self.args.obs_history_num)]
                    actions = root['/action'][:, :self.args.arm_joint_state_dim]
                qpos = np.array(qpos)
                if self.args.use_robot_base:
                    robot_base = [root['/base_action'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                                  for i in range(self.args.obs_history_num)]
                    if self.args.use_future:
                        robot_base += [root['base_action'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                       for i in range(self.args.obs_history_num)]
                        robot_base = np.array(robot_base)
                    qpos = np.concatenate((qpos, robot_base), axis=0)
                image_dict = dict()
                for cam_name in self.args.camera_color_names:
                    cam_name = 'cam_high' if cam_name == 'front' else cam_name
                    cam_name = 'cam_left_wrist' if cam_name == 'left' else cam_name
                    cam_name = 'cam_right_wrist' if cam_name == 'right' else cam_name
                    image_dict[cam_name] = [root[f'/observations/images/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                                            for i in range(self.args.obs_history_num)]
                    if self.args.use_future:
                        image_dict[cam_name] += [root[f'/observations/images/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                                 for i in range(self.args.obs_history_num)]
                start_action = min(start_index, max_action_len - 1)
                end_action = min(start_action + self.args.chunk_size, max_action_len)
                action_len = end_action - start_action
                action = actions[start_action:end_action]  # hack, to make timesteps more aligned
                if self.args.use_robot_base:
                    action = np.concatenate((action, root['/base_action'][end_action:end_action]), axis=1)
            padded_action = np.zeros((self.args.chunk_size, action.shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            action_is_pad = np.zeros(self.args.chunk_size)
            action_is_pad[action_len:] = 1
            all_cam_images = []
            for cam_name in self.args.camera_color_names:
                cam_name = 'cam_high' if cam_name == 'front' else cam_name
                cam_name = 'cam_left_wrist' if cam_name == 'left' else cam_name
                cam_name = 'cam_right_wrist' if cam_name == 'right' else cam_name
                for i in range(len(image_dict[cam_name])):
                    all_cam_images.append(image_dict[cam_name][i])
            all_cam_images = np.stack(all_cam_images, axis=0)
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            image_data = normalize(image_data)
            qpos_data = torch.from_numpy(qpos).float()
            qpos_data = (qpos_data - self.norm_stats["qpos_joint_state_mean"]) / self.norm_stats["qpos_joint_state_std"]
            action_data = torch.from_numpy(padded_action).float()
            action_is_pad = torch.from_numpy(action_is_pad).bool()
            action_data = (action_data - self.norm_stats["action_joint_state_mean"]) / self.norm_stats["action_joint_state_std"]
            image_depth_data = np.zeros(1, dtype=np.float32)
            camera_point_cloud_data = np.zeros(1, dtype=np.float32)
            qpos_end_pose_data = np.zeros(1, dtype=np.float32)
            qpos_robot_base_data = np.zeros(1, dtype=np.float32)
            next_action_joint_state_data = np.zeros(1, dtype=np.float32)
            next_action_end_pose_data = np.zeros(1, dtype=np.float32)
            next_action_robot_base_data = np.zeros(1, dtype=np.float32)
            next_action_is_pad = np.zeros(1, dtype=np.float32)
            instruction_input_ids = np.zeros(1, dtype=np.float32)
            instruction_attention_mask = np.zeros(1, dtype=np.float32)
            instruction_vector = np.zeros(1, dtype=np.float32)
            action_end_pose_data = np.zeros(1, dtype=np.float32)
            action_robot_base_data = np.zeros(1, dtype=np.float32)
            class_index = torch.from_numpy(np.array([0])).int()
            return (image_data, image_depth_data, camera_point_cloud_data,
                    qpos_data, qpos_end_pose_data, qpos_robot_base_data,
                    next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data,
                    next_action_is_pad,
                    instruction_input_ids, instruction_attention_mask, instruction_vector,
                    action_data, action_end_pose_data, action_robot_base_data, action_is_pad,
                    class_index)
        episode_id, start_index = self._locate_transition(index)
        dataset_index = self.dataset_index_list[episode_id]
        dataset_path = self.dataset_path_list[episode_id]
        max_action_len = 0

        instruction_input_ids = np.zeros(1, dtype=np.float32)
        instruction_attention_mask = np.zeros(1, dtype=np.float32)
        instruction_vector = np.zeros(1, dtype=np.float32)

        with h5py.File(dataset_path, 'r') as root:
            end_index = -1
            if self.args.use_instruction:
                instructions_vectors = None
                instructions_input_ids = None
                instructions_attention_mask = None
                time = root[f'timestamp'][start_index]
                # instruction_dir = os.path.join(os.path.dirname(dataset_path), root[f'instruction'][()].decode('utf-8'))
                # instruction_root = np.load(instruction_dir, allow_pickle=True).item()
                for i in range(len(root[f'instructions/segment_instructions/start_time'])):
                    start_time = root[f'instructions/segment_instructions/start_time'][i]
                    end_time = root[f'instructions/segment_instructions/end_time'][i]
                    if start_time <= time < end_time:
                        instructions_vectors = root[f'instructions/segment_instructions/{start_time}-{end_time}/vector']
                        instructions_input_ids = root[f'instructions/segment_instructions/{start_time}-{end_time}/input_ids']
                        instructions_attention_mask = root[
                            f'instructions/segment_instructions/{start_time}-{end_time}/attention_mask']
                        for j in range(start_index + 1, len(root['timestamp'])):
                            if root['timestamp'][j] > end_time:
                                end_index = j - 1
                                if end_index == start_index:
                                    end_index = -1
                                break
                        break
                rand_list = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
                random.shuffle(rand_list)
                if rand_list[0] == 0 or instructions_vectors is None or end_index == -1:
                    instructions_vectors = root[f'instructions/full_instructions/vector']
                    instructions_input_ids = root[f'instructions/full_instructions/input_ids']
                    instructions_attention_mask = root[f'instructions/full_instructions/attention_mask']
                instruction_index = random.randint(0, len(instructions_vectors) - 1)
                instruction_vector = instructions_vectors[instruction_index]
                instruction_input_ids = instructions_input_ids[instruction_index]
                instruction_attention_mask = instructions_attention_mask[instruction_index]
            if self.args.use_arm_joint_state:
                qposes_joint_state = np.concatenate([root[f'/arm/jointStatePosition/puppet{arm_name.capitalize()}'][()] for arm_name in self.args.arm_joint_state_names], axis=1)
                if self.args.use_dataset_action:
                    actions_joint_state = np.concatenate([root[f'/arm/jointStatePosition/master{arm_name.capitalize()}'][()] for arm_name in self.args.arm_joint_state_names], axis=1)
                else:
                    actions_joint_state = qposes_joint_state
                actions_joint_state = actions_joint_state[self.args.arm_delay_time:]
                actions_joint_state = np.concatenate((actions_joint_state, np.array([actions_joint_state[-1] for _ in range(self.args.arm_delay_time)])), axis=0)
                if end_index != -1:
                    qposes_joint_state = qposes_joint_state[:end_index]
                    actions_joint_state = actions_joint_state[:end_index]
                max_action_len = actions_joint_state.shape[0]
            if self.args.use_arm_end_pose:
                if self.args.use_arm_end_pose_incre:
                    qposes_end_pose = np.concatenate(
                        [np.concatenate([root[f'/localization/pose/{arm_name}'][()], root[f'/gripper/encoderAngle/{arm_name}'][()].reshape(-1, 1)], axis=1)
                         for arm_name in self.args.arm_end_pose_names], axis=1)
                    actions_end_pose = qposes_end_pose
                else:
                    qposes_end_pose = np.concatenate([root[f'/arm/endPose/puppet{arm_name.capitalize()}'][()] for arm_name in self.args.arm_end_pose_names], axis=1)
                    if self.args.use_dataset_action:
                        actions_end_pose = np.concatenate([root[f'/arm/endPose/master{arm_name.capitalize()}'][()] for arm_name in self.args.arm_end_pose_names], axis=1)
                    else:
                        actions_end_pose = qposes_end_pose
                actions_end_pose = actions_end_pose[self.args.arm_delay_time:]
                actions_end_pose = np.concatenate((actions_end_pose, np.array([actions_end_pose[-1] for _ in range(self.args.arm_delay_time)])), axis=0)
                if end_index != -1:
                    qposes_end_pose = qposes_end_pose[:end_index]
                    actions_end_pose = actions_end_pose[:end_index]
                max_action_len = actions_end_pose.shape[0]
            if self.args.use_robot_base:
                qposes_robot_base = root['/robotBase/vel/wheel'][()]
                actions_robot_base = root['/robotBase/vel/wheel'][()]
                actions_robot_base = actions_robot_base[self.args.arm_delay_time:]
                actions_robot_base = np.concatenate((actions_robot_base, np.array([actions_robot_base[-1] for _ in range(self.args.arm_delay_time)])), axis=0)
                if end_index != -1:
                    qposes_robot_base = qposes_robot_base[:end_index]
                    actions_robot_base = actions_robot_base[:end_index]
                max_action_len = actions_robot_base.shape[0]

            camera_color_dict = dict()
            camera_depth_dict = dict()
            camera_point_cloud_dict = dict()
            for cam_name in self.args.camera_color_names:
                if self.args.use_camera_color:
                    if root[f'/camera/color/{cam_name}'].ndim == 1:
                        camera_color_dict[cam_name] = [cv2.imread(os.path.join(os.path.dirname(dataset_path), root[f'/camera/color/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0].decode('utf-8')), cv2.IMREAD_UNCHANGED)
                                                       for i in range(self.args.obs_history_num)]
                        if self.args.use_future:
                            camera_color_dict[cam_name] += [cv2.imread(os.path.join(os.path.dirname(dataset_path), root[f'/camera/color/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1].decode('utf-8')), cv2.IMREAD_UNCHANGED)
                                                            for i in range(self.args.obs_history_num)]
                    else:
                        camera_color_dict[cam_name] = [root[f'/camera/color/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                                                       for i in range(self.args.obs_history_num)]
                        if self.args.use_future:
                            camera_color_dict[cam_name] += [root[f'/camera/color/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                                            for i in range(self.args.obs_history_num)]
                    for i in range(len(camera_color_dict[cam_name])):
                        default_width = 640
                        default_height = 480
                        camera_color_dict[cam_name] = [cv2.resize(camera_color_dict[cam_name][i], (default_width, default_height))
                                                       for i in range(len(camera_color_dict[cam_name]))]
                        # camera_width = camera_color_dict[cam_name][i].shape[1]
                        # camera_height = camera_color_dict[cam_name][i].shape[0]
                        # width_diff = default_width - camera_width
                        # height_diff = default_height - camera_height
                        # if width_diff < 0:
                        #     clip_width = abs(width_diff) // 2
                        #     camera_color_dict[cam_name] = [camera_color_dict[cam_name][i][:, clip_width:clip_width + default_width]
                        #                                    for i in range(len(camera_color_dict[cam_name]))]
                        # elif width_diff > 0:
                        #     add_width = width_diff // 2
                        #     top, bottom, left, right = 0, 0, add_width, add_width
                        #     camera_color_dict[cam_name] = [cv2.copyMakeBorder(camera_color_dict[cam_name][i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                        #                                    for i in range(len(camera_color_dict[cam_name]))]
                        # if height_diff < 0:
                        #     clip_height = abs(height_diff) // 2
                        #     camera_color_dict[cam_name] = [camera_color_dict[cam_name][i][clip_height:clip_height + default_height, :]
                        #                                    for i in range(len(camera_color_dict[cam_name]))]
                        # elif height_diff > 0:
                        #     add_height = height_diff // 2
                        #     top, bottom, left, right = add_height, add_height, 0, 0
                        #     camera_color_dict[cam_name] = [cv2.copyMakeBorder(camera_color_dict[cam_name][i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                        #                                    for i in range(len(camera_color_dict[cam_name]))]
                    # if cam_name == 'up':
                    #     clip_width = (848 - 640) // 2
                    #     camera_color_dict[cam_name] = [camera_color_dict[cam_name][i][:, clip_width:clip_width + 640]
                    #                                    for i in range(len(camera_color_dict[cam_name]))]
            for cam_name in self.args.camera_depth_names:
                if self.args.use_camera_depth:
                    if root[f'/camera/depth/{cam_name}'].ndim == 1:
                        camera_depth_dict[cam_name] = [cv2.imread(os.path.join(os.path.dirname(dataset_path), root[f'/camera/depth/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0].decode('utf-8')), cv2.IMREAD_UNCHANGED)
                                                       for i in range(self.args.obs_history_num)]
                        if self.args.use_future:
                            camera_depth_dict[cam_name] += [cv2.imread(os.path.join(os.path.dirname(dataset_path), root[f'/camera/depth/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1].decode('utf-8')), cv2.IMREAD_UNCHANGED)
                                                            for i in range(self.args.obs_history_num)]
                    else:
                        camera_depth_dict[cam_name] = [root[f'/camera/depth/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                                                       for i in range(self.args.obs_history_num)]
                        if self.args.use_future:
                            camera_depth_dict[cam_name] += [root[f'/camera/depth/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                                            for i in range(self.args.obs_history_num)]
                    for i in range(len(camera_depth_dict[cam_name])):
                        color_intrinsic = root[f'/camera/colorIntrinsic/{cam_name}'][()]
                        depth_intrinsic = root[f'/camera/depthIntrinsic/{cam_name}'][()]
                        color_extrinsic = root[f'/camera/colorExtrinsic/{cam_name}'][()]
                        depth_extrinsic = root[f'/camera/depthExtrinsic/{cam_name}'][()]
                        if not np.array_equal(color_extrinsic, depth_extrinsic):
                            camera_depth_dict[cam_name][i] = depth_to_color_projection(camera_depth_dict[cam_name][i], color_intrinsic, depth_intrinsic, np.dot(np.linalg.inv(color_extrinsic), depth_extrinsic))
                        default_width = 640
                        default_height = 480
                        camera_depth_dict[cam_name] = [cv2.resize(camera_depth_dict[cam_name][i], (default_width, default_height))
                                                       for i in range(len(camera_depth_dict[cam_name]))]
                        # camera_width = camera_depth_dict[cam_name][i].shape[1]
                        # camera_height = camera_depth_dict[cam_name][i].shape[0]
                        # width_diff = default_width - camera_width
                        # height_diff = default_height - camera_height
                        # if width_diff < 0:
                        #     clip_width = abs(width_diff) // 2
                        #     camera_depth_dict[cam_name] = [camera_depth_dict[cam_name][i][:, clip_width:clip_width + default_width]
                        #                                    for i in range(len(camera_depth_dict[cam_name]))]
                        # elif width_diff > 0:
                        #     add_width = width_diff // 2
                        #     top, bottom, left, right = 0, 0, add_width, add_width
                        #     camera_depth_dict[cam_name] = [cv2.copyMakeBorder(camera_depth_dict[cam_name][i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                        #                                    for i in range(len(camera_depth_dict[cam_name]))]
                        # if height_diff < 0:
                        #     clip_height = abs(height_diff) // 2
                        #     camera_depth_dict[cam_name] = [camera_depth_dict[cam_name][i][clip_height:clip_height + default_height, :]
                        #                                    for i in range(len(camera_depth_dict[cam_name]))]
                        # elif height_diff > 0:
                        #     add_height = height_diff // 2
                        #     top, bottom, left, right = add_height, add_height, 0, 0
                        #     camera_depth_dict[cam_name] = [cv2.copyMakeBorder(camera_depth_dict[cam_name][i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                        #                                    for i in range(len(camera_depth_dict[cam_name]))]
            for cam_name in self.args.camera_point_cloud_names:
                if self.args.use_camera_point_cloud:
                    if root[f'/camera/pointCloud/{cam_name}'].ndim == 1:
                        camera_point_cloud_dict[cam_name] = [np.load(os.path.join(os.path.dirname(dataset_path), root[f'/camera/pointCloud/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0].decode('utf-8')))[:, :(6 if self.args.use_camera_point_cloud_rgb else 3)]
                                                             for i in range(self.args.obs_history_num)]
                        if self.args.use_future:
                            camera_point_cloud_dict[cam_name] += [np.load(os.path.join(os.path.dirname(dataset_path), root[f'/camera/pointCloud/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1].decode('utf-8')))[:, :(6 if self.args.use_camera_point_cloud_rgb else 3)]
                                                                  for i in range(self.args.obs_history_num)]
                    else:
                        camera_point_cloud_dict[cam_name] = [root[f'/camera/pointCloud/{cam_name}'][(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0][:, :(6 if self.args.use_camera_point_cloud_rgb else 3)]
                                                             for i in range(self.args.obs_history_num)]
                        if self.args.use_future:
                            camera_point_cloud_dict[cam_name] += [root[f'/camera/pointCloud/{cam_name}'][(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1][:, :(6 if self.args.use_camera_point_cloud_rgb else 3)]
                                                                  for i in range(self.args.obs_history_num)]
        next_action_len = random.randint(0, self.args.next_action_num)
        start_next_action_index = start_index
        end_next_action_index = min(start_index + next_action_len, max_action_len - 1)
        start_action_index = end_next_action_index
        end_action_index = min(start_action_index + self.args.chunk_size, max_action_len)
        action_len = end_action_index - start_action_index
        next_action_len = end_next_action_index - start_next_action_index

        action_is_pad = np.zeros(self.args.chunk_size)
        action_is_pad[action_len:] = 1
        next_action_is_pad = np.zeros(self.args.next_action_num)
        if next_action_len <= 0:
            next_action_is_pad[:] = 1
        else:
            next_action_is_pad[next_action_len:] = 1

        qpos_joint_state = np.zeros(1, dtype=np.float32)
        padded_action_joint_state = np.zeros(1, dtype=np.float32)
        padded_next_action_joint_state = np.zeros(1, dtype=np.float32)
        if self.args.use_arm_joint_state:
            qpos_joint_state = [qposes_joint_state[(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                                for i in range(self.args.obs_history_num)]
            if self.args.use_future:
                qpos_joint_state += [qposes_joint_state[(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                     for i in range(self.args.obs_history_num)]
            qpos_joint_state = np.array(qpos_joint_state)
            if self.args.qpos_ignore_grasp:
                for i in range(len(self.args.arm_joint_state_names)):
                    qpos_joint_state[:, (i+1)*self.args.arm_joint_state_dim-1] = 0
            if self.args.augment_qpos:
                qpos_joint_state += np.random.uniform(-0.02, 0.02, qpos_joint_state.shape)
            action_joint_state = actions_joint_state[start_action_index:end_action_index]
            next_action_joint_state = actions_joint_state[start_next_action_index:end_next_action_index]
            padded_action_joint_state = np.zeros((self.args.chunk_size, action_joint_state.shape[1]), dtype=np.float32)
            padded_action_joint_state[:action_len] = action_joint_state
            padded_next_action_joint_state = np.zeros((self.args.next_action_num, action_joint_state.shape[1]), dtype=np.float32)
            padded_next_action_joint_state[:next_action_len] = next_action_joint_state

        qpos_end_pose = np.zeros(1, dtype=np.float32)
        padded_action_end_pose = np.zeros(1, dtype=np.float32)
        padded_next_action_end_pose = np.zeros(1, dtype=np.float32)
        if self.args.use_arm_end_pose:
            qpos_end_pose = [qposes_end_pose[(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                             for i in range(self.args.obs_history_num)]
            if self.args.use_future:
                qpos_end_pose += [qposes_end_pose[(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                  for i in range(self.args.obs_history_num)]
            qpos_end_pose = np.array(qpos_end_pose)
            if self.args.qpos_ignore_grasp:
                for i in range(len(self.args.arm_end_pose_names)):
                    qpos_end_pose[:, (i+1)*self.args.arm_end_pose_dim-1] = 0
            if self.args.augment_qpos:
                qpos_end_pose += np.random.uniform(-0.02, 0.02, qpos_end_pose.shape)
            if self.args.use_arm_end_pose_incre:
                qpos_end_pose = calc_relative_pose(qpos_end_pose, num=len(self.args.arm_end_pose_names), dim=self.args.arm_end_pose_dim)
            action_end_pose = actions_end_pose[start_action_index:end_action_index]
            if self.args.use_arm_end_pose_incre:
                action_end_pose = calc_pose_incres(actions_end_pose[start_action_index - 1 if start_action_index >= 1 else 0], action_end_pose, num=len(self.args.arm_end_pose_names), dim=self.args.arm_end_pose_dim, arm_end_pose_incre_mode=self.args.arm_end_pose_incre_mode)
            next_action_end_pose = actions_end_pose[start_next_action_index:end_next_action_index]
            padded_action_end_pose = np.zeros((self.args.chunk_size, action_end_pose.shape[1]), dtype=np.float32)
            padded_action_end_pose[:action_len] = action_end_pose
            padded_next_action_end_pose = np.zeros((self.args.next_action_num, action_end_pose.shape[1]), dtype=np.float32)
            padded_next_action_end_pose[:next_action_len] = next_action_end_pose

        qpos_robot_base = np.zeros(1, dtype=np.float32)
        padded_action_robot_base = np.zeros(1, dtype=np.float32)
        padded_next_action_robot_base = np.zeros(1, dtype=np.float32)
        if self.args.use_robot_base:
            qpos_robot_base = [qposes_robot_base[(start_index - self.args.obs_history_num + 1 + i) if (start_index - self.args.obs_history_num + 1 + i) > 0 else 0]
                               for i in range(self.args.obs_history_num)]
            if self.args.use_future:
                qpos_robot_base += [qposes_robot_base[(start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) if (start_index + self.args.chunk_size - self.args.obs_history_num + 1 + i) < max_action_len else max_action_len - 1]
                                    for i in range(self.args.obs_history_num)]
            qpos_robot_base = np.array(qpos_robot_base)
            if self.args.augment_qpos:
                qpos_robot_base += np.random.uniform(-0.02, 0.02, qpos_robot_base.shape)
            action_robot_base = actions_robot_base[start_action_index:end_action_index]
            next_action_robot_base = actions_robot_base[start_next_action_index:end_next_action_index]
            padded_action_robot_base = np.zeros((self.args.chunk_size, action_robot_base.shape[1]), dtype=np.float32)
            padded_action_robot_base[:action_len] = action_robot_base
            padded_next_action_robot_base = np.zeros((self.args.next_action_num, action_robot_base.shape[1]), dtype=np.float32)
            padded_next_action_robot_base[:next_action_len] = next_action_robot_base

        class_index = 0
        if self.args.class_num != 0:
            class_index = self.args.dataset_class[dataset_index]
        # instruction_input_ids = np.zeros(1, dtype=np.float32)
        # instruction_attention_mask = np.zeros(1, dtype=np.float32)
        # instruction_vector = np.zeros(1, dtype=np.float32)
        # if self.args.use_instruction:
        #     root = np.load(instruction_dir, allow_pickle=True).item()
        #     instruction_index = random.randint(0, len(root[f'instruction/vector']) - 1)
        #     instruction_vector = root[f'instruction/vector'][instruction_index]
        #     instruction_input_ids = root[f'instruction/input_ids'][instruction_index]
        #     instruction_attention_mask = root[f'instruction/attention_mask'][instruction_index]
        #     rand_list = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        #     random.shuffle(rand_list)
        #     if rand_list[0] == 0:
        #         rand_list = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        #         random.shuffle(rand_list)
        #         if rand_list[0] == 0 or 'invalid_instruction/vector' not in root or len(root[f'invalid_instruction/vector']) == 0:
        #             instruction_vector = np.random.normal(0, 1, instruction_vector.shape[0]).astype(np.float32)
        #         else:
        #             instruction_index = random.randint(0, len(root[f'invalid_instruction/vector']) - 1)
        #             instruction_vector = root[f'invalid_instruction/vector'][instruction_index]
        #             instruction_input_ids = root[f'invalid_instruction/input_ids'][instruction_index]
        #             instruction_attention_mask = root[f'invalid_instruction/attention_mask'][instruction_index]
        #         if self.args.class_num == 0:
        #             if self.args.use_arm_joint_state:
        #                 for i in range(padded_action_joint_state.shape[0]):
        #                     padded_action_joint_state[i] = padded_action_joint_state[0]
        #                 for i in range(padded_next_action_joint_state.shape[0]):
        #                     padded_next_action_joint_state[i] = padded_next_action_joint_state[0]
        #             if self.args.use_arm_end_pose:
        #                 for i in range(padded_action_end_pose.shape[0]):
        #                     padded_action_end_pose[i] = padded_action_end_pose[0]
        #                 for i in range(padded_next_action_end_pose.shape[0]):
        #                     padded_next_action_end_pose[i] = padded_next_action_end_pose[0]
        #             if self.args.use_robot_base:
        #                 for i in range(padded_action_robot_base.shape[0]):
        #                     padded_action_robot_base[i] = padded_action_robot_base[0]
        #                 for i in range(padded_next_action_robot_base.shape[0]):
        #                     padded_next_action_robot_base[i] = padded_next_action_robot_base[0]
        #         class_index = 0

        camera_color_data = np.zeros(1, dtype=np.float32)
        if self.args.use_camera_color:
            camera_colors = []
            for cam_name in self.args.camera_color_names:
                for i in range(len(camera_color_dict[cam_name])):
                    camera_colors.append(camera_color_dict[cam_name][i])
            camera_colors = np.stack(camera_colors, axis=0)
            # construct observations
            camera_color_data = torch.from_numpy(camera_colors)
            camera_color_data = torch.einsum('k h w c -> k c h w', camera_color_data)
            if self.transformations is None:
                print('Initializing transformations')
                original_size = camera_color_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
                ]
            if self.args.augment_images:
                for transform in self.transformations:
                    camera_color_data = transform(camera_color_data)
            camera_color_data = camera_color_data / 255.0
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            camera_color_data = normalize(camera_color_data)  # 图像归一化

        camera_depth_data = np.zeros(1, dtype=np.float32)
        if self.args.use_camera_depth:
            camera_depths = []
            for cam_name in self.args.camera_depth_names:
                for i in range(len(camera_depth_dict[cam_name])):
                    camera_depths.append(camera_depth_dict[cam_name][i])
                    # camera_depths[-1] = camera_depths[-1] / 65535.0
                    if self.args.camera_depth_norm_mode != 0:
                        mean = self.norm_stats[f"camera_depth_{cam_name}_mean"]
                        std = self.norm_stats[f"camera_depth_{cam_name}_std"]
                        scale = self.norm_stats[f"camera_depth_{cam_name}_scale"]
                        offset = self.norm_stats[f"camera_depth_{cam_name}_offset"]
                        if self.args.camera_depth_norm_mode == 1:
                            camera_depths[-1] = camera_depths[-1] * scale + offset
                        elif self.args.camera_depth_norm_mode == 2:
                            camera_depths[-1] = (camera_depths[-1] - mean) / std
                        else:
                            camera_depths[-1] = (camera_depths[-1] * scale + offset - mean) / std
            camera_depths = np.stack(camera_depths, axis=0)
            camera_depths = camera_depths.astype(np.float32)
            camera_depth_data = torch.from_numpy(camera_depths)

        camera_point_cloud_data = np.zeros(1, dtype=np.float32)
        if self.args.use_camera_point_cloud:
            camera_point_clouds = []
            for cam_name in self.args.camera_point_cloud_names:
                for i in range(len(camera_point_cloud_dict[cam_name])):
                    camera_point_clouds.append(camera_point_cloud_dict[cam_name][i])
                    if self.args.camera_point_cloud_norm_mode != 0:
                        scale = self.norm_stats[f"camera_point_cloud_{cam_name}_scale"]
                        offset = self.norm_stats[f"camera_point_cloud_{cam_name}_offset"]
                        mean = self.norm_stats[f"camera_point_cloud_{cam_name}_mean"]
                        std = self.norm_stats[f"camera_point_cloud_{cam_name}_std"]
                        index = 6 if self.args.use_camera_point_cloud_rgb else 3
                        if self.args.camera_point_cloud_norm_mode == 1:
                            camera_point_clouds[-1] = camera_point_clouds[-1] * scale[:index] + offset[:index]
                        elif self.args.camera_point_cloud_norm_mode == 2:
                            camera_point_clouds[-1] = (camera_point_clouds[-1] - mean[:index]) / std[:index]
                        else:
                            camera_point_clouds[-1] = ((camera_point_clouds[-1] * scale[:index] + offset[:index]) - mean[:index]) / std[:index]
            camera_point_clouds = np.stack(camera_point_clouds, axis=0)
            camera_point_cloud_data = torch.from_numpy(camera_point_clouds).float()

        instruction_input_ids = torch.from_numpy(instruction_input_ids).int()
        instruction_attention_mask = torch.from_numpy(instruction_attention_mask).bool()
        instruction_vector = torch.from_numpy(instruction_vector).float()
        qpos_joint_state_data = torch.from_numpy(qpos_joint_state).float()
        qpos_end_pose_data = torch.from_numpy(qpos_end_pose).float()
        qpos_robot_base_data = torch.from_numpy(qpos_robot_base).float()
        action_joint_state_data = torch.from_numpy(padded_action_joint_state).float()
        action_end_pose_data = torch.from_numpy(padded_action_end_pose).float()
        action_robot_base_data = torch.from_numpy(padded_action_robot_base).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        next_action_joint_state_data = torch.from_numpy(padded_next_action_joint_state).float()
        next_action_end_pose_data = torch.from_numpy(padded_next_action_end_pose).float()
        next_action_robot_base_data = torch.from_numpy(padded_next_action_robot_base).float()
        next_action_is_pad = torch.from_numpy(next_action_is_pad).bool()
        if self.args.use_arm_joint_state and self.args.qpos_norm_mode:
            qpos_joint_state_data, action_joint_state_data, next_action_joint_state_data = self.qpos_normalizer(qpos_joint_state_data, action_joint_state_data, next_action_joint_state_data, "joint_state")
        if self.args.use_arm_end_pose and self.args.qpos_norm_mode:
            if self.args.use_arm_end_pose_incre and self.args.arm_end_pose_incre_mode == 2:
                action_end_pose_data = action_end_pose_data.reshape((-1))
                next_action_end_pose_data = next_action_end_pose_data.reshape((-1))
            qpos_end_pose_data, action_end_pose_data, next_action_end_pose_data = self.qpos_normalizer(qpos_end_pose_data, action_end_pose_data, next_action_end_pose_data, "end_pose")
            if self.args.use_arm_end_pose_incre and self.args.arm_end_pose_incre_mode == 2:
                action_end_pose_data = action_end_pose_data.reshape((-1, self.args.arm_end_pose_dim * len(self.args.arm_end_pose_names)))
                next_action_end_pose_data = next_action_end_pose_data.reshape((-1, self.args.arm_end_pose_dim * len(self.args.arm_end_pose_names)))
        if self.args.use_robot_base and self.args.qpos_norm_mode:
            qpos_robot_base_data, action_robot_base_data, next_action_robot_base_data = self.qpos_normalizer(qpos_robot_base_data, action_robot_base_data, next_action_robot_base_data, "robot_base")
        class_index = torch.from_numpy(np.array([class_index])).long()
        return (camera_color_data, camera_depth_data, camera_point_cloud_data,
                qpos_joint_state_data, qpos_end_pose_data, qpos_robot_base_data,
                next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data, next_action_is_pad,
                instruction_input_ids, instruction_attention_mask, instruction_vector,
                action_joint_state_data, action_end_pose_data, action_robot_base_data, action_is_pad,
                class_index)


def load_data(args):
    dataset_dir_list = args.dataset_dir
    if type(dataset_dir_list) == str:
        dataset_dir_list = [dataset_dir_list]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, args.use_ario) for dataset_dir in dataset_dir_list]

    dataset_index_list = [i for i, dataset_path_list in enumerate(dataset_path_list_list) for _ in dataset_path_list]
    dataset_path_list = flatten_list(dataset_path_list_list)
    num_episodes_list = [0] + [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_list)

    train_episode_ids_list = []
    val_episode_ids_list = []
    # obtain train test split on dataset_dir_l[0]
    for i in range(len(dataset_path_list_list)):
        num_episodes = len(dataset_path_list_list[i])
        shuffled_episode_ids = np.random.permutation(num_episodes)
        train_episode_ids = shuffled_episode_ids[:int(args.train_ratio * num_episodes)]
        val_episode_ids = shuffled_episode_ids[int(args.train_ratio * num_episodes):]
        train_episode_ids_list.append(np.array([train_id+num_episodes_cumsum[i] for train_id in train_episode_ids]))
        val_episode_ids_list.append(np.array([val_id+num_episodes_cumsum[i] for val_id in val_episode_ids]))
    train_episode_ids = np.concatenate(train_episode_ids_list)
    val_episode_ids = np.concatenate(val_episode_ids_list)

    # obtain train test split on dataset_dir_l[0]
    # num_episodes_0 = len(dataset_path_list_list[0])
    # shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    # train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    # val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    # train_episode_ids_list = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_list[1:])]
    # val_episode_ids_list = [val_episode_ids_0]
    # train_episode_ids = np.concatenate(train_episode_ids_list)
    # val_episode_ids = np.concatenate(val_episode_ids_list)
    print(f'\n\nData from: {dataset_dir_list}\n- Train on {[len(x) for x in train_episode_ids_list]} episodes\n- Test on {[len(x) for x in val_episode_ids_list]} episodes\n\n')

    norm_stats, all_episode_len = get_norm_stats(args, dataset_path_list)
    train_episode_len_list = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_list]
    val_episode_len_list = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_list]
    train_episode_len = flatten_list(train_episode_len_list)
    val_episode_len = flatten_list(val_episode_len_list)
    # if stats_dir_l is None:
    #     stats_dir_l = dataset_dir_list
    # elif type(stats_dir_l) == str:
    #     stats_dir_l = [stats_dir_l]
    # norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir) for stats_dir in stats_dir_l]), use_robot_base, use_arm_end_pose)
    # print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = batch_sampler(args.train_batch_size, train_episode_len_list, args.sample_weights)
    batch_sampler_val = batch_sampler(args.val_batch_size, val_episode_len_list, None)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(args, dataset_index_list, dataset_path_list, train_episode_ids, train_episode_len, norm_stats)
    val_dataset = EpisodicDataset(args, dataset_index_list, dataset_path_list, val_episode_ids, val_episode_len, norm_stats)
    train_num_workers = 1
    val_num_workers = 1
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])

    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
