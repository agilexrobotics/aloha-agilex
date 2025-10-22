#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
from utils import set_seed, decode_pose_by_incre, calc_relative_pose, create_transformation_matrix, color_depth_to_point_cloud, depth_to_color_projection, matrix_to_xyzrpy  # helper functions
from policy import ACTPolicy, DiffusionPolicy, RTPolicy
from collections import deque
from transformers import BertTokenizer, BertModel
import rospy
from std_msgs.msg import Header, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
import threading
from torchvision import transforms
import cv2
from aloha_msgs.srv import StatusSrv, StatusSrvRequest, StatusSrvResponse
from functools import partial
import ros_numpy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys
import model_dict_mapping
from scipy.spatial import geometric_slerp
import open3d as o3d
import random
import tf
import math

sys.path.append("./")

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_category = None
inference_timestep = None
pre_instruction = ""
pre_instruction_attention_vector = None
pre_instruction_input_ids = None
pre_instruction_attention_mask = None


class BlockingDeque:
    def __init__(self):
        self.deque = deque()
        self.not_empty = threading.Condition()

    def append(self, item):
        with self.not_empty:
            self.deque.append(item)
            self.not_empty.notify()

    def popleft(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque.popleft()
        return item

    def left(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[0]
            return item

    def right(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[-1]
            return item

    def size(self):
        with self.not_empty:
            return len(self.deque)


def pre_process(norm_stats, key, qpos_norm_mode, is_action, data):
    if qpos_norm_mode == 0:
        return data
    qpos_scale = norm_stats["action_" + key + "_scale"] if is_action else norm_stats["qpos_" + key + "_scale"]
    qpos_offset = norm_stats["action_" + key + "_offset"] if is_action else norm_stats["qpos_" + key + "_offset"]
    qpos_mean = norm_stats["action_" + key + "_mean"] if is_action else norm_stats["qpos_" + key + "_mean"]
    qpos_std = norm_stats["action_" + key + "_std"] if is_action else norm_stats["qpos_" + key + "_std"]
    if qpos_norm_mode == 1:
        data = data * qpos_scale + qpos_offset
    elif qpos_norm_mode == 2:
        data = (data - qpos_mean) / qpos_std
    else:
        data = ((data * qpos_scale + qpos_offset) - qpos_mean) / qpos_std
    return data


def post_process(norm_stats, key, qpos_norm_mode, is_action, data):
    if qpos_norm_mode == 0:
        return data
    qpos_scale = norm_stats["action_" + key + "_scale"] if is_action else norm_stats["qpos_" + key + "_scale"]
    qpos_offset = norm_stats["action_" + key + "_offset"] if is_action else norm_stats["qpos_" + key + "_offset"]
    qpos_mean = norm_stats["action_" + key + "_mean"] if is_action else norm_stats["qpos_" + key + "_mean"]
    qpos_std = norm_stats["action_" + key + "_std"] if is_action else norm_stats["qpos_" + key + "_std"]
    if qpos_norm_mode == 1:
        data = (data - qpos_offset) / qpos_scale
    elif qpos_norm_mode == 2:
        data = data * qpos_std + qpos_mean
    else:
        data = (data * qpos_std + qpos_mean - qpos_offset) / qpos_scale
    return data


def make_policy(args):
    if args.policy_class == 'ACT':
        policy = ACTPolicy(args)
    elif args.policy_class == 'Diffusion':
        policy = DiffusionPolicy(args)
    elif args.policy_class == 'RT':
        policy = RTPolicy(args)
    else:
        raise NotImplementedError
    return policy


def get_camera_color(stats, camera_names, history_num, augment_color, camera_color):
    colors = []
    for cam_name in camera_names:
        for i in range(history_num):
            color = camera_color[cam_name][i]
            color = cv2.imencode('.jpg', color)[1].tobytes()
            color = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
            default_width = 640
            default_height = 480
            camera_width = color.shape[1]
            camera_height = color.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                color = color[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                color = color[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            color = rearrange(color, 'h w c -> c h w')
            colors.append(color)
    colors = np.stack(colors, axis=0)
    colors = torch.from_numpy(colors / 255.0).float().cuda().unsqueeze(0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    colors = normalize(colors)  # 图像归一化
    if augment_color:
        print('rand crop resize is used!')
        original_size = colors.shape[-2:]
        ratio = 0.95
        colors = colors[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                 int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        colors = colors.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        colors = resize_transform(colors)
        colors = colors.unsqueeze(0)
    return colors


def get_camera_depth(stats, camera_names, history_num, camera_depth_norm_mode, camera_depth):
    depths = []
    for cam_name in camera_names:
        for i in range(history_num):
            depth = camera_depth[cam_name][i]
            default_width = 640
            default_height = 480
            camera_width = depth.shape[1]
            camera_height = depth.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                depth = depth[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                depth = depth[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            depths.append(depth)
            if camera_depth_norm_mode != 0:
                mean = stats[f"camera_depth_{cam_name}_mean"]
                std = stats[f"camera_depth_{cam_name}_std"]
                scale = stats[f"camera_depth_{cam_name}_scale"]
                offset = stats[f"camera_depth_{cam_name}_offset"]
                if camera_depth_norm_mode == 1:
                    depths[-1] = depths[-1] * scale + offset
                elif camera_depth_norm_mode == 2:
                    depths[-1] = (depths[-1] - mean) / std
                else:
                    depths[-1] = (depths[-1] * scale + offset - mean) / std
                # depth_normalize = transforms.Normalize(mean=[mean], std=[std])
                # depths[-1] = depth_normalize(depths[-1])
                # depths.append(camera_depth[cam_name])
    depths = np.stack(depths, axis=0)
    depths = torch.from_numpy(depths).float().cuda().unsqueeze(0)
    return depths


def get_camera_point_cloud(stats, camera_names, history_num, voxel_size, use_farthest_point_down_sample,
                           point_num, augment_point_cloud, use_camera_point_cloud_rgb, camera_point_cloud_norm_mode,
                           camera_point_cloud, color_extrinsics, point_cloud_extrinsics):
    camera_point_clouds = []
    for cam_id, cam_name in enumerate(camera_names):
        for i in range(history_num):
            pc = camera_point_cloud[cam_name][i]

            if voxel_size != 0:
                condition = pc[:, 2] < 2
                pc = pc[condition, :]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
                if pc.shape[1] != 6:
                    rgbs = pc[:, 3].view(np.uint32)
                    r = (np.right_shift(rgbs, 16) % 256)[:, np.newaxis]
                    g = (np.right_shift(rgbs, 8) % 256)[:, np.newaxis]
                    b = (rgbs % 256)[:, np.newaxis]
                    r_g_b = np.concatenate([r, g, b], axis=-1)
                    pcd.colors = o3d.utility.Vector3dVector(r_g_b.astype(np.float64))
                else:
                    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:].astype(np.float64))

                downsampled_cloud = pcd.voxel_down_sample(voxel_size)
                if use_farthest_point_down_sample and len(downsampled_cloud.points) > point_num:
                    downsampled_cloud = downsampled_cloud.farthest_point_down_sample(point_num)

                pc = np.concatenate([downsampled_cloud.points, downsampled_cloud.colors], axis=-1)
                if pc.shape[0] > point_num:
                    idxs = np.random.choice(pc.shape[0], point_num, replace=False)
                    pc = pc[idxs]
                elif pc.shape[0] < point_num:
                    if pc.shape[0] == 0:
                        pc = np.zeros([1, 4], dtype=np.float32)
                    idxs1 = np.arange(pc.shape[0])
                    idxs2 = np.random.choice(pc.shape[0], point_num - pc.shape[0], replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                    pc = pc[idxs]
            else:
                condition = pc[:, 2] < 2
                pc = pc[condition, :]
                if pc.shape[0] >= point_num:
                    idxs = np.random.choice(pc.shape[0], point_num, replace=False)
                elif pc.shape[0] < point_num:
                    idxs1 = np.arange(pc.shape[0])
                    idxs2 = np.random.choice(pc.shape[0], point_num - pc.shape[0], replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                if pc.shape[1] != 6:
                    # rgbs = pc[idxs][:, 3].view(np.uint64)
                    rgbs = pc[idxs][:, 3].view(np.uint32)
                    r = (np.right_shift(rgbs, 16) % 256)[:, np.newaxis]
                    g = (np.right_shift(rgbs, 8) % 256)[:, np.newaxis]
                    b = (rgbs % 256)[:, np.newaxis]
                    r_g_b = np.concatenate([r, g, b], axis=-1)
                    pc = np.concatenate([pc[idxs][:, :3], r_g_b], axis=-1)
                else:
                    pc = pc[idxs]

            if not np.array_equal(color_extrinsics[cam_id], point_cloud_extrinsics[cam_id]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
                pc[:, 3:] = pc[:, 3:] / 255
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
                pcd.transform(np.dot(np.linalg.inv(color_extrinsics[cam_id]), point_cloud_extrinsics[cam_id]))
                pcd.colors = o3d.utility.Vector3dVector(
                    (np.asarray(pcd.colors) * 255).astype(np.float64))
                pc = np.concatenate([pcd.points, pcd.colors], axis=-1)

            if augment_point_cloud:
                # t = random.randint(0, 10)
                # for _ in range(t):
                #     center_point_idx = random.randint(0, pc.shape[0] - 1)
                #     dist = pc[center_point_idx, 2] / 2 * 0.20
                #     width = random.random() * dist
                #     height = random.random() * dist
                #     condition = (np.array(pc[:, 0] > (pc[center_point_idx, 0] + width / 2)) | np.array(
                #         pc[:, 0] < (pc[center_point_idx, 0] - width / 2))) | \
                #                 (np.array(pc[:, 1] > (pc[center_point_idx, 1] + height / 2)) | np.array(
                #                     pc[:, 1] < (pc[center_point_idx, 1] - height / 2)))
                #     condition = np.logical_not(condition)
                #     replace = np.random.uniform(-2, 2, 3)
                #     replace[2] = np.random.uniform(0, 2)
                #     pc[condition, :3] *= replace

                t = random.randint(0, 200)
                indexs = np.random.randint(0, pc.shape[0], t)
                replace = np.random.uniform(0, 1, (t, 6))
                replace[:, :2] = np.random.uniform(-1, 1, (t, 2))
                replace[:, :3] *= 2
                replace[:, 3:] *= 255
                pc[indexs, :] = replace

                t = random.randint(0, 1000)
                indexs = np.random.randint(0, pc.shape[0], t)
                replace = np.random.uniform(0, 2, (t, 3))
                replace[:, 1] = replace[:, 0]
                replace[:, 2] = replace[:, 0]
                pc[indexs, 3:] *= replace
                pc[indexs, 3:] = np.clip(pc[indexs, 3:], 0, 255)

                t = random.randint(0, 200)
                indexs = np.random.randint(0, pc.shape[0], t)
                pc[indexs, 3:] = np.random.uniform(0, 1, (t, 3)) * 255

            index = 6 if use_camera_point_cloud_rgb else 3
            camera_point_clouds.append(pc[:, :index])
            if camera_point_cloud_norm_mode != 0:
                scale = stats[f"camera_point_cloud_{cam_name}_scale"]
                offset = stats[f"camera_point_cloud_{cam_name}_offset"]
                mean = stats[f"camera_point_cloud_{cam_name}_mean"]
                std = stats[f"camera_point_cloud_{cam_name}_std"]
                if camera_point_cloud_norm_mode == 1:
                    camera_point_clouds[-1] = camera_point_clouds[-1] * scale[:index] + offset[:index]
                elif camera_point_cloud_norm_mode == 2:
                    camera_point_clouds[-1] = (camera_point_clouds[-1] - mean[:index]) / std[:index]
                else:
                    camera_point_clouds[-1] = (camera_point_clouds[-1] * scale[:index] + offset[:index] - mean[:index]) / std[:index]
    camera_point_clouds = torch.from_numpy(np.stack(camera_point_clouds, axis=0)).float().cuda().unsqueeze(0)
    return camera_point_clouds


def point_cloud_to_numpy(point_cloud):
    # points = point_cloud2.read_points_list(point_cloud, field_names=("x", "y", "z", "rgb"))
    # result = []
    # for point in points:
    #     result.append([point.x, point.y, point.z, point.rgb])
    # return np.array(result)  # , dtype=np.float32
    pc_data2 = ros_numpy.numpify(point_cloud)
    pc_x = pc_data2.flatten()[:]["x"]
    pc_y = pc_data2.flatten()[:]["y"]
    pc_z = pc_data2.flatten()[:]["z"]
    pc_rgb = pc_data2.flatten()[:]["rgb"]
    pc_array = np.vstack([pc_x, pc_y, pc_z, pc_rgb]).T
    return pc_array


def inference_process(args, stats, t, ros_operator, policy, tokenizer, encoder,
                      next_actions_joint_state=None, next_actions_end_pose=None, next_actions_robot_base=None):
    global inference_lock
    global inference_actions
    global inference_category
    global inference_timestep
    global pre_instruction
    global pre_instruction_attention_vector
    global pre_instruction_input_ids
    global pre_instruction_attention_mask
    bridge = CvBridge()

    (instruction,
     camera_colors, camera_depths, camera_point_clouds,
     arm_joint_states, arm_end_poses,  robot_base_vels, last_ctrl_arm_end_poses) = ros_operator.get_frame()

    camera_color = None
    camera_color_dict = dict()
    if args.use_camera_color_depth_to_point_cloud or args.use_camera_color:
        for j in range(len(args.camera_color_names)):
            camera_color_dict[args.camera_color_names[j]] = [bridge.imgmsg_to_cv2(camera_colors[i][j], 'bgr8') for i in range(args.obs_history_num)]
        if args.use_camera_color:
            camera_color = get_camera_color(stats, args.camera_color_names, args.obs_history_num, args.augment_color, camera_color_dict)

    camera_depth = None
    camera_depth_dict = dict()
    if args.use_camera_color_depth_to_point_cloud or args.use_camera_depth:
        for j in range(len(args.camera_depth_names)):
            if np.array_equal(ros_operator.get_camera_color_extrinsic(j), ros_operator.get_camera_depth_extrinsic(j)):
                camera_depth_dict[args.camera_depth_names[j]] = [bridge.imgmsg_to_cv2(camera_depths[i][j], 'passthrough') for i in range(args.obs_history_num)]
            else:
                camera_depth_dict[args.camera_depth_names[j]] = [depth_to_color_projection(bridge.imgmsg_to_cv2(camera_depths[i][j], 'passthrough'),
                                                                                           ros_operator.get_camera_color_intrinsic(j), ros_operator.get_camera_depth_intrinsic(j),
                                                                                           np.dot(np.linalg.inv(ros_operator.get_camera_color_extrinsic(j)), ros_operator.get_camera_depth_extrinsic(j))) for i in range(args.obs_history_num)]
        if args.use_camera_depth:
            camera_depth = get_camera_depth(stats, args.camera_depth_names, args.obs_history_num, args.camera_depth_norm_mode, camera_depth_dict)

    camera_point_cloud = None
    camera_point_cloud_dict = dict()
    color_extrinsics = []
    point_cloud_extrinsics = []
    if args.use_camera_point_cloud:
        if not args.use_camera_color_depth_to_point_cloud:
            for j in range(len(args.camera_point_cloud_names)):
                camera_point_cloud_dict[args.camera_point_cloud_names[j]] = [point_cloud_to_numpy(camera_point_clouds[i][j]) for i in range(args.obs_history_num)]
                color_extrinsics.append(ros_operator.get_camera_color_extrinsic(j))
                point_cloud_extrinsics.append(ros_operator.get_camera_point_cloud_extrinsic(j))
        else:
            for j in range(len(args.camera_point_cloud_names)):
                color_intrinsic = ros_operator.get_camera_color_intrinsic(j)
                depth_intrinsic = ros_operator.get_camera_color_intrinsic(j)
                color_extrinsic = ros_operator.get_camera_color_extrinsic(j)
                depth_extrinsic = ros_operator.get_camera_color_extrinsic(j)
                color_extrinsics.append(color_extrinsic)
                point_cloud_extrinsics.append(depth_extrinsic)
                camera_point_cloud_dict[args.camera_point_cloud_names[j]] = [color_depth_to_point_cloud(camera_color_dict[args.camera_color_names[j]][i], camera_depth_dict[args.camera_depth_names[j]][i],
                                                                                                        color_intrinsic, depth_intrinsic,
                                                                                                        color_extrinsic, depth_extrinsic) for i in range(args.obs_history_num)]
        camera_point_cloud = get_camera_point_cloud(stats, args.camera_point_cloud_names, args.obs_history_num,
                                                    args.camera_point_cloud_voxel_size, args.use_farthest_point_down_sample,
                                                    args.camera_point_cloud_point_num, args.augment_point_cloud,
                                                    args.use_camera_point_cloud_rgb, args.camera_point_cloud_norm_mode,
                                                    camera_point_cloud_dict, color_extrinsics, point_cloud_extrinsics)

    qpos_joint_state = None
    if args.use_arm_joint_state % 2 == 1:
        qpos_joint_state = [np.concatenate([np.array(arm_joint_states[i][j].position) for j in range(len(args.arm_joint_state_names))], axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_joint_state = np.concatenate(qpos_joint_state, axis=0)
        if args.qpos_ignore_grasp:
            for i in range(len(args.arm_joint_state_names)):
                qpos_joint_state[:, (i + 1) * args.arm_joint_state_dim - 1] = 0
        if args.augment_qpos:
            qpos_joint_state += np.random.uniform(-0.02, 0.02, qpos_joint_state.shape)
        qpos_joint_state = pre_process(stats, "joint_state", args.qpos_norm_mode, is_action=False, data=qpos_joint_state)
        qpos_joint_state = torch.from_numpy(qpos_joint_state).float().cuda().unsqueeze(0)

    qpos_end_pose = None
    if args.use_arm_end_pose % 2 == 1:
        qpos_end_pose = [np.concatenate(
            [np.array([arm_end_poses[i][j].pose.position.x, arm_end_poses[i][j].pose.position.y, arm_end_poses[i][j].pose.position.z,
                       arm_end_poses[i][j].pose.orientation.x, arm_end_poses[i][j].pose.orientation.y, arm_end_poses[i][j].pose.orientation.z, arm_end_poses[i][j].pose.orientation.w])
                for j in range(len(args.arm_end_pose_names))],
            axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_end_pose = np.concatenate(qpos_end_pose, axis=0)
        if args.use_arm_end_pose_incre:
            for i in range(qpos_end_pose.shape[0]):
                for j in range(len(args.arm_end_pose_names)):
                    world_base_matrix = create_transformation_matrix(args.arm_base_link_in_world[j][0], args.arm_base_link_in_world[j][1], args.arm_base_link_in_world[j][2],
                                                                     args.arm_base_link_in_world[j][3], args.arm_base_link_in_world[j][4], args.arm_base_link_in_world[j][5])
                    end_pose_matrix = create_transformation_matrix(qpos_end_pose[i][j*args.arm_end_pose_dim+0], qpos_end_pose[i][j*args.arm_end_pose_dim+1], qpos_end_pose[i][j*args.arm_end_pose_dim+2],
                                                                   qpos_end_pose[i][j*args.arm_end_pose_dim+3], qpos_end_pose[i][j*args.arm_end_pose_dim+4], qpos_end_pose[i][j*args.arm_end_pose_dim+5])
                    xyzrpy = matrix_to_xyzrpy(np.dot(world_base_matrix, end_pose_matrix))
                    qpos_end_pose[i][j * args.arm_end_pose_dim + 0] = xyzrpy[0]
                    qpos_end_pose[i][j * args.arm_end_pose_dim + 1] = xyzrpy[1]
                    qpos_end_pose[i][j * args.arm_end_pose_dim + 2] = xyzrpy[2]
                    qpos_end_pose[i][j * args.arm_end_pose_dim + 3] = xyzrpy[3]
                    qpos_end_pose[i][j * args.arm_end_pose_dim + 4] = xyzrpy[4]
                    qpos_end_pose[i][j * args.arm_end_pose_dim + 5] = xyzrpy[5]
            qpos_end_pose = calc_relative_pose(qpos_end_pose, num=len(args.arm_end_pose_names), dim=args.arm_end_pose_dim)
        if args.qpos_ignore_grasp:
            for i in range(len(args.arm_end_pose_names)):
                qpos_end_pose[:, (i + 1) * args.arm_end_pose_dim - 1] = 0
        if args.augment_qpos:
            qpos_end_pose += np.random.uniform(-0.02, 0.02, qpos_end_pose.shape)
        qpos_end_pose = pre_process(stats, "end_pose", args.qpos_norm_mode, is_action=False, data=qpos_end_pose)
        qpos_end_pose = torch.from_numpy(qpos_end_pose).float().cuda().unsqueeze(0)

    qpos_robot_base = None
    if args.use_robot_base % 2 == 1:
        qpos_robot_base = [np.array([robot_base_vels[i][0].twist.twist.linear.x,
                           robot_base_vels[i][0].twist.twist.linear.y,
                           robot_base_vels[i][0].twist.twist.angular.z])[np.newaxis, :]
                           for i in range(args.obs_history_num)]
        qpos_robot_base = np.concatenate(qpos_robot_base, axis=0)
        if args.augment_qpos:
            qpos_robot_base += np.random.uniform(-0.02, 0.02, qpos_robot_base.shape)
        qpos_robot_base = pre_process(stats, "robot_base", args.qpos_norm_mode, is_action=False, data=qpos_robot_base)
        qpos_robot_base = torch.from_numpy(qpos_robot_base).float().cuda().unsqueeze(0)

    next_action_is_pad = None
    if args.next_action_num != 0:
        if args.use_arm_joint_state:
            padded_next_action_joint_state = np.zeros((args.next_action_num, qpos_joint_state.shape[0]), dtype=np.float32)
            next_action_is_pad = np.zeros(args.next_action_num)
            if next_actions_joint_state is not None:
                padded_next_action_joint_state[0:next_actions_joint_state.shape[0]] = next_actions_joint_state
                next_action_is_pad[next_actions_joint_state.shape[0]:] = 1
            next_actions_joint_state = torch.from_numpy(padded_next_action_joint_state).float().cuda().unsqueeze(0)
            next_action_is_pad = torch.from_numpy(next_action_is_pad).float().cuda().unsqueeze(0)
        if args.use_arm_end_pose:
            padded_next_action_end_pose = np.zeros((args.next_action_num, qpos_end_pose.shape[0]), dtype=np.float32)
            next_action_is_pad = np.zeros(args.next_action_num)
            if next_actions_end_pose is not None:
                padded_next_action_end_pose[0:next_actions_end_pose.shape[0]] = next_actions_end_pose
                next_action_is_pad[next_actions_end_pose.shape[0]:] = 1
            next_actions_end_pose = torch.from_numpy(padded_next_action_end_pose).float().cuda().unsqueeze(0)
            next_action_is_pad = torch.from_numpy(next_action_is_pad).float().cuda().unsqueeze(0)
        if args.use_robot_base:
            padded_next_action_robot_base = np.zeros((args.next_action_num, qpos_robot_base.shape[0]), dtype=np.float32)
            next_action_is_pad = np.zeros(args.next_action_num)
            if next_actions_robot_base is not None:
                padded_next_action_robot_base[0:next_actions_robot_base.shape[0]] = next_actions_robot_base
                next_action_is_pad[next_actions_robot_base.shape[0]:] = 1
            next_actions_robot_base = torch.from_numpy(padded_next_action_robot_base).float().cuda().unsqueeze(0)
            next_action_is_pad = torch.from_numpy(next_action_is_pad).float().cuda().unsqueeze(0)
    else:
        next_actions_joint_state = None
        next_actions_end_pose = None
        next_actions_robot_base = None

    if args.use_instruction:
        if instruction != pre_instruction:
            if instruction == "" or instruction == "null":
                instruction_attention_vector = np.random.normal(0, 1, (1, args.instruction_max_len, args.instruction_hidden_dim)).astype(np.float32)
                instruction_input_ids = None
                instruction_attention_mask = None
            else:
                result = tokenizer.encode_plus(
                    instruction,  # 输入文本
                    return_attention_mask=True,  # 返回 attention mask
                    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                    padding='max_length',
                    truncation=True,
                    max_length=32,
                    return_tensors='pt',  # 返回 pytorch tensor 格式的数据
                )
                instruction_input_ids = result['input_ids']
                instruction_attention_mask = result['attention_mask']
                instruction_attention_vector = encoder(instruction_input_ids, instruction_attention_mask)["last_hidden_state"]
                instruction_attention_vector = instruction_attention_vector.cpu().detach().numpy()
            pre_instruction = instruction
            pre_instruction_attention_vector = instruction_attention_vector
            pre_instruction_input_ids = instruction_input_ids
            pre_instruction_attention_mask = instruction_attention_mask
        else:
            instruction_attention_vector = pre_instruction_attention_vector
            instruction_input_ids = pre_instruction_input_ids
            instruction_attention_mask = pre_instruction_attention_mask
        instruction_attention_vector = torch.from_numpy(instruction_attention_vector).float().cuda()  # .unsqueeze(0)
        instruction_input_ids = instruction_input_ids.cuda()
        instruction_attention_mask = instruction_attention_mask.cuda()
    else:
        instruction_attention_vector = None
        instruction_input_ids = None
        instruction_attention_mask = None

    all_actions, category = policy(camera_color, camera_depth, camera_point_cloud,
                                   qpos_joint_state, qpos_end_pose, qpos_robot_base,
                                   next_actions_joint_state, next_actions_end_pose, next_actions_robot_base, next_action_is_pad,
                                   instruction_input_ids, instruction_attention_mask, instruction_attention_vector)
    if args.use_arm_end_pose_incre:
        if args.arm_end_pose_incre_mode == 2:
            actions = all_actions.cpu().detach().numpy()
            actions = actions.reshape((-1))
            end_pose_incre = post_process(stats, "end_pose", args.qpos_norm_mode, is_action=True, data=actions)
            end_pose_incre = end_pose_incre.reshape((1, -1, args.arm_end_pose_dim * len(args.arm_end_pose_names)))
        else:
            end_pose_incre = post_process(stats, "end_pose", args.qpos_norm_mode, is_action=True, data=all_actions.cpu().detach().numpy())
        all_actions = []
        for i in range(len(args.arm_end_pose_names)):
            # arm_base_end_pose = [arm_end_poses[-1][i].pose.position.x, arm_end_poses[-1][i].pose.position.y, arm_end_poses[-1][i].pose.position.z, arm_end_poses[-1][i].pose.orientation.x, arm_end_poses[-1][i].pose.orientation.y, arm_end_poses[-1][i].pose.orientation.z]
            # all_actions.append(decode_pose_by_incre(arm_base_end_pose, end_pose_incre[0][:, i*args.arm_end_pose_dim:(i+1)*args.arm_end_pose_dim], args.arm_end_pose_incre_mode))
            all_actions.append(decode_pose_by_incre(last_ctrl_arm_end_poses[-1][i], end_pose_incre[0][:, i*args.arm_end_pose_dim:(i+1)*args.arm_end_pose_dim], args.arm_end_pose_incre_mode))
        all_actions = np.concatenate(all_actions, axis=-1)
    inference_lock.acquire()
    if args.use_arm_end_pose_incre:
        inference_actions = all_actions[np.newaxis, :, :]
    else:
        inference_actions = all_actions.cpu().detach().numpy()
    inference_category = int(category.cpu().detach().numpy().item()) if category is not None else category
    inference_timestep = t
    inference_lock.release()


def model_inference(args, ros_operator):
    global inference_lock
    global inference_actions
    global inference_category
    global inference_timestep
    global inference_thread

    policy_list = []
    stats_list = []
    set_seed(1000)
    if type(args.ckpt_dir) is str:
        args.ckpt_dir = [args.ckpt_dir]
    for i in range(len(args.ckpt_dir)):
        policy = make_policy(args)
        ckpt_path = os.path.join(args.ckpt_dir[i], args.ckpt_name)
        state_dict = torch.load(ckpt_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
                continue
            if args.next_action_num == 0 and key in ["model.input_proj_next_action.weight",
                                                     "model.input_proj_next_action.bias"]:
                continue
            if 'robot_state' in key:
                key = key.replace('robot_state', 'qpos')
            if 'encoder_joint_proj' in key:
                key = key.replace('encoder_joint_proj', 'encoder_qpos_proj')
            if key in model_dict_mapping.list1:
                key = model_dict_mapping.list2[model_dict_mapping.list1.index(key)]
            if args.backbone.startswith("resnet") and key == "model.pos.weight":
                continue
            new_state_dict[key] = value
        loading_status = policy.deserialize(new_state_dict)
        if not loading_status:
            print("ckpt path not exist")
            return False

        policy.cuda()
        policy.eval()

        stats_path = os.path.join(args.ckpt_dir[i], args.ckpt_stats_name)
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        policy_list.append(policy)
        stats_list.append(stats)

    if args.use_instruction:
        tokenizer = BertTokenizer.from_pretrained(args.instruction_encoder_dir)
        encoder = BertModel.from_pretrained(args.instruction_encoder_dir)
    else:
        tokenizer = None
        encoder = None

    chunk_size = args.chunk_size
    joint_state0_zero = [[0, 0, 0, 0, 0, 0, 0.08] for _ in range(len(args.arm_joint_state_ctrl_topics))]
    joint_state1_zero = [[0, 0, 0, 0, 0, 0, 0.08] for _ in range(len(args.arm_joint_state_ctrl_topics))]

    joint_state0 = [[0.0055820800000000005, 0.8687984200000001, 0.0, -0.015280944000000001, -0.028747712, -0.010169852, 0.08] for _ in range(len(args.arm_joint_state_ctrl_topics))]
    joint_state1 = [[0.0055820800000000005, 0.8687984200000001, 0.0, -0.015280944000000001, -0.028747712, -0.010169852, 0.08] for _ in range(len(args.arm_joint_state_ctrl_topics))]
    end_pose0 = [[0.2408368057645897, 0.00287136069042693, -0.0009838999821004113, -0.024625887251749915, 0.7530362776143732, 0.006795072511917707, 0.08] for _ in range(len(args.arm_end_pose_ctrl_topics))]
    end_pose1 = [[0.2408368057645897, 0.00287136069042693, -0.0009838999821004113, -0.024625887251749915, 0.7530362776143732, 0.006795072511917707, 0.08] for _ in range(len(args.arm_end_pose_ctrl_topics))]
    robot_base_vel0 = [0, 0, 0]
    robot_base_vel1 = [0, 0, 0]
    if args.use_arm_joint_state > 1:
        ros_operator.arm_joint_state_ctrl_linear_interpolation_thread(joint_state0, True, calc_step=True)
    if args.use_arm_end_pose > 1:
        ros_operator.arm_joint_state_ctrl(joint_state0_zero)
        rate_tmp = rospy.Rate(1)
        rate_tmp.sleep()
        ros_operator.arm_end_pose_ctrl(end_pose0)
        ros_operator.arm_joint_state_ctrl(joint_state0)
        # ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(end_pose0, True, calc_step=True)
    if args.use_robot_base > 1:
        ros_operator.robot_base_vel_ctrl(robot_base_vel0)
    pre_inference_status = -1
    ctrl_rate = rospy.Rate(10)

    all_actions = None
    category = None

    with torch.inference_mode():
        while not rospy.is_shutdown():

            t = 0
            max_t = 0
            if inference_thread is not None:
                inference_thread.join()
                inference_thread = None
            if args.temporal_agg:
                state_dim = 0 + (args.arm_joint_state_dim * len(args.arm_joint_state_names) if args.use_arm_joint_state > 1 else 0) + \
                                (args.arm_end_pose_dim * len(args.arm_end_pose_names) if args.use_arm_end_pose > 1 else 0) + \
                                (args.robot_base_dim if args.use_robot_base > 1 else 0)
                all_time_actions = np.zeros([args.max_publish_step, args.max_publish_step + args.chunk_size, state_dim])

            while t < args.max_publish_step and not rospy.is_shutdown():
                inference_status = ros_operator.get_inference_status()
                # print("inference_status:", inference_status, pre_inference_status)
                if inference_status == -1:
                    if len(args.ckpt_dir) == 1:
                        input("Please press any key to start inference:")
                        ros_operator.set_inference_status(0)
                        continue
                    if pre_inference_status != inference_status:
                        if args.use_arm_joint_state > 1:
                            ros_operator.arm_joint_state_ctrl_linear_interpolation_thread(joint_state0, True, calc_step=True)
                        if args.use_arm_end_pose > 1:
                            ros_operator.arm_joint_state_ctrl(joint_state0_zero)
                            rate_tmp = rospy.Rate(1)
                            rate_tmp.sleep()
                            ros_operator.arm_end_pose_ctrl(end_pose0)
                            ros_operator.arm_joint_state_ctrl(joint_state0)
                            # ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(end_pose0, True, calc_step=True)
                        if args.use_robot_base > 1:
                            ros_operator.robot_base_vel_ctrl(robot_base_vel0)
                        pre_inference_status = inference_status
                    ctrl_rate.sleep()
                    break
                else:
                    if pre_inference_status != inference_status:
                        if args.use_arm_joint_state > 1:
                            ros_operator.arm_joint_state_ctrl_linear_interpolation_thread(joint_state1, True, calc_step=True)
                        if args.use_arm_end_pose > 1:
                            ros_operator.arm_end_pose_ctrl(end_pose1)
                            ros_operator.arm_joint_state_ctrl(joint_state1)
                            # ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(end_pose1, True, calc_step=True)
                        if args.use_robot_base > 1:
                            ros_operator.robot_base_vel_ctrl(robot_base_vel1)
                        pre_inference_status = inference_status
                        break
                    policy = policy_list[inference_status]
                    stats = stats_list[inference_status]
                is_new_action = False
                if args.next_action_num != 0:
                    if t == 0:
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, stats, t,
                                                                  ros_operator, policy, tokenizer, encoder))
                        inference_thread.start()
                    if t >= max_t:
                        if inference_thread is not None:
                            inference_thread.join()
                            inference_lock.acquire()
                            inference_thread = None
                            all_actions = inference_actions
                            category = inference_category
                            t_start = t
                            max_t = t_start + args.pos_lookahead_step
                            if args.temporal_agg:
                                all_time_actions[[t_start], t_start:t_start + args.chunk_size] = all_actions
                            is_new_action = True
                            inference_lock.release()
                            inference_thread = threading.Thread(target=inference_process,
                                                                args=(args, stats, t,
                                                                      ros_operator, policy, tokenizer, encoder,
                                                                      all_actions[0][:args.pos_lookahead_step]))
                            inference_thread.start()

                else:
                    if args.asynchronous_inference:
                        if inference_thread is None and (args.pos_lookahead_step == 0 or t % args.pos_lookahead_step == 0 or t >= max_t) and ros_operator.check_frame():
                            inference_thread = threading.Thread(target=inference_process,
                                                                args=(args, stats, t, ros_operator,
                                                                      policy, tokenizer, encoder))
                            inference_thread.start()

                        if inference_thread is not None and (not inference_thread.is_alive() or t >= max_t):
                            inference_thread.join()
                            inference_lock.acquire()
                            inference_thread = None
                            all_actions = inference_actions
                            category = inference_category
                            if category is not None:
                                print("category:", category)
                            t_start = inference_timestep
                            max_t = t_start + args.chunk_size
                            if args.temporal_agg:
                                all_time_actions[[t_start], t_start:t_start + args.chunk_size] = all_actions
                            is_new_action = True
                            inference_lock.release()
                    else:
                        if t >= max_t or args.pos_lookahead_step == 0 or t % args.pos_lookahead_step == 0:
                            inference_thread = threading.Thread(target=inference_process,
                                                                args=(args, stats, t,
                                                                      ros_operator, policy, tokenizer, encoder))
                            inference_thread.start()
                            inference_thread.join()
                            inference_lock.acquire()
                            inference_thread = None
                            all_actions = inference_actions
                            category = inference_category
                            t_start = inference_timestep
                            max_t = t_start + args.chunk_size
                            if args.temporal_agg:
                                all_time_actions[[t_start], t_start:t_start + chunk_size] = all_actions
                            is_new_action = True
                            inference_lock.release()
                if t >= max_t:
                    print("inference time error")
                    continue
                if args.temporal_agg:
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = exp_weights[:, np.newaxis]
                    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                else:
                    if args.next_action_num != 0:
                        raw_action = all_actions[:, t % args.pos_lookahead_step]
                    else:
                        raw_action = all_actions[:, t - t_start]
                    # if args.pos_lookahead_step != 0:
                    #     raw_action = all_actions[:, t % args.pos_lookahead_step]
                    # else:
                    #     raw_action = all_actions[:, t % chunk_size]

                raw_action = raw_action[0]
                if category is not None and category == 0:
                    if args.use_arm_joint_state > 1:
                        ros_operator.arm_joint_state_ctrl_linear_interpolation_thread(joint_state1, True, calc_step=True)
                    if args.use_arm_end_pose > 1:
                        ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(end_pose1, True, calc_step=True)
                    if args.use_robot_base > 1:
                        ros_operator.robot_base_vel_ctrl(robot_base_vel1)
                else:
                    if args.use_arm_joint_state > 1:
                        action_joint_state = raw_action[:args.arm_joint_state_dim * len(args.arm_joint_state_names)]
                        raw_action = raw_action[args.arm_joint_state_dim * len(args.arm_joint_state_names):]
                        action_joint_state = post_process(stats, "joint_state", args.qpos_norm_mode, is_action=True, data=action_joint_state)
                        action_joint_state = [action_joint_state[i*args.arm_joint_state_dim:(i+1)*args.arm_joint_state_dim] for i in range(len(args.arm_joint_state_names))]
                        ros_operator.arm_joint_state_ctrl_interpolation_thread(action_joint_state, args.blocking_publish, is_new_action if args.asynchronous_inference else False)
                    if args.use_arm_end_pose > 1:
                        action_end_pose = raw_action[:args.arm_end_pose_dim * len(args.arm_end_pose_names)]
                        raw_action = raw_action[args.arm_end_pose_dim * len(args.arm_end_pose_names):]
                        if not args.use_arm_end_pose_incre:
                            action_end_pose = post_process(stats, "end_pose", args.qpos_norm_mode, is_action=True, data=action_end_pose)
                        action_end_pose = [action_end_pose[i*args.arm_end_pose_dim:(i+1)*args.arm_end_pose_dim] for i in range(len(args.arm_end_pose_names))]
                        ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(action_end_pose, args.blocking_publish, calc_step=False)
                        # ros_operator.arm_end_pose_ctrl(action_end_pose)
                        # ctrl_rate.sleep()
                    if args.use_robot_base > 1:
                        action_robot_base = raw_action[:args.robot_base_dim]
                        action_robot_base = post_process(stats, "robot_base", args.qpos_norm_mode, is_action=True, data=action_robot_base)
                        ros_operator.robot_base_vel_ctrl(action_robot_base)
                print("t:", t)
                t += 1


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        self.instruction = args.instruction
        self.camera_color_deques = [BlockingDeque() for _ in range(len(args.camera_color_names))]
        self.camera_depth_deques = [BlockingDeque() for _ in range(len(args.camera_depth_names))]
        self.camera_point_cloud_deques = [BlockingDeque() for _ in range(len(args.camera_point_cloud_names))]
        self.arm_joint_state_deques = [BlockingDeque() for _ in range(len(args.arm_joint_state_names))]
        self.arm_end_pose_deques = [BlockingDeque() for _ in range(len(args.arm_end_pose_names))]
        self.robot_base_vel_deques = [BlockingDeque() for _ in range(len(args.robot_base_vel_names))]

        self.camera_color_intrinsics = [None for _ in range(len(self.args.camera_color_names))]
        self.camera_depth_intrinsics = [None for _ in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_intrinsics = [None for _ in range(len(self.args.camera_point_cloud_names))]
        self.camera_color_extrinsics = [None for _ in range(len(self.args.camera_color_names))]
        self.camera_depth_extrinsics = [None for _ in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_extrinsics = [None for _ in range(len(self.args.camera_point_cloud_names))]

        self.all_config_exist = False

        self.camera_color_history_list = []
        self.camera_depth_history_list = []
        self.camera_point_cloud_history_list = []
        self.arm_joint_state_history_list = []
        self.arm_end_pose_history_list = []
        self.robot_base_vel_history_list = []

        rospy.init_node('inference', anonymous=True)
        self.instruction_subscriber = rospy.Subscriber(self.args.instruction_topic, String, self.instruction_callback)
        self.camera_color_subscriber = [rospy.Subscriber(self.args.camera_color_topics[i], Image, partial(self.camera_color_callback, i), queue_size=1) for i in range(len(self.args.camera_color_names))]
        self.camera_depth_subscriber = [rospy.Subscriber(self.args.camera_depth_topics[i], Image, partial(self.camera_depth_callback, i), queue_size=1) for i in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_subscriber = [rospy.Subscriber(self.args.camera_point_cloud_topics[i], PointCloud2, partial(self.camera_point_cloud_callback, i), queue_size=1) for i in range(len(self.args.camera_point_cloud_names))]
        self.arm_joint_state_subscriber = [rospy.Subscriber(self.args.arm_joint_state_topics[i], JointState, partial(self.arm_joint_state_callback, i), queue_size=1) for i in range(len(self.args.arm_joint_state_names))]
        self.arm_end_pose_subscriber = [rospy.Subscriber(self.args.arm_end_pose_topics[i], PoseStamped, partial(self.arm_end_pose_callback, i), queue_size=1) for i in range(len(self.args.arm_end_pose_names))]
        self.robot_base_vel_subscriber = [rospy.Subscriber(self.args.robot_base_vel_topics[i], Odometry, partial(self.robot_base_vel_callback, i), queue_size=1) for i in range(len(self.args.robot_base_vel_names))]

        self.camera_color_config_subscriber = [rospy.Subscriber(self.args.camera_color_config_topics[i], CameraInfo, partial(self.camera_color_config_callback, i), queue_size=1) for i in range(len(self.args.camera_color_names))]
        self.camera_depth_config_subscriber = [rospy.Subscriber(self.args.camera_depth_config_topics[i], CameraInfo, partial(self.camera_depth_config_callback, i), queue_size=1) for i in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_config_subscriber = [rospy.Subscriber(self.args.camera_point_cloud_config_topics[i], CameraInfo, partial(self.camera_point_cloud_config_callback, i), queue_size=1) for i in range(len(self.args.camera_point_cloud_names))]

        self.arm_joint_state_ctrl_publisher = [rospy.Publisher(self.args.arm_joint_state_ctrl_topics[i], JointState, queue_size=1) for i in range(len(self.args.arm_joint_state_ctrl_topics))]
        self.arm_end_pose_ctrl_publisher = [rospy.Publisher(self.args.arm_end_pose_ctrl_topics[i], PoseStamped, queue_size=1) for i in range(len(self.args.arm_end_pose_ctrl_topics))]
        self.robot_base_vel_ctrl_publisher = rospy.Publisher(self.args.robot_base_vel_ctrl_topic, Twist, queue_size=1)

        self.inference_status_service = rospy.Service(self.args.aloha_inference_status_service, StatusSrv, self.change_inference_status)
        self.inference_status = -1

        self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread_return_lock = threading.Lock()
        self.arm_joint_state_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_joint_state = None

        self.k = 3
        self.times = np.array([i for i in range(self.k)])
        self.arm_joint_state_ctrl_history_list = []

        self.arm_end_pose_ctrl_thread = None
        self.arm_end_pose_ctrl_thread_return_lock = threading.Lock()
        self.arm_end_pose_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_end_poses = []
        self.arm_end_pose_ctrl_history_list = []

    def get_camera_color_intrinsic(self, index):
        return self.camera_color_intrinsics[index]

    def get_camera_depth_intrinsic(self, index):
        return self.camera_depth_intrinsics[index]

    def get_camera_point_cloud_intrinsic(self, index):
        return self.camera_point_cloud_intrinsics[index]

    def get_camera_color_extrinsic(self, index):
        return self.camera_color_extrinsics[index]

    def get_camera_depth_extrinsic(self, index):
        return self.camera_depth_extrinsics[index]

    def get_camera_point_cloud_extrinsic(self, index):
        return self.camera_point_cloud_extrinsics[index]

    def instruction_callback(self, msg):
        self.instruction = msg.data

    def camera_color_callback(self, index, msg):
        if self.camera_color_deques[index].size() >= 200:
            self.camera_color_deques[index].popleft()
        self.camera_color_deques[index].append(msg)

    def camera_depth_callback(self, index, msg):
        if self.camera_depth_deques[index].size() >= 200:
            self.camera_depth_deques[index].popleft()
        self.camera_depth_deques[index].append(msg)

    def camera_point_cloud_callback(self, index, msg):
        if self.camera_point_cloud_deques[index].size() >= 200:
            self.camera_point_cloud_deques[index].popleft()
        self.camera_point_cloud_deques[index].append(msg)

    def arm_joint_state_callback(self, index, msg):
        if self.arm_joint_state_deques[index].size() >= 200:
            self.arm_joint_state_deques[index].popleft()
        self.arm_joint_state_deques[index].append(msg)

    def arm_end_pose_callback(self, index, msg):
        if self.arm_end_pose_deques[index].size() >= 200:
            self.arm_end_pose_deques[index].popleft()
        self.arm_end_pose_deques[index].append(msg)

    def robot_base_vel_callback(self, index, msg):
        if self.robot_base_vel_deques[index].size() >= 200:
            self.robot_base_vel_deques[index].popleft()
        self.robot_base_vel_deques[index].append(msg)

    def camera_color_config_callback(self, index, msg):
        self.camera_color_intrinsics[index] = np.array(msg.K).reshape(3, 3)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(self.args.camera_color_parent_frame_ids[index], msg.header.frame_id, rospy.Time(), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform(self.args.camera_color_parent_frame_ids[index], msg.header.frame_id, rospy.Time())
                rot = euler_from_quaternion(rot)
                self.camera_color_extrinsics[index] = create_transformation_matrix(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(f'Failed to get transform: {e}')
                continue
        self.camera_color_config_subscriber[index].unregister()

    def camera_depth_config_callback(self, index, msg):
        self.camera_depth_intrinsics[index] = np.array(msg.K).reshape(3, 3)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(self.args.camera_depth_parent_frame_ids[index], msg.header.frame_id, rospy.Time(), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform(self.args.camera_depth_parent_frame_ids[index], msg.header.frame_id, rospy.Time())
                rot = euler_from_quaternion(rot)
                self.camera_depth_extrinsics[index] = create_transformation_matrix(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(f'Failed to get transform: {e}')
                continue
        self.camera_depth_config_subscriber[index].unregister()

    def camera_point_cloud_config_callback(self, index, msg):
        self.camera_point_cloud_intrinsics[index] = np.array(msg.K).reshape(3, 3)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(self.args.camera_point_cloud_parent_frame_ids[index], msg.header.frame_id, rospy.Time(), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform(self.args.camera_point_cloud_parent_frame_ids[index], msg.header.frame_id, rospy.Time())
                rot = euler_from_quaternion(rot)
                self.camera_point_cloud_extrinsics[index] = create_transformation_matrix(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(f'Failed to get transform: {e}')
                continue
        self.camera_point_cloud_config_subscriber[index].unregister()

    def interpolation_param(self, positions):
        positions = np.array(positions)
        # 构建矩阵A和向量b
        A = [np.ones_like(self.times)]
        for i in range(self.k - 1):
            A.append(self.times ** (i + 1))
        A = np.vstack(A).T
        b = positions
        # 解线性方程组得到多项式系数
        coeffs = np.linalg.solve(A, b)
        # 使用多项式系数计算给定时间的速度
        return coeffs

    def arm_joint_state_ctrl(self, joint_states):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [f'joint{i+1}' for i in range(self.args.arm_joint_state_dim)]
        self.last_ctrl_arm_joint_state = joint_states
        for i in range(len(joint_states)):
            joint_state_msg.position = joint_states[i]
            self.arm_joint_state_ctrl_publisher[i].publish(joint_state_msg)

    def arm_joint_state_ctrl_interpolation(self, joint_states, is_new_action):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        if is_new_action or max(joint_state_diff) > 0.5:
            append_to_history_list_step = 10
            hz = 200
            if max(joint_state_diff) > 0.5:
                step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            else:
                step = 50
            rate = rospy.Rate(hz)
            joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
            for i in range(1, len(joint_state_list)):
                if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                    return
                ctrl_joint_state = [joint_state_list[i][j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
                self.arm_joint_state_ctrl(ctrl_joint_state)
                if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                    self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
                rate.sleep()
            self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
            return

        if len(self.arm_joint_state_ctrl_history_list) == 0:
            for i in range(self.k):
                self.arm_joint_state_ctrl_history_list.append([last_ctrl_joint_state[j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))])
        self.arm_joint_state_ctrl_history_list.append(joint_states)
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        coeffs = [self.interpolation_param([self.arm_joint_state_ctrl_history_list[k][i][j] for k in range(self.k)]) for i in range(len(self.args.arm_joint_state_names)) for j in range(self.args.arm_joint_state_dim)]
        hz = 200
        step = 10
        rate = rospy.Rate(hz)
        for i in range(step):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [np.polyval(coeffs[j][::-1], (self.k - 2) + (i + 1) * (1.0 / step)) for j in range(len(coeffs))]
            self.arm_joint_state_ctrl([ctrl_joint_state[j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))])
            rate.sleep()

    def arm_joint_state_ctrl_linear_interpolation(self, joint_states, calc_step):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        hz = 200
        if calc_step:
            step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            step = 1 if step == 0 else step
        else:
            step = 10
        rate = rospy.Rate(hz)
        append_to_history_list_step = 10
        joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
        for i in range(1, len(joint_state_list)):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [joint_state_list[i][j * self.args.arm_joint_state_dim: (j + 1) * self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
            self.arm_joint_state_ctrl(ctrl_joint_state)
            if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
            rate.sleep()
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        return

    def arm_joint_state_ctrl_interpolation_thread(self, joint_states, block, is_new_action):
        if self.arm_joint_state_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_joint_state_ctrl_thread_return_lock.locked():
                self.arm_joint_state_ctrl_thread_return_lock.release()
            self.arm_joint_state_ctrl_thread.join()
            self.arm_joint_state_ctrl_thread_return_lock.acquire(False)
            self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread = threading.Thread(target=self.arm_joint_state_ctrl_interpolation,
                                                            args=(joint_states, is_new_action))
        self.arm_joint_state_ctrl_thread.start()
        if block:
            self.arm_joint_state_ctrl_thread.join()

    def arm_joint_state_ctrl_linear_interpolation_thread(self, joint_states, block, calc_step):
        if self.arm_joint_state_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_joint_state_ctrl_thread_return_lock.locked():
                self.arm_joint_state_ctrl_thread_return_lock.release()
            self.arm_joint_state_ctrl_thread.join()
            self.arm_joint_state_ctrl_thread_return_lock.acquire(False)
            self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread = threading.Thread(target=self.arm_joint_state_ctrl_linear_interpolation,
                                                            args=(joint_states, calc_step))
        self.arm_joint_state_ctrl_thread.start()
        if block:
            self.arm_joint_state_ctrl_thread.join()

    def arm_end_pose_ctrl(self, end_poses):
        # print("left:", left)
        # print("right:", right)
        end_pose_msg = PoseStamped()
        end_pose_msg.header = Header()
        end_pose_msg.header.frame_id = "map"
        end_pose_msg.header.stamp = rospy.Time.now()
        self.last_ctrl_arm_end_poses.append(end_poses)
        self.last_ctrl_arm_end_poses = self.last_ctrl_arm_end_poses[-self.args.obs_history_num:]
        for i in range(len(end_poses)):
            end_pose_msg.pose.position.x = end_poses[i][0]
            end_pose_msg.pose.position.y = end_poses[i][1]
            end_pose_msg.pose.position.z = end_poses[i][2]
            # q = quaternion_from_euler(end_poses[i][3], end_poses[i][4], end_poses[i][5])
            # end_pose_msg.pose.orientation.x = q[0]
            # end_pose_msg.pose.orientation.y = q[1]
            # end_pose_msg.pose.orientation.z = q[2]
            # end_pose_msg.pose.orientation.w = q[3]
            end_pose_msg.pose.orientation.x = end_poses[i][3]
            end_pose_msg.pose.orientation.y = end_poses[i][4]
            end_pose_msg.pose.orientation.z = end_poses[i][5]
            end_pose_msg.pose.orientation.w = end_poses[i][6]
            end_pose_msg.pose.orientation.w += self.args.gripper_offset[i]
            self.arm_end_pose_ctrl_publisher[i].publish(end_pose_msg)

    def arm_end_pose_ctrl_linear_interpolation(self, end_poses, calc_step):
        if len(self.last_ctrl_arm_end_poses) == 0:
            last_ctrl_end_pose = np.concatenate([
                np.array([self.arm_end_pose_deques[i].right().pose.position.x, self.arm_end_pose_deques[i].right().pose.position.y,
                          self.arm_end_pose_deques[i].right().pose.position.z,
                          self.arm_end_pose_deques[i].right().pose.orientation.x, self.arm_end_pose_deques[i].right().pose.orientation.y,
                          self.arm_end_pose_deques[i].right().pose.orientation.z, self.arm_end_pose_deques[i].right().pose.orientation.w])
                for i in range(len(self.args.arm_end_pose_names))], axis=0)
        else:
            last_ctrl_end_pose = np.concatenate(
                [np.array(end_poses) for end_poses in self.last_ctrl_arm_end_poses[-1]], axis=0)

        hz = 200
        if calc_step:
            ctrl_end_pose = np.concatenate([np.array(end_pose) for end_pose in end_poses], axis=0)
            end_pose_diff = ctrl_end_pose - last_ctrl_end_pose

            step_position = int(max([max(abs(end_pose_diff[i*self.args.arm_end_pose_dim: i*self.args.arm_end_pose_dim+3]) / np.array(self.args.arm_steps_length[:3])) for i in range(len(self.args.arm_end_pose_names))]))
            step_grasp = int(max([abs(end_pose_diff[(i+1)*self.args.arm_end_pose_dim-1]) / np.array(self.args.arm_steps_length[self.args.arm_end_pose_dim-1]) for i in range(len(self.args.arm_end_pose_names))]))
            step = max([step_grasp, step_position])
        else:
            step = 10
        rate = rospy.Rate(hz)

        ctrl_traj_xyzg = []
        ctrl_traj_xyzw = []
        for i in range(len(self.args.arm_end_pose_names)):
            ctrl_end_pose_xyzg = [end_poses[i][0], end_poses[i][1], end_poses[i][2], end_poses[i][6]]
            ctrl_end_pose_xyzw = quaternion_from_euler(end_poses[i][3], end_poses[i][4], end_poses[i][5])
            last_ctrl_end_pose_xyzg = [last_ctrl_end_pose[i*self.args.arm_end_pose_dim+0], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+1], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+2], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+6]]
            last_ctrl_end_pose_xyzw = quaternion_from_euler(last_ctrl_end_pose[i*self.args.arm_end_pose_dim+3], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+4], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+5])
            traj_xyzg = np.linspace(last_ctrl_end_pose_xyzg, ctrl_end_pose_xyzg, step + 1)[1:]
            traj_xyzw = [geometric_slerp(np.array(last_ctrl_end_pose_xyzw), np.array(ctrl_end_pose_xyzw), (j+1) / step) for j in range(step)]
            ctrl_traj_xyzg.append(traj_xyzg)
            ctrl_traj_xyzw.append(traj_xyzw)
        for i in range(step):
            if self.arm_end_pose_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_end_poses = []
            for j in range(len(self.args.arm_end_pose_names)):
                ctrl_rpy = euler_from_quaternion([ctrl_traj_xyzw[j][i][0], ctrl_traj_xyzw[j][i][1], ctrl_traj_xyzw[j][i][2], ctrl_traj_xyzw[j][i][3]])
                ctrl_end_pose = [ctrl_traj_xyzg[j][i][0], ctrl_traj_xyzg[j][i][1], ctrl_traj_xyzg[j][i][2],
                                 ctrl_rpy[0], ctrl_rpy[1], ctrl_rpy[2],
                                 ctrl_traj_xyzg[j][i][3]]
                ctrl_end_poses.append(ctrl_end_pose)
            self.arm_end_pose_ctrl(ctrl_end_poses)
            rate.sleep()

    def arm_end_pose_ctrl_linear_interpolation_thread(self, end_poses, block, calc_step):
        if self.arm_end_pose_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_end_pose_ctrl_thread_return_lock.locked():
                self.arm_end_pose_ctrl_thread_return_lock.release()
            self.arm_end_pose_ctrl_thread.join()
            self.arm_end_pose_ctrl_thread_return_lock.acquire(False)
            self.arm_end_pose_ctrl_thread = None
        self.arm_end_pose_ctrl_thread = threading.Thread(target=self.arm_end_pose_ctrl_linear_interpolation,
                                                         args=(end_poses, calc_step))
        self.arm_end_pose_ctrl_thread.start()
        if block:
            self.arm_end_pose_ctrl_thread.join()

    def robot_base_vel_ctrl(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[2]
        self.robot_base_vel_ctrl_publisher.publish(vel_msg)

    def get_frame(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and not self.all_config_exist:
            self.all_config_exist = True
            if any(item is None for item in self.camera_color_intrinsics):
                self.all_config_exist = False
                print("camera color intrinsic config not exist")
            if any(item is None for item in self.camera_depth_intrinsics):
                self.all_config_exist = False
                print("camera depth intrinsic config not exist")
            if not self.args.use_camera_color_depth_to_point_cloud and any(item is None for item in self.camera_point_cloud_intrinsics):
                self.all_config_exist = False
                print("camera point cloud intrinsic config not exist")
            if any(item is None for item in self.camera_color_extrinsics):
                self.all_config_exist = False
                print("camera color extrinsic config not exist")
            if any(item is None for item in self.camera_depth_extrinsics):
                self.all_config_exist = False
                print("camera depth extrinsic config not exist")
            if not self.args.use_camera_color_depth_to_point_cloud and any(item is None for item in self.camera_point_cloud_extrinsics):
                self.all_config_exist = False
                print("camera point cloud extrinsic config not exist")
            rate.sleep()

        camera_colors = [self.camera_color_deques[i].right() for i in range(len(self.args.camera_color_names))]
        camera_depths = [self.camera_depth_deques[i].right() for i in range(len(self.args.camera_depth_names))]
        camera_point_clouds = [self.camera_point_cloud_deques[i].right() for i in range(len(self.args.camera_point_cloud_names))] if not self.args.use_camera_color_depth_to_point_cloud else []
        arm_joint_states = [self.arm_joint_state_deques[i].right() for i in range(len(self.args.arm_joint_state_names))]
        arm_end_poses = [self.arm_end_pose_deques[i].right() for i in range(len(self.args.arm_end_pose_names))]
        robot_base_vels = [self.robot_base_vel_deques[i].right() for i in range(len(self.args.robot_base_vel_names))]
        frame_time = max([msg.header.stamp.to_sec() for msg in (camera_colors + camera_depths + camera_point_clouds +
                                                                arm_joint_states + arm_end_poses + robot_base_vels)])
        for i in range(len(self.args.camera_color_names)):
            closer_time_diff = math.inf
            while (self.camera_color_deques[i].size() > 0 and
                   abs(self.camera_color_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.camera_color_deques[i].left().header.stamp.to_sec() - frame_time)
                camera_colors[i] = self.camera_color_deques[i].popleft()
        for i in range(len(self.args.camera_depth_names)):
            closer_time_diff = math.inf
            while (self.camera_depth_deques[i].size() > 0 and
                   abs(self.camera_depth_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.camera_depth_deques[i].left().header.stamp.to_sec() - frame_time)
                camera_depths[i] = self.camera_depth_deques[i].popleft()
        if not self.args.use_camera_color_depth_to_point_cloud:
            for i in range(len(self.args.camera_point_cloud_names)):
                closer_time_diff = math.inf
                while (self.camera_point_cloud_deques[i].size() > 0 and
                    abs(self.camera_point_cloud_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                    closer_time_diff = abs(self.camera_point_cloud_deques[i].left().header.stamp.to_sec() - frame_time)
                    camera_point_clouds[i] = self.camera_point_cloud_deques[i].popleft()
        for i in range(len(self.args.arm_joint_state_names)):
            closer_time_diff = math.inf
            while (self.arm_joint_state_deques[i].size() > 0 and
                   abs(self.arm_joint_state_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.arm_joint_state_deques[i].left().header.stamp.to_sec() - frame_time)
                arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()
        for i in range(len(self.args.arm_end_pose_names)):
            closer_time_diff = math.inf
            while (self.arm_end_pose_deques[i].size() > 0 and
                   abs(self.arm_end_pose_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.arm_end_pose_deques[i].left().header.stamp.to_sec() - frame_time)
                arm_end_poses[i] = self.arm_end_pose_deques[i].popleft()
        for i in range(len(self.args.robot_base_vel_names)):
            closer_time_diff = math.inf
            while (self.robot_base_vel_deques[i].size() > 0 and
                   abs(self.robot_base_vel_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.robot_base_vel_deques[i].left().header.stamp.to_sec() - frame_time)
                robot_base_vels[i] = self.robot_base_vel_deques[i].popleft()

        # for i in range(len(self.args.camera_color_names)):
        #     while self.camera_color_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.camera_color_deques[i].popleft()
        #     camera_colors[i] = self.camera_color_deques[i].popleft()
        # for i in range(len(self.args.camera_depth_names)):
        #     while self.camera_depth_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.camera_depth_deques[i].popleft()
        #     camera_depths[i] = self.camera_depth_deques[i].popleft()
        # if not self.args.use_camera_color_depth_to_point_cloud:
        #     for i in range(len(self.args.camera_point_cloud_names)):
        #         while self.camera_point_cloud_deques[i].left().header.stamp.to_sec() < frame_time:
        #             self.camera_point_cloud_deques[i].popleft()
        #         camera_point_clouds[i] = self.camera_point_cloud_deques[i].popleft()
        # for i in range(len(self.args.arm_joint_state_names)):
        #     while self.arm_joint_state_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.arm_joint_state_deques[i].popleft()
        #     arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()
        # for i in range(len(self.args.arm_end_pose_names)):
        #     while self.arm_end_pose_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.arm_end_pose_deques[i].popleft()
        #     arm_end_poses[i] = self.arm_end_pose_deques[i].popleft()
        # for i in range(len(self.args.robot_base_vel_names)):
        #     while self.robot_base_vel_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.robot_base_vel_deques[i].popleft()
        #     robot_base_vels[i] = self.robot_base_vel_deques[i].popleft()

        if len(self.camera_color_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list = self.camera_color_history_list[-self.args.obs_history_num:]

        if len(self.camera_depth_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_depth_history_list.append(camera_depths)
        self.camera_depth_history_list.append(camera_depths)
        self.camera_depth_history_list = self.camera_depth_history_list[-self.args.obs_history_num:]

        if len(self.camera_point_cloud_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_point_cloud_history_list.append(camera_point_clouds)
        self.camera_point_cloud_history_list.append(camera_point_clouds)
        self.camera_point_cloud_history_list = self.camera_point_cloud_history_list[-self.args.obs_history_num:]

        if len(self.arm_joint_state_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list = self.arm_joint_state_history_list[-self.args.obs_history_num:]

        if len(self.arm_end_pose_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_end_pose_history_list.append(arm_end_poses)
        self.arm_end_pose_history_list.append(arm_end_poses)
        self.arm_end_pose_history_list = self.arm_end_pose_history_list[-self.args.obs_history_num:]

        if len(self.robot_base_vel_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.robot_base_vel_history_list.append(robot_base_vels)
        self.robot_base_vel_history_list.append(robot_base_vels)
        self.robot_base_vel_history_list = self.robot_base_vel_history_list[-self.args.obs_history_num:]

        return (self.instruction,
                self.camera_color_history_list, self.camera_depth_history_list, self.camera_point_cloud_history_list,
                self.arm_joint_state_history_list, self.arm_end_pose_history_list,
                self.robot_base_vel_history_list, self.last_ctrl_arm_end_poses)

    def check_frame(self):
        for i in range(len(self.args.camera_color_names)):
            if self.camera_color_deques[i].size() == 0:
                print(self.args.camera_color_topics[i], "has no data")
                return False
        for i in range(len(self.args.camera_depth_names)):
            if self.camera_depth_deques[i].size() == 0:
                print(self.args.camera_depth_topics[i], "has no data")
                return False
        if not self.args.use_camera_color_depth_to_point_cloud:
            for i in range(len(self.args.camera_point_cloud_names)):
                if self.camera_point_cloud_deques[i].size() == 0:
                    print(self.args.camera_point_cloud_topics[i], "has no data")
                    return False
        for i in range(len(self.args.arm_joint_state_names)):
            if self.arm_joint_state_deques[i].size() == 0:
                print(self.args.arm_joint_state_topics[i], "has no data")
                return False
        for i in range(len(self.args.arm_end_pose_names)):
            if self.arm_end_pose_deques[i].size() == 0:
                print(self.args.arm_end_pose_topics[i], "has no data")
                return False
        for i in range(len(self.args.robot_base_vel_names)):
            if self.robot_base_vel_deques[i].size() == 0:
                print(self.args.robot_base_vel_topics[i], "has no data")
                return False
        return True

    def change_inference_status(self, request):
        response = StatusSrvResponse()
        self.inference_status = request.status
        return response

    def get_inference_status(self):
        return self.inference_status

    def set_inference_status(self, status):
        self.inference_status = status


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', nargs='+', required=True)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000,
                        required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt',
                        required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name',
                        default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='RT',
                        required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
                        required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200,
                        required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=False, required=False)

    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1',
                        required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--instruction_topic', action='store', type=str, help='instruction_topic',
                        default="/instruction",
                        required=False)

    parser.add_argument('--camera_color_names', action='store', type=str, help='camera_color_names',
                        default=['pikaDepthCamera', 'pikaFisheyeCamera'],
                        required=False)
    parser.add_argument('--camera_color_parent_frame_ids', action='store', type=str, help='camera_color_parent_frame_ids',
                        default=['camera_link', 'camera_fisheye_link'],
                        required=False)
    parser.add_argument('--camera_color_topics', action='store', type=str, help='camera_color_topics',
                        default=['/camera/color/image_raw', '/camera_fisheye/color/image_raw'],
                        required=False)
    parser.add_argument('--camera_color_config_topics', action='store', type=str, help='camera_color_config_topics',
                        default=['/camera/color/camera_info', '/camera_fisheye/color/camera_info'],
                        required=False)
    parser.add_argument('--camera_depth_names', action='store', type=str, help='camera_depth_names',
                        default=['pikaDepthCamera'],
                        required=False)
    parser.add_argument('--camera_depth_parent_frame_ids', action='store', type=str, help='camera_depth_parent_frame_ids',
                        default=['camera_link'],
                        required=False)
    parser.add_argument('--camera_depth_topics', action='store', type=str, help='camera_depth_topics',
                        default=['/camera/aligned_depth_to_color/image_raw'],
                        required=False)
    parser.add_argument('--camera_depth_config_topics', action='store', type=str, help='camera_depth_config_topics',
                        default=['/camera/aligned_depth_to_color/camera_info'],
                        required=False)
    parser.add_argument('--use_camera_color_depth_to_point_cloud', action='store', type=bool, help='use_camera_color_depth_to_point_cloud',
                        default=True,
                        required=False)
    parser.add_argument('--camera_point_cloud_names', action='store', type=str, help='camera_point_cloud_names',
                        default=['pikaDepthCamera'],
                        required=False)
    parser.add_argument('--camera_point_cloud_parent_frame_ids', action='store', type=str, help='camera_point_cloud_parent_frame_ids',
                        default=['camera_link'],
                        required=False)
    parser.add_argument('--camera_point_cloud_topics', action='store', type=str, help='camera_point_cloud_topics',
                        default=['/camera/depth/color/points'],
                        required=False)
    parser.add_argument('--camera_point_cloud_config_topics', action='store', type=str, help='camera_point_cloud_config_topics',
                        default=['/camera/aligned_depth_to_color/camera_info'],
                        required=False)
    parser.add_argument('--arm_joint_state_names', action='store', type=str, help='arm_joint_state_names',
                        default=[],
                        required=False)
    parser.add_argument('--arm_joint_state_topics', action='store', type=str, help='arm_joint_state_topics',
                        default=[],
                        required=False)
    parser.add_argument('--arm_end_pose_names', action='store', type=str, help='arm_end_pose_names',
                        default=['piper'],
                        required=False)
    parser.add_argument('--arm_end_pose_topics', action='store', type=str, help='arm_end_pose_topics',
                        default=['/piper_FK/urdf_end_pose'],
                        required=False)
    parser.add_argument('--robot_base_vel_names', action='store', type=str, help='robot_base_vel_names',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_topics', action='store', type=str, help='robot_base_vel_topics',
                        default=[],
                        required=False)
    parser.add_argument('--arm_joint_state_ctrl_topics', action='store', type=str, help='arm_joint_state_ctrl_topics',
                        default=[],
                        required=False)
    parser.add_argument('--arm_end_pose_ctrl_topics', action='store', type=str, help='arm_end_pose_ctrl_topics',
                        default=['/piper_IK/ctrl_end_pose'],
                        required=False)
    parser.add_argument('--robot_base_vel_ctrl_topic', action='store', type=str, help='robot_base_vel_ctrl_topic',
                        default='/cmd_vel',
                        required=False)
    parser.add_argument('--gripper_offset', nargs='+', action='store', type=float, help='gripper_offset', default=[0], required=False)

    parser.add_argument('--use_camera_color', action='store', type=bool, help='use_camera_color', default=False, required=False)
    parser.add_argument('--use_camera_depth', action='store', type=bool, help='use_camera_depth', default=False, required=False)
    parser.add_argument('--camera_depth_norm_mode', action='store', type=int, help='camera_depth_norm_mode', default=3, required=False)
    parser.add_argument('--use_camera_point_cloud', action='store', type=bool, help='use_camera_point_cloud', default=True, required=False)
    parser.add_argument('--use_camera_point_cloud_rgb', action='store', type=bool, help='use_camera_point_cloud_rgb', default=True, required=False)
    parser.add_argument('--camera_point_cloud_norm_mode', action='store', type=int, help='camera_point_cloud_norm_mode', default=3, required=False)
    parser.add_argument('--use_robot_base', action='store', type=int, help='use_robot_base', default=0, required=False)
    parser.add_argument('--robot_base_dim', action='store', type=int, help='robot_base_dim', default=3, required=False)
    parser.add_argument('--robot_base_loss_weight', action='store', type=float, help='robot_base_loss_weight', default=1.0, required=False)
    parser.add_argument('--use_arm_joint_state', action='store', type=int, help='use_arm_joint_state', default=0, required=False)
    parser.add_argument('--arm_joint_state_dim', action='store', type=int, help='arm_joint_state_dim', default=7, required=False)
    parser.add_argument('--arm_joint_state_loss_weight', action='store', type=float, help='arm_joint_state_loss_weight', default=1.0, required=False)
    parser.add_argument('--use_arm_end_pose', action='store', type=int, help='use_arm_end_pose', default=3, required=False)
    parser.add_argument('--use_arm_end_pose_incre', action='store', type=bool, help='use_arm_end_pose_incre', default=True, required=False)
    parser.add_argument('--arm_end_pose_incre_mode', action='store', type=int, help='arm_end_pose_incre_mode', default=1, required=False)
    parser.add_argument('--arm_base_link_in_world', action='store', type=float, help='arm_base_link_in_world', default=[[0, 0, 0, 0, 0, 0]], required=False)
    parser.add_argument('--arm_end_pose_dim', action='store', type=int, help='arm_end_pose_dim', default=7, required=False)
    parser.add_argument('--arm_end_pose_loss_weight', action='store', type=float, help='arm_end_pose_loss_weight', default=1.0, required=False)
    parser.add_argument('--qpos_norm_mode', action='store', type=int, help='qpos_norm_mode', default=2, required=False)
    parser.add_argument('--augment_qpos', action='store', type=bool, help='augment_qpos', default=False, required=False)
    parser.add_argument('--qpos_ignore_grasp', action='store', type=bool, help='qpos_ignore_grasp', default=False, required=False)
    parser.add_argument('--use_future', action='store', type=bool, help='use_future', default=False, required=False)
    parser.add_argument('--future_loss_weight', action='store', type=float, help='future_loss_weight', default=1.0, required=False)

    parser.add_argument('--use_multi_camera_backbone', action='store', type=bool, help='use_multi_camera_backbone', default=True, required=False)
    parser.add_argument('--obs_history_num', action='store', type=int, help='obs_history_num', default=1, required=False)
    parser.add_argument('--use_instruction', action='store', type=bool, help='use_instruction', default=False, required=False)
    parser.add_argument('--instruction_hidden_dim', action='store', type=int, help='instruction_hidden_dim', default=768, required=False)
    parser.add_argument('--instruction_max_len', action='store', type=int, help='instruction_max_len', default=32, required=False)
    parser.add_argument('--instruction_encoder_dir', action='store', type=str, help='instruction_encoder_dir',
                        default='/home/agilex/aloha_ws/src/aloha-devel/text_encoder/encoder-bert-base-chinese', required=False)
    parser.add_argument('--instruction', action='store', type=str, help='instruction',
                        default='null', required=False)
    parser.add_argument('--use_instruction_film', action='store', type=bool, help='use_instruction_film',
                        default=False, required=False)
    parser.add_argument('--class_num', action='store', type=int, help='class_num', default=0, required=False)
    parser.add_argument('--use_qpos_film', action='store', type=bool, help='use_qpos_film',
                        default=False, required=False)
    parser.add_argument('--camera_point_cloud_point_num', action='store', type=int, help='camera_point_cloud_point_num', default=5000, required=False)
    parser.add_argument('--camera_point_cloud_voxel_size', action='store', type=float, help='camera_point_cloud_voxel_size', default=0.01, required=False)
    parser.add_argument('--use_farthest_point_down_sample', action='store', type=bool, help='use_farthest_point_down_sample', default=False, required=False)

    parser.add_argument('--aloha_inference_status_service', action='store', type=str,
                        help='aloha_inference_status_service',
                        default='/aloha/inference_status_service', required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=32, required=False)
    parser.add_argument('--next_action_num', action='store', type=int, help='next_action_num',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.03], required=False)
    parser.add_argument('--robot_base_steps_length', action='store', type=float, help='robot_base_steps_length',
                        default=[0.1, 0.1], required=False)
    parser.add_argument('--asynchronous_inference', action='store', type=bool, help='asynchronous_inference',
                        default=True, required=False)
    parser.add_argument('--preemptive_publishing', action='store', type=bool, help='preemptive_publishing',
                        default=False, required=False)
    parser.add_argument('--blocking_publish', action='store', type=bool, help='blocking_publish',
                        default=True, required=False)

    parser.add_argument('--use_dataset_action', action='store', type=bool, help='use_dataset_action',
                        default=True, required=False)
    parser.add_argument('--augment_color', action='store', type=bool, help='augment_color', default=False,
                        required=False)
    parser.add_argument('--augment_point_cloud', action='store', type=bool, help='augment_point_cloud', default=False,
                        required=False)

    # for Diffusion
    parser.add_argument('--use_transformer', action='store', type=bool, help='use_transformer', default=False, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=float, help='ema_power', default=0.75, required=False)
    parser.add_argument('--prediction_type', action='store', type=str, help='prediction_type sample epsilon', default='sample', required=False)

    # for RT
    parser.add_argument('--transformer_layer_num', action='store', type=int, help='transformer_layer_num', default=1, required=False)
    parser.add_argument('--use_diffusion', action='store', type=bool, help='use_diffusion', default=False, required=False)
    parser.add_argument('--use_rdt', action='store', type=bool, help='use_rdt', default=False, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    model_inference(args, ros_operator)


if __name__ == '__main__':
    main()
