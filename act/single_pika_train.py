import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, DiffusionPolicy, RTPolicy
from lr_scheduler import get_scheduler
import math
import IPython
e = IPython.embed

import sys
sys.path.append("./")


def train(args):
    set_seed(1)
    train_dataloader, val_dataloader, stats = load_data(args)

    # save dataset stats
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    stats_path = os.path.join(args.ckpt_dir, args.ckpt_stats_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_process(args, train_dataloader, val_dataloader)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(args):
    if args.policy_class == 'ACT':
        policy = ACTPolicy(args)
        if len(args.pretrain_ckpt_path) != 0:
            state_dict = torch.load(args.pretrain_ckpt_path)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
                    continue
                if args.next_action_num == 0 and key in ["model.input_proj_next_action.weight",
                                                         "model.input_proj_next_action.bias"]:
                    continue
                new_state_dict[key] = value
            loading_status = policy.deserialize(new_state_dict)
            if not loading_status:
                print("ckpt path not exist")
    elif args.policy_class == 'Diffusion':
        policy = DiffusionPolicy(args)
        if len(args.pretrain_ckpt_path) != 0:
            loading_status = policy.deserialize(torch.load(args.pretrain_ckpt_path))
            if not loading_status:
                print("ckpt path not exist")
    elif args.policy_class == 'RT':
        policy = RTPolicy(args)
        if len(args.pretrain_ckpt_path) != 0:
            loading_status = policy.deserialize(torch.load(args.pretrain_ckpt_path))
            if not loading_status:
                print("ckpt path not exist")
    else:
        raise NotImplementedError
    return policy


def make_optimizer(args, policy):
    if args.policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif args.policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    elif args.policy_class == 'RT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(args, policy, data):
    (camera_color_data, camera_depth_data, camera_point_cloud_data,
     qpos_joint_state_data, qpos_end_pose_data, qpos_robot_base_data,
     next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data, next_action_is_pad,
     instruction_input_ids, instruction_attention_mask, instruction_vector,
     action_joint_state_data, action_end_pose_data, action_robot_base_data, action_is_pad,
     class_index) = data

    action_is_pad = action_is_pad.cuda()
    if args.use_arm_joint_state:
        qpos_joint_state_data = qpos_joint_state_data.cuda()
        next_action_joint_state_data = next_action_joint_state_data.cuda()
        action_joint_state_data = action_joint_state_data.cuda()
    else:
        qpos_joint_state_data = None
        next_action_joint_state_data = None
        action_joint_state_data = None
    if args.use_arm_end_pose:
        qpos_end_pose_data = qpos_end_pose_data.cuda()
        next_action_end_pose_data = next_action_end_pose_data.cuda()
        action_end_pose_data = action_end_pose_data.cuda()
    else:
        qpos_end_pose_data = None
        next_action_end_pose_data = None
        action_end_pose_data = None
    if args.use_robot_base:
        qpos_robot_base_data = qpos_robot_base_data.cuda()
        next_action_robot_base_data = next_action_robot_base_data.cuda()
        action_robot_base_data = action_robot_base_data.cuda()
    else:
        qpos_robot_base_data = None
        next_action_robot_base_data = None
        action_robot_base_data = None
    if args.use_camera_color:
        camera_color_data = camera_color_data.cuda()
    else:
        camera_color_data = None
    if args.use_camera_depth:
        camera_depth_data = camera_depth_data.cuda()
    else:
        camera_depth_data = None
    if args.use_camera_point_cloud:
        camera_point_cloud_data = camera_point_cloud_data.cuda()
    else:
        camera_point_cloud_data = None
    if args.use_instruction:
        instruction_input_ids = instruction_input_ids.cuda()
        instruction_attention_mask = instruction_attention_mask.cuda()
        instruction_vector = instruction_vector.cuda()
    else:
        instruction_input_ids = None
        instruction_attention_mask = None
        instruction_vector = None
    if args.next_action_num == 0:
        next_action_joint_state_data = None
        next_action_end_pose_data = None
        next_action_robot_base_data = None
        next_action_is_pad = None
    else:
        next_action_is_pad = next_action_is_pad.cuda()
    if args.class_num != 0:
        class_index = class_index.cuda()
    else:
        class_index = None
    return policy(camera_color_data, camera_depth_data, camera_point_cloud_data,
                  qpos_joint_state_data, qpos_end_pose_data, qpos_robot_base_data,
                  next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data, next_action_is_pad,
                  instruction_input_ids, instruction_attention_mask, instruction_vector,
                  action_joint_state_data, action_end_pose_data, action_robot_base_data, action_is_pad, class_index)


def train_process(args, train_dataloader, val_dataloader):
    set_seed(args.seed)

    policy = make_policy(args)
    policy.cuda()
    optimizer = make_optimizer(args, policy)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_train_step*args.num_epochs // 10,
        num_training_steps=args.num_train_step*args.num_epochs,
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
        last_epoch=-1
    )

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(args.num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(args, policy, data)
                epoch_dicts.append(forward_dict)
                if batch_idx >= args.num_eval_step:
                    break
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(detach_dict(epoch_summary))

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.serialize()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(args, policy, data)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            train_history.append(detach_dict(forward_dict))
            if batch_idx >= args.num_train_step:
                break
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'policy_epoch_{epoch}_seed_{args.seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            plot_history(train_history, validation_history, epoch, args.ckpt_dir, args.seed)

    ckpt_path = os.path.join(args.ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(args.ckpt_dir, f'policy_epoch_{best_epoch}_seed_{args.seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {args.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, args.num_epochs, args.ckpt_dir, args.seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', nargs='+', required=True)
    parser.add_argument('--dataset_class', action='store', type=int, help='dataset_class', nargs='+', default=[], required=False)
    parser.add_argument('--class_num', action='store', type=int, help='class_num', default=0, required=False)
    parser.add_argument('--class_loss_weight', action='store', type=float, help='class_loss_weight', default=1, required=False)
    parser.add_argument('--ckpt_stats_dir', action='store', type=str, help='ckpt_stats_dir', default='', required=False)
    parser.add_argument('--sample_weights', action='store', type=float, help='sample_weights', nargs='+', default=None, required=False)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--pretrain_ckpt_path', action='store', type=str, help='pretrain_ckpt_path', default='', required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize, ACT, Diffusion, RT', default='RT', required=False)
    parser.add_argument('--train_batch_size', action='store', type=int, help='train_batch_size', default=32, required=False)
    parser.add_argument('--val_batch_size', action='store', type=int, help='val_batch_size', default=32, required=False)
    parser.add_argument('--train_ratio', action='store', type=float, help='train_ratio', default=0.9, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--num_eval_step', action='store', type=int, help='num_eval_step', default=1, required=False)
    parser.add_argument('--num_train_step', action='store', type=int, help='num_train_step', default=10, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=4e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=4e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=32, required=False)
    parser.add_argument('--next_action_num', action='store', type=int, help='next_action_num', default=0, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg',  action='store', type=bool, help='temporal_agg', default=True, required=False)

    # for Diffusion
    parser.add_argument('--use_transformer', action='store', type=bool, help='use_transformer', default=False, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=float, help='ema_power', default=0.75, required=False)
    parser.add_argument('--prediction_type', action='store', type=str, help='prediction_type sample epsilon', default='sample', required=False)

    # for RT
    parser.add_argument('--transformer_layer_num', action='store', type=int, help='transformer_layer_num', default=1, required=False)
    parser.add_argument('--use_diffusion', action='store', type=bool, help='use_diffusion', default=False, required=False)
    parser.add_argument('--use_rdt', action='store', type=bool, help='use_rdt', default=False, required=False)

    parser.add_argument('--camera_color_names', action='store', type=str, help='camera_color_names', nargs='+', default=['pikaDepthCamera', 'pikaFisheyeCamera'], required=False)
    parser.add_argument('--use_camera_color', action='store', type=bool, help='use_camera_color', default=True, required=False)
    parser.add_argument('--camera_depth_names', action='store', type=str, help='camera_depth_names', nargs='+', default=['pikaDepthCamera'], required=False)
    parser.add_argument('--use_camera_depth', action='store', type=bool, help='use_camera_depth', default=True, required=False)
    parser.add_argument('--camera_depth_norm_mode', action='store', type=int, help='camera_depth_norm_mode', default=3, required=False)
    parser.add_argument('--camera_point_cloud_names', action='store', type=str, help='camera_point_cloud_names', nargs='+', default=['pikaDepthCamera'], required=False)
    parser.add_argument('--use_camera_point_cloud', action='store', type=bool, help='use_camera_point_cloud', default=False, required=False)
    parser.add_argument('--use_camera_point_cloud_rgb', action='store', type=bool, help='use_camera_point_cloud_rgb', default=True, required=False)
    parser.add_argument('--camera_point_cloud_norm_mode', action='store', type=int, help='camera_point_cloud_norm_mode', default=3, required=False)
    parser.add_argument('--use_robot_base', action='store', type=int, help='use_robot_base', default=0, required=False)
    parser.add_argument('--robot_base_dim', action='store', type=int, help='robot_base_dim', default=3, required=False)
    parser.add_argument('--robot_base_loss_weight', action='store', type=float, help='robot_base_loss_weight', default=1.0, required=False)
    parser.add_argument('--arm_joint_state_names', action='store', type=str, help='arm_joint_state_names', default=[], nargs='+', required=False)
    parser.add_argument('--use_arm_joint_state', action='store', type=int, help='use_arm_joint_state', default=0, required=False)
    parser.add_argument('--arm_joint_state_dim', action='store', type=int, help='arm_joint_state_dim', default=7, required=False)
    parser.add_argument('--arm_joint_state_loss_weight', action='store', type=float, help='arm_joint_state_loss_weight', default=1.0, required=False)
    parser.add_argument('--arm_end_pose_names', action='store', type=str, help='arm_joint_state_names', default=['pika'], nargs='+', required=False)
    parser.add_argument('--use_arm_end_pose', action='store', type=int, help='use_arm_end_pose', default=3, required=False)
    parser.add_argument('--arm_end_pose_dim', action='store', type=int, help='arm_end_pose_dim', default=7, required=False)
    parser.add_argument('--use_arm_end_pose_incre', action='store', type=bool, help='use_arm_end_pose_incre', default=True, required=False)
    parser.add_argument('--arm_end_pose_incre_mode', action='store', type=int, help='arm_end_pose_incre_mode', default=1, required=False)
    parser.add_argument('--arm_end_pose_loss_weight', action='store', type=float, help='arm_end_pose_loss_weight', default=1.0, required=False)
    parser.add_argument('--qpos_norm_mode', action='store', type=int, help='qpos_norm_mode', default=2, required=False)
    parser.add_argument('--augment_qpos', action='store', type=bool, help='augment_qpos', default=False, required=False)
    parser.add_argument('--qpos_ignore_grasp', action='store', type=bool, help='qpos_ignore_grasp', default=False, required=False)
    parser.add_argument('--use_future', action='store', type=bool, help='use_future', default=False, required=False)
    parser.add_argument('--future_loss_weight', action='store', type=float, help='future_loss_weight', default=1.0, required=False)
    parser.add_argument('--use_dataset_action', action='store', type=bool, help='use_dataset_action', default=True, required=False)

    parser.add_argument('--use_multi_camera_backbone', action='store', type=bool, help='use_multi_camera_backbone', default=True, required=False)
    parser.add_argument('--obs_history_num', action='store', type=int, help='obs_history_num', default=1, required=False)
    parser.add_argument('--use_instruction', action='store', type=bool, help='use_instruction', default=False, required=False)
    parser.add_argument('--instruction_hidden_dim', action='store', type=int, help='instruction_hidden_dim', default=768, required=False)
    parser.add_argument('--instruction_max_len', action='store', type=int, help='instruction_max_len', default=32, required=False)
    parser.add_argument('--instruction_encoder_dir', action='store', type=str, help='instruction_encoder_dir',
                        default='/home/agilex/aloha_ws/src/aloha-devel/unsup-simcse-bert-base-uncased', required=False)
    parser.add_argument('--use_instruction_film', action='store', type=bool, help='use_instruction_film',
                        default=False, required=False)
    parser.add_argument('--use_qpos_film', action='store', type=bool, help='use_qpos_film',
                        default=False, required=False)
    parser.add_argument('--camera_point_cloud_z_limit', action='store', type=float, help='point_cloud_distance_limit', nargs='+', default=[0, 3], required=False)
    parser.add_argument('--camera_point_cloud_point_num', action='store', type=int, help='point_cloud_num', default=20000, required=False)

    parser.add_argument('--augment_images', action='store', type=bool, help='augment_images', default=False, required=False)
    parser.add_argument('--arm_delay_time', action='store', type=int, help='arm_delay_time', default=1, required=False)

    parser.add_argument('--use_ario', action='store', type=bool, help='use_ario', default=True, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    train(args)


if __name__ == '__main__':
    main()

