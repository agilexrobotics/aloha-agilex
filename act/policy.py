import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from detr.models.detr_vae import build_diffusion, build_act, build_rt
import torch
from einops import rearrange, reduce
import IPython
import numpy as np

e = IPython.embed


class DiffusionPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = None
        self.optimizer = None
        self.build_diffusion_model_and_optimizer(args)

    def build_diffusion_model_and_optimizer(self, args):
        model = build_diffusion(args)
        model.cuda()

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

        self.model = model
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

    def __call__(self, camera_color_data, camera_depth_data, camera_point_cloud_data,
                 qpos_joint_state_data, qpos_end_pose_data, qpos_robot_base_data,
                 next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data, next_action_is_pad,
                 instruction_input_ids, instruction_attention_mask, instruction_vector,
                 action_joint_state_data=None, action_end_pose_data=None, action_robot_base_data=None, action_is_pad=None,
                 class_index=None):
        qposes = []
        actions = []
        next_actions = []
        if self.args.use_arm_joint_state % 2 == 1:
            qposes.append(qpos_joint_state_data)
            next_actions.append(next_action_joint_state_data)
        if self.args.use_arm_end_pose % 2 == 1:
            qposes.append(qpos_end_pose_data)
            next_actions.append(next_action_end_pose_data)
        if self.args.use_robot_base % 2 == 1:
            qposes.append(qpos_robot_base_data)
            next_actions.append(next_action_robot_base_data)
        if self.args.use_arm_joint_state > 1:
            actions.append(action_joint_state_data)
        if self.args.use_arm_end_pose > 1:
            actions.append(action_end_pose_data)
        if self.args.use_robot_base > 1:
            actions.append(action_robot_base_data)
        qposes = torch.cat(qposes, axis=-1) if len(qposes) > 0 else None
        if action_is_pad is not None:
            actions = torch.cat(actions, axis=-1)
        else:
            actions = None
        if self.args.next_action_num != 0:
            next_actions = torch.cat(next_actions, axis=-1)
        else:
            next_actions = None
        result = self.model(camera_color_data, camera_depth_data, camera_point_cloud_data,
                            qposes, next_actions, next_action_is_pad,
                            instruction_input_ids, instruction_attention_mask, instruction_vector,
                            actions, action_is_pad)
        if action_is_pad is not None:
            # L2 loss
            loss = F.mse_loss(result['noise_pred'], result['noise'], reduction='none')
            # loss = F.l1_loss(result['noise_pred'], result['noise'], reduction='none')
            loss = (loss * ~action_is_pad.unsqueeze(-1)).mean()
            loss_dict = {'diffusion': loss}
            if self.args.class_num != 0 and self.args.use_instruction:
                loss_func = nn.CrossEntropyLoss(reduction='mean')
                loss_dict['class'] = loss_func(result['class_prob'], class_index.reshape(-1))
                loss += self.args.class_loss_weight * loss_dict['class']
            loss_dict['loss'] = loss
            return loss_dict
        else:
            return result['result'], result['class'] if 'class' in result else None

    def serialize(self):
        return self.model.serialize()

    def deserialize(self, model_dict):
        return self.model.deserialize(model_dict)


class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = None
        self.optimizer = None
        self.build_act_model_and_optimizer(args)

    def build_act_model_and_optimizer(self, args):
        model = build_act(args)
        model.cuda()

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, camera_color_data, camera_depth_data, camera_point_cloud_data,
                 qpos_joint_state_data, qpos_end_pose_data, qpos_robot_base_data,
                 next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data, next_action_is_pad,
                 instruction_input_ids, instruction_attention_mask, instruction_vector,
                 action_joint_state_data=None, action_end_pose_data=None, action_robot_base_data=None, action_is_pad=None,
                 class_index=None):
        qposes = []
        actions = []
        next_actions = []
        if self.args.use_arm_joint_state % 2 == 1:
            qposes.append(qpos_joint_state_data)
            next_actions.append(next_action_joint_state_data)
        if self.args.use_arm_end_pose % 2 == 1:
            qposes.append(qpos_end_pose_data)
            next_actions.append(next_action_end_pose_data)
        if self.args.use_robot_base % 2 == 1:
            qposes.append(qpos_robot_base_data)
            next_actions.append(next_action_robot_base_data)
        if self.args.use_arm_joint_state > 1:
            actions.append(action_joint_state_data)
        if self.args.use_arm_end_pose > 1:
            actions.append(action_end_pose_data)
        if self.args.use_robot_base > 1:
            actions.append(action_robot_base_data)

        qposes = torch.cat(qposes, axis=-1) if len(qposes) > 0 else None
        if action_is_pad is not None:
            actions = torch.cat(actions, axis=-1)
        else:
            actions = None
        if self.args.next_action_num != 0:
            next_actions = torch.cat(next_actions, axis=-1)
        else:
            next_actions = None
        result = self.model(camera_color_data, camera_depth_data, camera_point_cloud_data,
                            qposes, next_actions, next_action_is_pad,
                            instruction_input_ids, instruction_attention_mask, instruction_vector,
                            actions, action_is_pad)
        if action_is_pad is None:
            return result['result'], result['class'] if 'class' in result else None
        else:
            loss_dict = dict()
            loss = 0
            if self.args.loss_function == 'l1':
                loss_func = F.l1_loss
            elif self.args.loss_function == 'l2':
                loss_func = F.mse_loss
            else:
                loss_func = F.smooth_l1_loss
            if self.args.use_arm_joint_state > 1:
                loss_joint_state = loss_func(action_joint_state_data, result['result'][:, :, :self.args.arm_joint_state_dim*len(self.args.arm_joint_state_names)], reduction='none')
                loss_joint_state = (loss_joint_state * ~action_is_pad.unsqueeze(-1)).mean()
                loss_dict['joint_state'] = loss_joint_state
                loss += self.args.arm_joint_state_loss_weight * loss_joint_state
            if self.args.use_arm_end_pose > 1:
                start = self.args.arm_joint_state_dim*len(self.args.arm_joint_state_names) if self.args.use_arm_joint_state else 0
                loss_end_pose = loss_func(action_end_pose_data, result['result'][:, :, start:start+self.args.arm_end_pose_dim*len(self.args.arm_end_pose_names)], reduction='none')
                loss_end_pose = (loss_end_pose * ~action_is_pad.unsqueeze(-1)).mean()
                loss_dict['end_pose'] = loss_end_pose
                loss += self.args.arm_end_pose_loss_weight * loss_end_pose
            if self.args.use_robot_base > 1:
                loss_robot_base = loss_func(action_robot_base_data, result['result'][:, :, -self.args.arm_robot_base_dim:], reduction='none')
                loss_robot_base = (loss_robot_base * ~action_is_pad.unsqueeze(-1)).mean()
                loss_dict['robot_base'] = loss_robot_base
                loss += self.args.robot_base_loss_weight * loss_robot_base
            if self.args.kl_weight != 0:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(result['mu'], result['logvar'])
                loss_dict['kl'] = total_kld[0]
                loss += self.args.kl_weight * loss_dict['kl']
            if self.args.class_num != 0 and self.args.use_instruction:
                loss_func = nn.CrossEntropyLoss(reduction='mean')
                loss_dict['class'] = loss_func(result['class_prob'], class_index.reshape(-1))
                loss += self.args.class_loss_weight * loss_dict['class']
            loss_dict['loss'] = loss
            return loss_dict

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


class RTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = None
        self.optimizer = None
        self.build_rt_model_and_optimizer(args)

    def build_rt_model_and_optimizer(self, args):
        model = build_rt(args)
        model.cuda()

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, camera_color_data, camera_depth_data, camera_point_cloud_data,
                 qpos_joint_state_data, qpos_end_pose_data, qpos_robot_base_data,
                 next_action_joint_state_data, next_action_end_pose_data, next_action_robot_base_data, next_action_is_pad,
                 instruction_input_ids, instruction_attention_mask, instruction_vector,
                 action_joint_state_data=None, action_end_pose_data=None, action_robot_base_data=None, action_is_pad=None,
                 class_index=None):
        qposes = []
        actions = []
        next_actions = []
        if self.args.use_arm_joint_state % 2 == 1:
            qposes.append(qpos_joint_state_data)
            next_actions.append(next_action_joint_state_data)
        if self.args.use_arm_end_pose % 2 == 1:
            qposes.append(qpos_end_pose_data)
            next_actions.append(next_action_end_pose_data)
        if self.args.use_robot_base % 2 == 1:
            qposes.append(qpos_robot_base_data)
            next_actions.append(next_action_robot_base_data)
        if self.args.use_arm_joint_state > 1:
            actions.append(action_joint_state_data)
        if self.args.use_arm_end_pose > 1:
            actions.append(action_end_pose_data)
        if self.args.use_robot_base > 1:
            actions.append(action_robot_base_data)

        qposes = torch.cat(qposes, axis=-1) if len(qposes) > 0 else None
        if action_is_pad is not None:
            actions = torch.cat(actions, axis=-1)
        else:
            actions = None
        if self.args.next_action_num != 0:
            next_actions = torch.cat(next_actions, axis=-1)
        else:
            next_actions = None
        if self.args.use_future and action_is_pad is not None:
            camera_color_future_index = [(j * self.args.obs_history_num * (2 if self.args.use_future else 1) + self.args.obs_history_num + i) for j in range(len(self.args.camera_color_names)) for i in range(self.args.obs_history_num)]
            camera_color_future_data = camera_color_data[:, camera_color_future_index] if self.args.use_camera_color else camera_color_data
            camera_depth_future_index = [(j * self.args.obs_history_num * (2 if self.args.use_future else 1) + self.args.obs_history_num + i) for j in range(len(self.args.camera_depth_names)) for i in range(self.args.obs_history_num)]
            camera_depth_future_data = camera_depth_data[:, camera_depth_future_index] if self.args.use_camera_depth else camera_depth_data
            camera_point_cloud_future_index = [(j * self.args.obs_history_num * (2 if self.args.use_future else 1) + self.args.obs_history_num + i) for j in range(len(self.args.camera_point_cloud_names)) for i in range(self.args.obs_history_num)]
            camera_point_cloud_future_data = camera_point_cloud_data[:, camera_point_cloud_future_index] if self.args.use_camera_point_cloud else camera_point_cloud_data
            qposes_future = qposes[:, -self.args.obs_history_num:]
            camera_color_index = [(j * self.args.obs_history_num * (2 if self.args.use_future else 1) + i) for j in range(len(self.args.camera_color_names)) for i in range(self.args.obs_history_num)]
            camera_color_data = camera_color_data[:, camera_color_index] if self.args.use_camera_color else camera_color_data
            camera_depth_index = [(j * self.args.obs_history_num * (2 if self.args.use_future else 1) + i) for j in range(len(self.args.camera_depth_names)) for i in range(self.args.obs_history_num)]
            camera_depth_data = camera_depth_data[:, camera_depth_index] if self.args.use_camera_depth else camera_depth_data
            camera_point_cloud_index = [(j * self.args.obs_history_num * (2 if self.args.use_future else 1) + i) for j in range(len(self.args.camera_point_cloud_names)) for i in range(self.args.obs_history_num)]
            camera_point_cloud_data = camera_point_cloud_data[:, camera_point_cloud_index] if self.args.use_camera_point_cloud else camera_point_cloud_data
            qposes = qposes[:, :self.args.obs_history_num]

        result = self.model(camera_color_data, camera_depth_data, camera_point_cloud_data,
                            qposes, next_actions, next_action_is_pad,
                            instruction_input_ids, instruction_attention_mask, instruction_vector,
                            actions)
        if action_is_pad is None:
            return result['result'], result['class'] if 'class' in result else None
        else:
            if self.args.use_diffusion:
                # L2 loss
                loss = F.mse_loss(result['noise_pred'], result['noise'], reduction='none')
                # loss = F.l1_loss(result['noise_pred'], result['noise'], reduction='none')
                loss = (loss * ~action_is_pad.unsqueeze(-1)).mean()
                loss_dict = {'diffusion': loss}
                if self.args.class_num != 0 and self.args.use_instruction:
                    loss_func = nn.CrossEntropyLoss(reduction='mean')
                    loss_dict['class'] = loss_func(result['class_prob'], class_index.reshape(-1))
                    loss += self.args.class_loss_weight * loss_dict['class']
                loss_dict['loss'] = loss
                return loss_dict
            else:
                loss_dict = dict()
                loss = 0
                if self.args.loss_function == 'l1':
                    loss_func = F.l1_loss
                elif self.args.loss_function == 'l2':
                    loss_func = F.mse_loss
                else:
                    loss_func = F.smooth_l1_loss
                if self.args.use_arm_joint_state > 1:
                    loss_joint_state = loss_func(action_joint_state_data, result['result'][:, :, :self.args.arm_joint_state_dim*len(self.args.arm_joint_state_names)], reduction='none')
                    loss_joint_state = (loss_joint_state * ~action_is_pad.unsqueeze(-1)).mean()
                    loss_dict['joint_state'] = loss_joint_state
                    loss += self.args.arm_joint_state_loss_weight * loss_joint_state
                if self.args.use_arm_end_pose > 1:
                    start = self.args.arm_joint_state_dim*len(self.args.arm_joint_state_names) if self.args.use_arm_joint_state else 0
                    loss_end_pose = loss_func(action_end_pose_data, result['result'][:, :, start:start+self.args.arm_end_pose_dim*len(self.args.arm_end_pose_names)], reduction='none')
                    loss_end_pose = (loss_end_pose * ~action_is_pad.unsqueeze(-1)).mean()
                    loss_dict['end_pose'] = loss_end_pose
                    loss += self.args.arm_end_pose_loss_weight * loss_end_pose
                if self.args.use_robot_base > 1:
                    loss_robot_base = loss_func(action_robot_base_data, result['result'][:, :, -self.args.arm_robot_base_dim:], reduction='none')
                    loss_robot_base = (loss_robot_base * ~action_is_pad.unsqueeze(-1)).mean()
                    loss_dict['robot_base'] = loss_robot_base
                    loss += self.args.robot_base_loss_weight * loss_robot_base
                if self.args.use_future:
                    result_future = self.model.future_forward(camera_color_future_data, camera_depth_future_data, camera_point_cloud_future_data,
                                                              qposes_future, next_actions, next_action_is_pad,
                                                              instruction_input_ids, instruction_attention_mask, instruction_vector)
                    result.update(result_future)
                    if self.args.use_camera_color:
                        loss_dict['color_future_loss'] = F.mse_loss(result['color'], result['color_future']).mean()
                        loss += self.args.future_loss_weight * loss_dict['color_future_loss']
                    if self.args.use_camera_depth:
                        loss_dict['depth_future_loss'] = F.mse_loss(result['depth'], result['depth_future']).mean()
                        loss += self.args.future_loss_weight * loss_dict['depth_future_loss']
                    if self.args.use_camera_point_cloud:
                        loss_dict['point_cloud_future_loss'] = F.mse_loss(result['point_cloud'], result['point_cloud_future']).mean()
                        loss += self.args.future_loss_weight * loss_dict['point_cloud_future_loss']
                if self.args.class_num != 0 and self.args.use_instruction:
                    loss_func = nn.CrossEntropyLoss(reduction='mean')
                    loss_dict['class'] = loss_func(result['class_prob'], class_index.reshape(-1))
                    loss += self.args.class_loss_weight * loss_dict['class']
                loss_dict['loss'] = loss
                return loss_dict

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
