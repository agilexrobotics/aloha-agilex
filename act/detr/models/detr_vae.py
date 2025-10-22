# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import Backbone, Joiner, build_position_encoding, DepthNet, Pointnet2Backbone, PointNetEncoderXYZ, PointNetEncoderXYZRGB
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer, OriginalTransformerEncoder, build_original_transformer_encoder

import numpy as np
from collections import OrderedDict
import sys
import os
# sys.path.append("/home/agilex/aloha_ws/src/aloha-devel/aloha")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + "/../../../../aloha")
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D, TransformerForDiffusion
# from diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from cleandiffuser.nn_condition import EarlyConvViTMultiViewImageCondition
from transformers import BertModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
import math
import re
import IPython
e = IPython.embed


class ApproxGELU(nn.Module):
    """
    使用tanh近似实现的GELU激活函数
    """
    def forward(self, x):
        # GELU近似版本：0.5 * x * (1 + torch.tanh(sqrt(2 / pi) * (x + 0.044715 * torch.pow(x, 3))))
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def build_condition_adapter(projector_type, in_features, out_features):
    projector = None
    if projector_type == 'linear':
        projector = nn.Linear(in_features, out_features)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, mlp_depth):
                # modules.append(nn.GELU(approximate="tanh"))
                modules.append(ApproxGELU())
                modules.append(nn.Linear(out_features, out_features))
            projector = nn.Sequential(*modules)

    if projector is None:
        raise ValueError(f'Unknown projector type: {projector_type}')

    return projector


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def create_mlp(
        input_dim,
        output_dim,
        net_arch,
        activation_fn=nn.ReLU,
        squash_output=False,
):
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class RT(nn.Module):
    """ This is the decoder only transformer """

    def __init__(self, args, input_state_dim, output_state_dim,
                 backbones, depth_backbones, point_cloud_backbones,
                 instruction_encoder, transformer):
        super().__init__()
        self.args = args
        self.output_state_dim = output_state_dim
        self.input_state_dim = input_state_dim
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        self.instruction_encoder = instruction_encoder
        self.input_proj_instruction = None
        self.instruction_pos = None
        if self.instruction_encoder is not None:
            self.input_proj_instruction = nn.Linear(self.args.instruction_hidden_dim, self.hidden_dim)
            self.instruction_pos = nn.Embedding(self.args.instruction_max_len, self.hidden_dim)

        self.input_proj_qpos = None
        self.qpos_pos = None
        if self.input_state_dim != 0:
            self.input_proj_qpos = nn.Linear(self.input_state_dim, self.hidden_dim)
            self.qpos_pos = nn.Embedding(self.args.obs_history_num, self.hidden_dim)

        self.input_proj_next_action = None
        self.next_action_pos = None
        if self.args.next_action_num != 0 and self.input_state_dim != 0:
            self.input_proj_next_action = nn.Linear(self.input_state_dim, self.hidden_dim)
            self.next_action_pos = nn.Embedding(self.args.next_action_num, self.hidden_dim)

        self.backbones = backbones
        self.input_proj = None
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            if self.args.backbone.startswith("resnet"):
                if self.args.use_multi_camera_backbone:
                    self.input_proj = [nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1) for i in range(len(self.args.camera_color_names))]
                else:
                    self.input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
            else:
                self.pos = nn.Embedding(len(self.args.camera_color_names) * self.args.obs_history_num, self.hidden_dim)

        self.depth_backbones = depth_backbones
        if depth_backbones is not None:
            if self.args.backbone.startswith("resnet"):
                self.depth_backbones = nn.ModuleList(depth_backbones)
                if self.args.use_multi_camera_backbone:
                    for i in range(len(self.args.camera_depth_names)):
                        self.input_proj[i] = nn.Conv2d(backbones[0].num_channels + depth_backbones[0].num_channels, self.hidden_dim, kernel_size=1)
                else:
                    self.input_proj = nn.Conv2d(backbones[0].num_channels + depth_backbones[0].num_channels, self.hidden_dim, kernel_size=1)
        if self.input_proj is not None and self.args.use_multi_camera_backbone:
            self.input_proj = nn.ModuleList(self.input_proj)

        self.point_cloud_backbones = point_cloud_backbones
        self.point_cloud_pos = None
        if point_cloud_backbones is not None:
            self.point_cloud_backbones = nn.ModuleList(point_cloud_backbones)
            self.point_cloud_pos = nn.Embedding(len(self.args.camera_point_cloud_names) * self.args.obs_history_num, self.hidden_dim)

        self.action_head = nn.Linear(self.hidden_dim, self.output_state_dim)
        self.query_embed = nn.Embedding(self.args.chunk_size, self.hidden_dim)
        if self.args.class_num != 0 and self.args.use_instruction:
            self.class_head = nn.Linear(self.hidden_dim, self.args.class_num)

        if self.args.use_diffusion:
            # if self.args.use_rdt:
            #     self.input_proj_instruction = build_condition_adapter(
            #         "mlp2x_gelu",
            #         in_features=self.args.instruction_hidden_dim,
            #         out_features=self.hidden_dim
            #     )
            #     self.input_proj_image = build_condition_adapter(
            #         "mlp2x_gelu",
            #         in_features=512,
            #         out_features=self.hidden_dim
            #     )
            #     # A `state` refers to an action or a proprioception vector
            #     self.input_proj_qpos = build_condition_adapter(
            #         "mlp3x_gelu",
            #         in_features=input_state_dim,  # state + state mask (indicator)
            #         out_features=self.hidden_dim
            #     )
            self.t_embedder = TimestepEmbedder(self.hidden_dim)
            self.t_pos = nn.Embedding(1, self.hidden_dim)
            self.noise_mlp = nn.Linear(self.output_state_dim, self.hidden_dim)
            # Create the noise scheduler
            self.num_train_timesteps = 100
            self.num_inference_timesteps = 10
            self.prediction_type = self.args.prediction_type
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=self.prediction_type,
                clip_sample=False,
            )
            self.noise_scheduler_sample = DPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=self.prediction_type
            )


    def forward(self, camera_color, camera_depth, camera_point_cloud,
                qpos, next_actions, next_action_is_pad,
                instruction_input_ids, instruction_attention_mask, instruction_vector,
                actions=None, action_is_pad=None):
        bs = 0
        if camera_color is not None:
            bs = camera_color.shape[0]
        if camera_depth is not None:
            bs = camera_depth.shape[0]
        if camera_point_cloud is not None:
            bs = camera_point_cloud.shape[0]
        if qpos is not None:
            bs = qpos.shape[0]

        instruction_input = None
        if self.instruction_encoder is not None:
            instruction_input_ids = None if instruction_input_ids is None else instruction_input_ids[:, :self.args.instruction_max_len]
            instruction_attention_mask = None if instruction_attention_mask is None else instruction_attention_mask[:, :self.args.instruction_max_len]
            instruction_vector = self.instruction_encoder(instruction_input_ids, instruction_attention_mask)["last_hidden_state"] if type(self.instruction_encoder) is not int else instruction_vector
            instruction_vector = instruction_vector[:, :self.args.instruction_max_len, :]
            instruction_input = torch.transpose(self.input_proj_instruction(instruction_vector), 0, 1)

        all_cam_features = []
        all_cam_pos = []
        src = None
        src_pos = None
        if self.args.use_camera_color:
            for cam_id, cam_name in enumerate(self.args.camera_color_names):
                for i in range(self.args.obs_history_num):
                    if self.args.backbone.startswith("resnet"):
                        features, src_pos = (self.backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                                             (camera_color[:, cam_id * self.args.obs_history_num + i],
                                              qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                              instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                        features = features[0]
                        src_pos = src_pos[0]
                        if self.depth_backbones is not None and camera_depth is not None and cam_id < len(self.args.camera_depth_names):
                            features_depth = (self.depth_backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                                              (camera_depth[:, cam_id * self.args.obs_history_num + i].unsqueeze(dim=1),
                                               qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                               instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                            if self.args.use_multi_camera_backbone:
                                all_cam_features.append(self.input_proj[cam_id](torch.cat([features, features_depth], axis=1)))
                            else:
                                all_cam_features.append(self.input_proj(torch.cat([features, features_depth], axis=1)))
                        else:
                            if self.args.use_multi_camera_backbone:
                                all_cam_features.append(self.input_proj[cam_id](features))
                            else:
                                all_cam_features.append(self.input_proj(features))
                        all_cam_pos.append(src_pos)
                    else:
                        condition = {"image": [], "qpos": qpos if self.args.use_qpos_film and self.input_state_dim != 0 else None, "instruction": instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None}
                        input_color = torch.unsqueeze(camera_color[:, cam_id * self.args.obs_history_num + i], axis=1)
                        condition["image"].append(input_color)
                        if self.args.use_camera_depth:
                            input_depth = torch.unsqueeze(camera_depth[:, cam_id * self.args.obs_history_num + i], axis=1)
                            input_depth = torch.unsqueeze(input_depth, axis=1)
                            condition["image"].append(input_depth)
                        features = self.backbones[cam_id if self.args.use_multi_camera_backbone else 0](condition)
                        features = torch.unsqueeze(features, axis=1)
                        all_cam_features.append(features)
            if self.args.backbone.startswith("resnet"):
                src = torch.cat(all_cam_features, axis=3)
                src_pos = torch.cat(all_cam_pos, axis=3)
                src = src.flatten(2).permute(0, 2, 1)
                # if self.args.use_diffusion and self.args.use_rdt:
                #     src = self.input_proj_image(src)
                src_pos = src_pos.flatten(2).permute(0, 2, 1).squeeze(axis=0)
            else:
                src = torch.cat(all_cam_features, axis=1)
                src_pos = self.pos.weight
        # proprioception features
        qpos_input = self.input_proj_qpos(qpos).permute(1, 0, 2) if self.input_state_dim != 0 else None
        # qpos_input = torch.unsqueeze(qpos_input, axis=0)
        # qpos_input = torch.randn(qpos_input.shape).to(qpos_input.device)
        next_action_input = None
        if self.args.next_action_num != 0 and next_actions is not None and self.input_state_dim != 0:
            next_action_input = self.input_proj_next_action(next_actions).permute(1, 0, 2)

        if self.point_cloud_backbones is not None and camera_point_cloud is not None:
            point_cloud_features = []
            for cam_id, cam_name in enumerate(self.args.camera_point_cloud_names):
                for i in range(self.args.obs_history_num):
                    point_cloud_features.append(torch.unsqueeze(
                        self.point_cloud_backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                        (camera_point_cloud[:, cam_id * self.args.obs_history_num + i],
                         qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                         instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None), axis=1))
            point_cloud_intput = torch.cat(point_cloud_features, axis=1)
        else:
            point_cloud_intput = None

        if self.args.use_diffusion:
            result = {}
            if actions is not None:
                noise = torch.randn(actions.shape,  dtype=actions.dtype, device=actions.device)
                timesteps = torch.randint(
                    0, self.num_train_timesteps,
                    (bs,), device=actions.device
                ).long()

                noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
                result = self.transformer(bs, self.noise_mlp(noisy_actions).permute(1, 0, 2),
                                              timesteps if self.args.use_rdt else self.t_embedder(timesteps).unsqueeze(0), self.t_pos.weight,
                                              src, src_pos,
                                              None, None,
                                              # depth_features, depth_pos,
                                              point_cloud_intput, self.point_cloud_pos.weight if point_cloud_intput is not None else None,
                                              qpos_input, self.qpos_pos.weight if self.qpos_pos is not None else None,
                                              next_action_input, self.next_action_pos.weight if next_action_input is not None else None, next_action_is_pad,
                                              instruction_input, self.instruction_pos.weight if instruction_input is not None else None, instruction_attention_mask if instruction_input is not None else None)
                noise_pred = result['action'] if self.args.use_rdt else self.action_head(result['action'])
                result['noise_pred'] = noise_pred
                if self.args.prediction_type == 'sample':
                    result['noise'] = actions
                else:
                    result['noise'] = noise
            else:
                noisy_actions = torch.randn((bs, self.args.chunk_size, self.output_state_dim), device=qpos_input.device)
                self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
                for t in self.noise_scheduler_sample.timesteps:
                    result = self.transformer(bs, self.noise_mlp(noisy_actions).permute(1, 0, 2),
                                                  t.unsqueeze(0).to(qpos_input.device) if self.args.use_rdt else self.t_embedder(t.unsqueeze(0).unsqueeze(0).to(qpos_input.device)), self.t_pos.weight,
                                                  src, src_pos,
                                                  None, None,
                                                  # depth_features, depth_pos,
                                                  point_cloud_intput, self.point_cloud_pos.weight if point_cloud_intput is not None else None,
                                                  qpos_input, self.qpos_pos.weight if self.qpos_pos is not None else None,
                                                  next_action_input, self.next_action_pos.weight if next_action_input is not None else None, next_action_is_pad,
                                                  instruction_input, self.instruction_pos.weight if instruction_input is not None else None, instruction_attention_mask if instruction_input is not None else None)
                    noise_pred = result['action'] if self.args.use_rdt else self.action_head(result['action'])
                    noisy_actions = self.noise_scheduler_sample.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=noisy_actions
                    ).prev_sample
                result['result'] = noisy_actions
        else:
            result = self.transformer(bs, self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1),
                                      None, None,
                                      src, src_pos,
                                      None, None,
                                      # depth_features, depth_pos,
                                      point_cloud_intput, self.point_cloud_pos.weight if point_cloud_intput is not None else None,
                                      qpos_input, self.qpos_pos.weight if self.qpos_pos is not None else None,
                                      next_action_input, self.next_action_pos.weight if next_action_input is not None else None, next_action_is_pad,
                                      instruction_input, self.instruction_pos.weight if instruction_input is not None else None, instruction_attention_mask if instruction_input is not None else None)
            result['result'] = self.action_head(result['action'])
        if self.args.class_num != 0 and self.args.use_instruction:
            result['class_prob'] = self.class_head(result['instruction']).reshape(-1, self.args.class_num)
            _, result['class'] = torch.max(result['class_prob'], 1)
        return result

    def future_forward(self, camera_color, camera_depth, camera_point_cloud,
                       qpos, next_actions, next_action_is_pad,
                       instruction_input_ids, instruction_attention_mask, instruction_vector):
        all_cam_features = []
        src = None
        if self.args.use_camera_color:
            for cam_id, cam_name in enumerate(self.args.camera_color_names):
                for i in range(self.args.obs_history_num):
                    if self.args.backbone.startswith("resnet"):
                        features, _ = (self.backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                                       (camera_color[:, cam_id * self.args.obs_history_num + i],
                                        qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                        instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                        features = features[0]
                        if self.depth_backbones is not None and camera_depth is not None and cam_id < len(self.args.camera_depth_names):
                            features_depth = (self.depth_backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                                              (camera_depth[:, cam_id * self.args.obs_history_num + i].unsqueeze(dim=1),
                                               qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                               instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                            if self.args.use_multi_camera_backbone:
                                all_cam_features.append(self.input_proj[cam_id](torch.cat([features, features_depth], axis=1)))
                            else:
                                all_cam_features.append(self.input_proj(torch.cat([features, features_depth], axis=1)))
                        else:
                            if self.args.use_multi_camera_backbone:
                                all_cam_features.append(self.input_proj[cam_id](features))
                            else:
                                all_cam_features.append(self.input_proj(features))
                    else:
                        condition = {"image": [], "qpos": qpos if self.args.use_qpos_film and self.input_state_dim != 0 else None, "instruction": instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None}
                        input_color = torch.unsqueeze(camera_color[:, cam_id * self.args.obs_history_num + i], axis=1)
                        condition["image"].append(input_color)
                        if self.args.use_camera_depth:
                            input_depth = torch.unsqueeze(camera_depth[:, cam_id * self.args.obs_history_num + i], axis=1)
                            input_depth = torch.unsqueeze(input_depth, axis=1)
                            condition["image"].append(input_depth)
                        features = self.backbones[cam_id if self.args.use_multi_camera_backbone else 0](condition)
                        features = torch.unsqueeze(features, axis=1)
                        all_cam_features.append(features)
            if self.args.backbone.startswith("resnet"):
                src = torch.cat(all_cam_features, axis=3)
                src = src.flatten(2).permute(0, 2, 1)
            else:
                src = torch.cat(all_cam_features, axis=1)
        if self.point_cloud_backbones is not None and camera_point_cloud is not None:
            point_cloud_features = []
            for cam_id, cam_name in enumerate(self.args.camera_point_cloud_names):
                for i in range(self.args.obs_history_num):
                    point_cloud_features.append(torch.unsqueeze(
                        self.point_cloud_backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                        (camera_point_cloud[:, cam_id * self.args.obs_history_num + i],
                         qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                         instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None), axis=1))
            point_cloud_intput = torch.cat(point_cloud_features, axis=1)
        else:
            point_cloud_intput = None
        result = {}
        result['color_future'] = src
        result['point_cloud_future'] = point_cloud_intput
        return result


class ACT(nn.Module):
    def __init__(self, args, input_state_dim, output_state_dim,
                 backbones, depth_backbones, point_cloud_backbones,
                 instruction_encoder, transformer, encoder):
        super().__init__()
        self.args = args
        self.input_state_dim = input_state_dim
        self.output_state_dim = output_state_dim
        self.hidden_dim = transformer.d_model
        self.transformer = transformer

        self.encoder = encoder
        self.latent_dim = 32
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.latent_pos = nn.Embedding(1, self.hidden_dim)
        if self.args.kl_weight != 0:
            self.cls_embed = nn.Embedding(1, self.hidden_dim)
            self.encoder_action_proj = nn.Linear(self.output_state_dim, self.hidden_dim)
            self.encoder_qpos_proj = None
            self.encoder_next_action_proj = None
            if self.input_state_dim != 0:
                self.encoder_qpos_proj = nn.Linear(self.input_state_dim, self.hidden_dim)
                if self.args.next_action_num != 0:
                    self.encoder_next_action_proj = nn.Linear(self.input_state_dim, self.hidden_dim)
            self.latent_proj = nn.Linear(self.hidden_dim, self.latent_dim*2)
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1+(self.args.obs_history_num if self.input_state_dim != 0 else 0)+self.args.next_action_num+self.args.chunk_size, self.hidden_dim))

        self.instruction_encoder = instruction_encoder
        self.input_proj_instruction = None
        self.instruction_pos = None
        if self.instruction_encoder is not None:
            self.input_proj_instruction = nn.Linear(self.args.instruction_hidden_dim, self.hidden_dim)
            self.instruction_pos = nn.Embedding(self.args.instruction_max_len, self.hidden_dim)

        self.input_proj_qpos = None
        self.qpos_pos = None
        if self.input_state_dim != 0:
            self.input_proj_qpos = nn.Linear(self.input_state_dim, self.hidden_dim)
            self.qpos_pos = nn.Embedding(self.args.obs_history_num, self.hidden_dim)

        self.input_proj_next_action = None
        self.next_action_pos = None
        if self.args.next_action_num != 0:
            self.input_proj_next_action = nn.Linear(self.input_state_dim, self.hidden_dim)
            self.next_action_pos = nn.Embedding(self.args.next_action_num, self.hidden_dim)

        self.backbones = backbones
        self.input_proj = None
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            if self.args.backbone.startswith("resnet"):
                if self.args.use_multi_camera_backbone:
                    self.input_proj = [nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1) for i in range(len(self.args.camera_color_names))]
                else:
                    self.input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
            else:
                self.pos = nn.Embedding(len(self.args.camera_color_names) * self.args.obs_history_num, self.hidden_dim)

        self.depth_backbones = depth_backbones
        if depth_backbones is not None:
            if self.args.backbone.startswith("resnet"):
                self.depth_backbones = nn.ModuleList(depth_backbones)
                if self.args.use_multi_camera_backbone:
                    for i in range(len(self.args.camera_depth_names)):
                        self.input_proj[i] = nn.Conv2d(backbones[0].num_channels + depth_backbones[0].num_channels, self.hidden_dim, kernel_size=1)
                else:
                    self.input_proj = nn.Conv2d(backbones[0].num_channels + depth_backbones[0].num_channels, self.hidden_dim, kernel_size=1)
        if self.input_proj is not None and self.args.use_multi_camera_backbone:
            self.input_proj = nn.ModuleList(self.input_proj)

        self.point_cloud_backbones = point_cloud_backbones
        self.point_cloud_pos = None
        if point_cloud_backbones is not None:
            self.point_cloud_backbones = nn.ModuleList(point_cloud_backbones)
            self.point_cloud_pos = nn.Embedding(len(self.args.camera_point_cloud_names) * self.args.obs_history_num, self.hidden_dim)

        self.query_embed = nn.Embedding(self.args.chunk_size, self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.output_state_dim)
        if self.args.class_num != 0 and self.args.use_instruction:
            self.class_head = nn.Linear(self.hidden_dim, self.args.class_num)

    def forward(self, camera_color, camera_depth, camera_point_cloud,
                qpos, next_actions, next_action_is_pad,
                instruction_input_ids, instruction_attention_mask, instruction_vector,
                actions=None, action_is_pad=None):

        is_training = actions is not None
        bs = 0
        if camera_color is not None:
            bs = camera_color.shape[0]
        if camera_depth is not None:
            bs = camera_depth.shape[0]
        if camera_point_cloud is not None:
            bs = camera_point_cloud.shape[0]
        if qpos is not None:
            bs = qpos.shape[0]

        instruction_input = None
        if self.instruction_encoder is not None:
            instruction_input_ids = None if instruction_input_ids is None else instruction_input_ids[:, :self.args.instruction_max_len]
            instruction_attention_mask = None if instruction_attention_mask is None else instruction_attention_mask[:, :self.args.instruction_max_len]
            instruction_vector = self.instruction_encoder(instruction_input_ids, instruction_attention_mask)["last_hidden_state"] if type(self.instruction_encoder) is not int else instruction_vector
            instruction_vector = instruction_vector[:, :self.args.instruction_max_len, :]
            instruction_input = torch.transpose(self.input_proj_instruction(instruction_vector), 0, 1)

        if is_training and self.args.kl_weight != 0:
            cls_embed = torch.unsqueeze(self.cls_embed.weight, axis=0).repeat(bs, 1, 1)
            qpos_embed = None
            if self.input_state_dim != 0:
                qpos_embed = self.encoder_qpos_proj(qpos)
            next_action_embed = None
            if self.args.next_action_num != 0 and next_actions is not None and self.input_state_dim != 0:
                next_action_embed = self.encoder_next_action_proj(next_actions)
            action_embed = self.encoder_action_proj(actions)
            encoder_input = cls_embed
            if self.input_state_dim != 0:
                encoder_input = torch.cat([encoder_input, qpos_embed], axis=1)
                if next_actions is not None:
                    encoder_input = torch.cat([encoder_input, next_action_embed], axis=1)
            encoder_input = torch.cat([encoder_input, action_embed], axis=1)
            is_pad = torch.full((bs, 1+(self.args.obs_history_num if self.input_state_dim != 0 else 0)), False).to(actions.device)
            if next_action_is_pad is not None and self.input_state_dim != 0:
                is_pad = torch.cat([is_pad, next_action_is_pad, action_is_pad], axis=1)
            else:
                is_pad = torch.cat([is_pad, action_is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            encoder_input = encoder_input.permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        all_cam_features = []
        all_cam_pos = []
        src = None
        src_pos = None
        if self.args.use_camera_color:
            for cam_id, cam_name in enumerate(self.args.camera_color_names):
                for i in range(self.args.obs_history_num):
                    if self.args.backbone.startswith("resnet"):
                        features, src_pos = (self.backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                                             (camera_color[:, cam_id * self.args.obs_history_num + i],
                                              qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                              instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                        features = features[0]
                        src_pos = src_pos[0]
                        if self.depth_backbones is not None and camera_depth is not None and cam_id < len(self.args.camera_depth_names):
                            features_depth = (self.depth_backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                                              (camera_depth[:, cam_id * self.args.obs_history_num + i].unsqueeze(dim=1),
                                               qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                               instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                            if self.args.use_multi_camera_backbone:
                                all_cam_features.append(self.input_proj[cam_id](torch.cat([features, features_depth], axis=1)))
                            else:
                                all_cam_features.append(self.input_proj(torch.cat([features, features_depth], axis=1)))
                        else:
                            if self.args.use_multi_camera_backbone:
                                all_cam_features.append(self.input_proj[cam_id](features))
                            else:
                                all_cam_features.append(self.input_proj(features))
                        all_cam_pos.append(src_pos)
                    else:
                        condition = {"image": [], "qpos": qpos if self.args.use_qpos_film and self.input_state_dim != 0 else None, "instruction": instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None}
                        input_color = torch.unsqueeze(camera_color[:, cam_id * self.args.obs_history_num + i], axis=1)
                        condition["image"].append(input_color)
                        if self.args.use_camera_depth:
                            input_depth = torch.unsqueeze(camera_depth[:, cam_id * self.args.obs_history_num + i], axis=1)
                            input_depth = torch.unsqueeze(input_depth, axis=1)
                            condition["image"].append(input_depth)
                        features = self.backbones[cam_id if self.args.use_multi_camera_backbone else 0](condition)
                        features = torch.unsqueeze(features, axis=1)
                        all_cam_features.append(features)
            if self.args.backbone.startswith("resnet"):
                src = torch.cat(all_cam_features, axis=3)
                src_pos = torch.cat(all_cam_pos, axis=3)
                src = src.flatten(2).permute(0, 2, 1)
                src_pos = src_pos.flatten(2).permute(0, 2, 1).squeeze(axis=0)
            else:
                src = torch.cat(all_cam_features, axis=1)
                src_pos = self.pos.weight
        # proprioception features
        qpos_input = self.input_proj_qpos(qpos).permute(1, 0, 2) if self.input_state_dim != 0 else None
        # qpos_input = torch.unsqueeze(qpos_input, axis=0)
        # qpos_input = torch.randn(qpos_input.shape).to(qpos_input.device)

        next_action_input = None
        if self.args.next_action_num != 0 and next_actions is not None and self.input_state_dim != 0:
            next_action_input = self.input_proj_next_action(next_actions).permute(1, 0, 2)

        latent_input = torch.unsqueeze(latent_input, axis=0)

        if self.point_cloud_backbones is not None and camera_point_cloud is not None:
            point_cloud_features = []
            for cam_id, cam_name in enumerate(self.args.camera_point_cloud_names):
                for i in range(self.args.obs_history_num):
                    point_cloud_features.append(torch.unsqueeze(
                        self.point_cloud_backbones[cam_id if self.args.use_multi_camera_backbone else 0]
                        (camera_point_cloud[:, cam_id * self.args.obs_history_num + i],
                         qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                         instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None), axis=1))
            point_cloud_intput = torch.cat(point_cloud_features, axis=1)
        else:
            point_cloud_intput = None

        result = self.transformer(bs, self.query_embed.weight,
                              src, src_pos,
                              None, None,
                              # depth_features, depth_pos,
                              point_cloud_intput, self.point_cloud_pos.weight if point_cloud_intput is not None else None,
                              qpos_input, self.qpos_pos.weight if self.qpos_pos is not None else None,
                              next_action_input, self.next_action_pos.weight if next_action_input is not None else None, next_action_is_pad,
                              instruction_input, self.instruction_pos.weight if instruction_input is not None else None, instruction_attention_mask if instruction_input is not None else None,
                              latent_input, self.latent_pos.weight)
        result['result'] = self.action_head(result['action'])
        result['mu'] = mu
        result['logvar'] = logvar
        if self.args.class_num != 0 and self.args.use_instruction:
            result['class_prob'] = self.class_head(result['instruction'])
            _, result['class'] = torch.max(result['class_prob'], 1)
        return result


class Diffusion(nn.Module):
    def __init__(self, args, input_state_dim, output_state_dim,
                 backbones, depth_backbones, point_cloud_backbones,
                 instruction_encoder, pools, linears):
        super().__init__()
        self.args = args
        self.input_state_dim = input_state_dim
        self.output_state_dim = output_state_dim
        self.backbones = nn.ModuleList(backbones)
        self.backbones = replace_bn_with_gn(self.backbones)  # TODO

        if point_cloud_backbones is not None:
            self.point_cloud_backbones = nn.ModuleList(point_cloud_backbones)
        else:
            self.point_cloud_backbones = None
        self.instruction_encoder = instruction_encoder

        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)
        self.depth_backbones = depth_backbones
        if depth_backbones is not None:
            self.depth_backbones = nn.ModuleList(depth_backbones)
        self.weight_decay = 0
        self.num_kp = 32
        if args.backbone.startswith("resnet"):
            self.feature_dimension = 64
        else:
            self.feature_dimension = 512
        self.state_mlp = None
        if input_state_dim != 0:
            self.state_mlp = nn.Sequential(*create_mlp(self.input_state_dim, 64, [64], nn.ReLU))
            self.input_state_dim = 64
        self.instruction_input_dim = self.args.instruction_hidden_dim
        self.instruction_output_dim = 32
        if self.instruction_encoder is not None:
            self.instruction_mlp = nn.Linear(self.instruction_input_dim*self.args.instruction_max_len, self.instruction_output_dim)
        else:
            self.instruction_mlp = None
        if self.args.use_camera_color:
            self.obs_dim = ((self.feature_dimension * len(self.args.camera_color_names) * self.args.obs_history_num +
                             ((self.args.hidden_dim * len(self.args.camera_point_cloud_names) * self.args.obs_history_num) if point_cloud_backbones is not None else 0)) +
                            self.input_state_dim * self.args.obs_history_num + (self.instruction_output_dim if instruction_encoder is not None else 0)
                            + self.input_state_dim * self.args.next_action_num)  # camera features and proprio
        else:
            self.obs_dim = ((0 +
                             ((self.args.hidden_dim * len(self.args.camera_point_cloud_names) * self.args.obs_history_num) if point_cloud_backbones is not None else 0)) +
                            self.input_state_dim * self.args.obs_history_num + (self.instruction_output_dim if instruction_encoder is not None else 0)
                            + self.input_state_dim * self.args.next_action_num)  # camera features and proprio
        if self.args.class_num != 0 and self.args.use_instruction:
            self.class_head = nn.Linear(self.obs_dim, self.args.class_num)
        else:
            self.class_head = None
        if self.args.use_transformer:
            self.noise_pred_net = TransformerForDiffusion(
                self.output_state_dim,
                self.output_state_dim,
                self.args.chunk_size,
                n_obs_steps=1,
                cond_dim=self.obs_dim,
                causal_attn=True,
                time_as_cond=True
            )
        else:
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.output_state_dim,
                global_cond_dim=self.obs_dim
            )
            # self.noise_pred_net = ConditionalUnet1D(
            #     input_dim=self.output_state_dim,
            #     local_cond_dim=None,
            #     global_cond_dim=self.obs_dim,
            #     diffusion_step_embed_dim=128,
            #     down_dims=[512, 1024, 2048],
            #     kernel_size=5,
            #     n_groups=8,
            #     condition_type="film",
            #     use_down_condition=True,
            #     use_mid_condition=True,
            #     use_up_condition=True,
            # )

        self.nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'backbones': self.backbones,
                    'depth_backbones': self.depth_backbones,
                    'point_cloud_backbones': self.point_cloud_backbones,
                    'instruction_encoder': self.instruction_encoder if type(self.instruction_encoder) is not int else None,
                    'instruction_mlp': self.instruction_mlp,
                    'class_head': self.class_head,
                    'state_mlp': self.state_mlp,
                    'pools': self.pools,
                    'linears': self.linears,
                    'noise_pred_net': self.noise_pred_net
                })
            })
        self.nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            self.ema = EMAModel(model=self.nets, power=self.args.ema_power)
        else:
            self.ema = None

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=self.args.prediction_type
        )
        # self.noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=100,
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     # set_alpha_to_one=True,
        #     # steps_offset=0,
        #     prediction_type='sample'
        # )

    def forward(self, camera_color, camera_depth, camera_point_cloud,
                qpos, next_actions, next_action_is_pad,
                instruction_input_ids, instruction_attention_mask, instruction_vector,
                actions=None, action_is_pad=None):
        B = 0
        device = None
        if camera_color is not None:
            B = camera_color.shape[0]
            device = camera_color.device
        if camera_depth is not None:
            B = camera_depth.shape[0]
            device = camera_depth.device
        if camera_point_cloud is not None:
            B = camera_point_cloud.shape[0]
            device = camera_point_cloud.device
        if qpos is not None:
            B = qpos.shape[0]
            device = qpos.device
        if self.instruction_encoder is not None:
            instruction_vector = self.instruction_encoder(instruction_input_ids, instruction_attention_mask)["last_hidden_state"] if type(self.instruction_encoder) is not int else instruction_vector
            instruction_vector = instruction_vector[:, :self.args.instruction_max_len, :]
            instruction_input = self.instruction_mlp(instruction_vector.reshape((instruction_vector.shape[0], -1)))

        nets = self.nets
        if actions is None and self.ema is not None:
            nets = self.ema.averaged_model
        all_features = None
        if self.args.use_camera_color:
            all_features = []
            for cam_id in range(len(self.args.camera_color_names)):
                for i in range(self.args.obs_history_num):
                    if self.args.backbone.startswith("resnet"):
                        features, src_pos = (nets['policy']['backbones'][cam_id if self.args.use_multi_camera_backbone else 0]
                                             (camera_color[:, cam_id * self.args.obs_history_num + i],
                                              qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                              instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                        features = features[0]  # take the last layer feature
                        if camera_depth is not None and cam_id < len(self.args.camera_depth_names):
                            features_depth = (self.nets['policy']['depth_backbones'][cam_id if self.args.use_multi_camera_backbone else 0]
                                              (camera_depth[:, cam_id * self.args.obs_history_num + i].unsqueeze(dim=1),
                                               qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                               instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
                            features = torch.cat([features, features_depth], axis=1)
                        pool_features = nets['policy']['pools'][cam_id if self.args.use_multi_camera_backbone else 0](features)
                        pool_features = torch.flatten(pool_features, start_dim=1)
                        out_features = nets['policy']['linears'][cam_id if self.args.use_multi_camera_backbone else 0](pool_features)
                        all_features.append(out_features)
                    else:
                        condition = {"image": [], "qpos": qpos if self.args.use_qpos_film and self.input_state_dim != 0 else None, "instruction": instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None}
                        input_color = torch.unsqueeze(camera_color[:, cam_id * self.args.obs_history_num + i], axis=1)
                        condition["image"].append(input_color)
                        if self.args.use_camera_depth:
                            input_depth = torch.unsqueeze(camera_depth[:, cam_id * self.args.obs_history_num + i], axis=1)
                            input_depth = torch.unsqueeze(input_depth, axis=1)
                            condition["image"].append(input_depth)
                        features = self.nets['policy']['backbones'][cam_id if self.args.use_multi_camera_backbone else 0](condition)
                        all_features.append(features)
            all_features = torch.cat(all_features, dim=1)
        obs_cond = torch.tensor([]).to(device)
        if all_features is not None:
            # obs_cond = torch.cat([all_features, qpos.reshape(B, -1)], dim=1)
            obs_cond = all_features
        if self.input_state_dim != 0:
            obs_cond = torch.cat([obs_cond, self.state_mlp(qpos.reshape(B, -1))], dim=1)
            if self.args.next_action_num != 0 and next_actions is not None:
                next_actions = next_actions.reshape(B, -1)
                obs_cond = torch.cat([obs_cond, next_actions], dim=1)

        if self.point_cloud_backbones is not None and camera_point_cloud is not None:
            point_cloud_features = []
            for cam_id in range(len(self.args.camera_point_cloud_names)):
                for i in range(self.args.obs_history_num):
                    point_cloud_features.append(self.nets['policy']['point_cloud_backbones'][cam_id if self.args.use_multi_camera_backbone else 0]
                                                (camera_point_cloud[:, cam_id * self.args.obs_history_num + i],
                                                 qpos[:, i] if self.args.use_qpos_film and self.input_state_dim != 0 else None,
                                                 instruction_vector.reshape((instruction_vector.shape[0], -1)) if (self.args.use_instruction and self.args.use_instruction_film) else None))
            point_cloud_intput = torch.cat(point_cloud_features, axis=1).reshape(B, -1)
            obs_cond = obs_cond.to(point_cloud_intput.device)
            obs_cond = torch.cat([obs_cond, point_cloud_intput], dim=1)
        if self.instruction_encoder is not None:
            obs_cond = torch.cat([obs_cond, instruction_input], dim=1)
        result = {}
        if actions is not None:
            noise = torch.randn(actions.shape, device=obs_cond.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            if self.args.use_transformer:
                noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond.unsqueeze(dim=1))
            else:
                noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            if self.ema is not None:
                self.ema.step(nets)
            result['noise_pred'] = noise_pred
            if self.args.prediction_type == 'sample':
                result['noise'] = actions
            else:
                result['noise'] = noise
        else:
            noisy_action = torch.randn((B, self.args.chunk_size, self.output_state_dim), device=obs_cond.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.args.num_inference_timesteps)
            for k in self.noise_scheduler.timesteps:
                if self.args.use_transformer:
                    noise_pred = nets['policy']['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond.unsqueeze(dim=1)
                    )
                else:
                    noise_pred = nets['policy']['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            result['result'] = naction
        if self.args.class_num != 0 and self.args.use_instruction:
            result['class_prob'] = self.class_head(obs_cond)
            _, result['class'] = torch.max(result['class_prob'], 1)
        return result

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status


def build_diffusion(args):
    input_state_dim = 0
    output_state_dim = 0
    if args.use_arm_joint_state % 2 == 1:
        input_state_dim += args.arm_joint_state_dim * len(args.arm_joint_state_names)
    if args.use_arm_end_pose % 2 == 1:
        input_state_dim += args.arm_end_pose_dim * len(args.arm_end_pose_names)
    if args.use_robot_base % 2 == 1:
        input_state_dim += args.robot_base_dim
    if args.use_arm_joint_state > 1:
        output_state_dim += args.arm_joint_state_dim * len(args.arm_joint_state_names)
    if args.use_arm_end_pose > 1:
        output_state_dim += args.arm_end_pose_dim * len(args.arm_end_pose_names)
    if args.use_robot_base > 1:
        output_state_dim += args.robot_base_dim

    color_backbones = []
    pools = []
    linears = []
    depth_backbones = []
    point_cloud_backbones = []

    if args.use_multi_camera_backbone:
        for i in range(len(args.camera_color_names)):
            if args.backbone.startswith("resnet"):
                if args.use_camera_color:
                    backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation,
                                        qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                        instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None)
                    model = Joiner(backbone, build_position_encoding(args))
                    model.num_channels = backbone.num_channels
                    color_backbones.append(model)
                    num_channels = 512
                    if args.use_camera_depth and i < len(args.camera_depth_names):
                        depth_backbones.append(DepthNet(qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                        instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                        num_channels += depth_backbones[-1].num_channels
                    pools.append(SpatialSoftmax(**{'input_shape': [num_channels, 15, 20], 'num_kp': 32, 'temperature': 1.0,
                                                   'learnable_temperature': False, 'noise_std': 0.0}))
                    linears.append(torch.nn.Linear(int(np.prod([32, 2])), 64))
            else:
                if args.use_camera_color:
                    if args.use_camera_depth:
                        color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480), (640, 480)), in_channels=(3, 1), d_model=args.hidden_dim, nhead=4,
                                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                    else:
                        color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480),), in_channels=(3,), d_model=args.hidden_dim, nhead=4,
                                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        for _ in args.camera_point_cloud_names:
            if args.use_camera_point_cloud:
                if args.use_camera_point_cloud_rgb:
                    point_cloud_backbones.append(PointNetEncoderXYZRGB(in_channels=6, out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                       qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                       instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                else:
                    point_cloud_backbones.append(PointNetEncoderXYZ(out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                    qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                # point_cloud_backbones.append(Pointnet2Backbone(args.hidden_dim, qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                #                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
    else:
        if args.backbone.startswith("resnet"):
            if args.use_camera_color:
                backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation,
                                    qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None)
                model = Joiner(backbone, build_position_encoding(args))
                model.num_channels = backbone.num_channels
                color_backbones.append(model)
                num_channels = 512
                if args.use_camera_depth:
                    depth_backbones.append(DepthNet(qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                    num_channels += depth_backbones[-1].num_channels
                pools.append(SpatialSoftmax(**{'input_shape': [num_channels, 15, 20], 'num_kp': 32, 'temperature': 1.0,
                                               'learnable_temperature': False, 'noise_std': 0.0}))
                linears.append(torch.nn.Linear(int(np.prod([32, 2])), 64))
        else:
            if args.use_camera_color:
                if args.use_camera_depth:
                    color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480), (640, 480)), in_channels=(3, 1), d_model=args.hidden_dim, nhead=4,
                                                                               qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                               instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                else:
                    color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480),), in_channels=(3,), d_model=args.hidden_dim, nhead=4,
                                                                               qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                               instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        if args.use_camera_point_cloud:
            if args.use_camera_point_cloud_rgb:
                point_cloud_backbones.append(PointNetEncoderXYZRGB(in_channels=6, out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
            else:
                point_cloud_backbones.append(PointNetEncoderXYZ(out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
            # point_cloud_backbones.append(Pointnet2Backbone(args.hidden_dim, qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
            #                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))

    instruction_encoder = None
    if args.use_instruction:
        instruction_encoder = 0
    #     instruction_encoder = BertModel.from_pretrained(args.instruction_encoder_dir)

    model = Diffusion(
        args,
        input_state_dim,
        output_state_dim,
        color_backbones if len(color_backbones) > 0 else None,
        depth_backbones if len(depth_backbones) > 0 else None,
        point_cloud_backbones if len(point_cloud_backbones) > 0 else None,
        instruction_encoder,
        pools if len(pools) > 0 else None,
        linears if len(linears) > 0 else None
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    return model


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_act_encoder(args):
    
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_act(args):
    input_state_dim = 0
    output_state_dim = 0
    if args.use_arm_joint_state % 2 == 1:
        input_state_dim += args.arm_joint_state_dim * len(args.arm_joint_state_names)
    if args.use_arm_end_pose % 2 == 1:
        input_state_dim += args.arm_end_pose_dim * len(args.arm_end_pose_names)
    if args.use_robot_base % 2 == 1:
        input_state_dim += args.robot_base_dim
    if args.use_arm_joint_state > 1:
        output_state_dim += args.arm_joint_state_dim * len(args.arm_joint_state_names)
    if args.use_arm_end_pose > 1:
        output_state_dim += args.arm_end_pose_dim * len(args.arm_end_pose_names)
    if args.use_robot_base > 1:
        output_state_dim += args.robot_base_dim

    color_backbones = []
    depth_backbones = []
    point_cloud_backbones = []

    if args.use_multi_camera_backbone:
        for i in range(len(args.camera_color_names)):
            if args.backbone.startswith("resnet"):
                if args.use_camera_color:
                    backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation,
                                        qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                        instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None)
                    model = Joiner(backbone, build_position_encoding(args))
                    model.num_channels = backbone.num_channels
                    color_backbones.append(model)
                    if args.use_camera_depth and i < len(args.camera_depth_names):
                        depth_backbones.append(DepthNet(qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                        instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
            else:
                if args.use_camera_color:
                    if args.use_camera_depth:
                        color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480), (640, 480)), in_channels=(3, 1), d_model=args.hidden_dim, nhead=4,
                                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                    else:
                        color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480),), in_channels=(3,), d_model=args.hidden_dim, nhead=4,
                                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        for _ in args.camera_point_cloud_names:
            if args.use_camera_point_cloud:
                if args.use_camera_point_cloud_rgb:
                    point_cloud_backbones.append(PointNetEncoderXYZRGB(in_channels=6, out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                       qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                       instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                else:
                    point_cloud_backbones.append(PointNetEncoderXYZ(out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                    qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                    # point_cloud_backbones.append(Pointnet2Backbone(args.hidden_dim, qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                    #                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
    else:
        if args.backbone.startswith("resnet"):
            if args.use_camera_color:
                backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation,
                                    qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None)
                model = Joiner(backbone, build_position_encoding(args))
                model.num_channels = backbone.num_channels
                color_backbones.append(model)
                if args.use_camera_depth:
                    depth_backbones.append(DepthNet(qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        else:
            if args.use_camera_color:
                if args.use_camera_depth:
                    color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480), (640, 480)), in_channels=(3, 1), d_model=args.hidden_dim, nhead=4,
                                                                               qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                               instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                else:
                    color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480),), in_channels=(3,), d_model=args.hidden_dim, nhead=4,
                                                                               qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                               instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        if args.use_camera_point_cloud:
            if args.use_camera_point_cloud_rgb:
                point_cloud_backbones.append(
                    PointNetEncoderXYZRGB(in_channels=6, out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                          qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                          instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
            else:
                point_cloud_backbones.append(PointNetEncoderXYZ(out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                # point_cloud_backbones.append(Pointnet2Backbone(args.hidden_dim, use_qpos_film=args.use_qpos_film, qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                #                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))

    instruction_encoder = None
    if args.use_instruction:
        instruction_encoder = 0
        # instruction_encoder = BertModel.from_pretrained(args.instruction_encoder_dir)

    transformer = build_transformer(args)

    encoder = None
    if args.kl_weight != 0:
        encoder = build_act_encoder(args)

    model = ACT(
        args,
        input_state_dim,
        output_state_dim,
        color_backbones if len(color_backbones) > 0 else None,
        depth_backbones if len(depth_backbones) > 0 else None,
        point_cloud_backbones if len(point_cloud_backbones) > 0 else None,
        instruction_encoder,
        transformer,
        encoder
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_rt(args):
    input_state_dim = 0
    output_state_dim = 0
    if args.use_arm_joint_state % 2 == 1:
        input_state_dim += args.arm_joint_state_dim * len(args.arm_joint_state_names)
    if args.use_arm_end_pose % 2 == 1:
        input_state_dim += args.arm_end_pose_dim * len(args.arm_end_pose_names)
    if args.use_robot_base % 2 == 1:
        input_state_dim += args.robot_base_dim
    if args.use_arm_joint_state > 1:
        output_state_dim += args.arm_joint_state_dim * len(args.arm_joint_state_names)
    if args.use_arm_end_pose > 1:
        output_state_dim += args.arm_end_pose_dim * len(args.arm_end_pose_names)
    if args.use_robot_base > 1:
        output_state_dim += args.robot_base_dim

    color_backbones = []
    depth_backbones = []
    point_cloud_backbones = []

    if args.use_multi_camera_backbone:
        for i in range(len(args.camera_color_names)):
            if args.backbone.startswith("resnet"):
                if args.use_camera_color:
                    backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation,
                                        qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                        instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None)
                    model = Joiner(backbone, build_position_encoding(args))
                    model.num_channels = backbone.num_channels
                    color_backbones.append(model)
                    if args.use_camera_depth and i < len(args.camera_depth_names):
                        depth_backbones.append(DepthNet(qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                        instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
            else:
                if args.use_camera_color:
                    if args.use_camera_depth:
                        color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480), (640, 480)), in_channels=(3, 1), d_model=args.hidden_dim, nhead=4,
                                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                    else:
                        color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480),), in_channels=(3,), d_model=args.hidden_dim, nhead=4,
                                                                                   qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                                   instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        for _ in args.camera_point_cloud_names:
            if args.use_camera_point_cloud:
                if args.use_camera_point_cloud_rgb:
                    point_cloud_backbones.append(PointNetEncoderXYZRGB(in_channels=6, out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                       qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                       instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                else:
                    point_cloud_backbones.append(PointNetEncoderXYZ(out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                    qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                    # point_cloud_backbones.append(Pointnet2Backbone(args.hidden_dim,qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                    #                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
    else:
        if args.backbone.startswith("resnet"):
            if args.use_camera_color:
                backbone = Backbone(args.backbone, args.lr_backbone > 0, args.masks, args.dilation,
                                    qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None)
                model = Joiner(backbone, build_position_encoding(args))
                model.num_channels = backbone.num_channels
                color_backbones.append(model)
                if args.use_camera_depth:
                    depth_backbones.append(DepthNet(qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                    instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        else:
            if args.use_camera_color:
                if args.use_camera_depth:
                    color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480), (640, 480)), in_channels=(3, 1), d_model=args.hidden_dim, nhead=4,
                                                                               qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                               instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                else:
                    color_backbones.append(EarlyConvViTMultiViewImageCondition(image_sz=((640, 480),), in_channels=(3,), d_model=args.hidden_dim, nhead=4,
                                                                               qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                               instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
        if args.use_camera_point_cloud:
            if args.use_camera_point_cloud_rgb:
                point_cloud_backbones.append(
                    PointNetEncoderXYZRGB(in_channels=6, out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                          qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                          instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
            else:
                point_cloud_backbones.append(PointNetEncoderXYZ(out_channels=args.hidden_dim, use_layernorm=True, final_norm="layernorm",
                                                                qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))
                # point_cloud_backbones.append(Pointnet2Backbone(args.hidden_dim, qpos_vector_dim=input_state_dim if args.use_qpos_film and input_state_dim != 0 else None,
                #                                                instruction_vector_dim=args.instruction_hidden_dim*args.instruction_max_len if (args.use_instruction_film and args.use_instruction) else None))

    instruction_encoder = None
    if args.use_instruction:
        instruction_encoder = 0
        # instruction_encoder = BertModel.from_pretrained(args.instruction_encoder_dir)

    transformer = build_original_transformer_encoder(args, output_state_dim)

    model = RT(
        args,
        input_state_dim,
        output_state_dim,
        color_backbones if len(color_backbones) > 0 else None,
        depth_backbones if len(depth_backbones) > 0 else None,
        point_cloud_backbones if len(point_cloud_backbones) > 0 else None,
        instruction_encoder,
        transformer,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
