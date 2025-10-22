# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
from .transformer_bert import Transformer_BERT
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..rdt.model import RDT
import IPython
e = IPython.embed


class RDTransformer(nn.Module):

    def __init__(self, args, output_state_dim):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.nhead = args.nheads
        self.num_encoder_layers = args.enc_layers
        self.dim_feedforward = 2048
        self.num_layers = args.transformer_layer_num
        self.dropout = args.dropout
        self.use_diffusion = args.use_diffusion
        self.encoder = RDT(hidden_size=args.hidden_dim, horizon=args.chunk_size,
                           img_cond_len=15 * 20 * len(args.camera_color_names),
                           max_lang_cond_len=args.instruction_max_len,
                           output_dim=output_state_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, bs, query_embed,
                timestamp_input, timestamp_pos,
                color_input, color_pos,
                depth_input, depth_pos,
                point_cloud_intput, point_cloud_pos,
                robot_state_input, robot_state_pos=None,
                next_action_input=None, next_action_pos=None, next_action_is_pad=None,
                instruction_input=None, instruction_pos=None, instruction_pos_is_pad=None):
        device = None
        if color_input is not None:
            device = color_input.device
        if depth_input is not None:
            device = depth_input.device
        if point_cloud_intput is not None:
            device = point_cloud_intput.device
        if robot_state_input is not None:
            device = robot_state_input.device
        src = torch.tensor([]).to(device)
        if color_input is not None:
            src = torch.cat([src, color_input], axis=1)
        if depth_input is not None:
            src = torch.cat([src, depth_input], axis=1)
        if point_cloud_intput is not None:
            src = torch.cat([src, point_cloud_intput], axis=1)
        hs = self.encoder(torch.cat([robot_state_input, query_embed], axis=0).permute(1, 0, 2),
                          torch.full(timestamp_input.shape, 30).to(device),
                          timestamp_input,
                          instruction_input.permute(1, 0, 2), src, lang_mask=instruction_pos_is_pad)
        result = {}
        result['action'] = hs[:, -query_embed.shape[0]:, :]
        return result


class OriginalTransformerEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.nhead = args.nheads
        self.num_encoder_layers = args.enc_layers
        self.dim_feedforward = 2048
        self.num_layers = args.transformer_layer_num
        self.dropout = args.dropout
        self.use_diffusion = args.use_diffusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu'
        )
        self.encoder = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers) for _ in range(self.num_layers)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, bs, query_embed,
                timestamp_input, timestamp_pos,
                color_input, color_pos,
                depth_input, depth_pos,
                point_cloud_intput, point_cloud_pos,
                robot_state_input, robot_state_pos=None,
                next_action_input=None, next_action_pos=None, next_action_is_pad=None,
                instruction_input=None, instruction_pos=None, instruction_pos_is_pad=None):
        device = None
        if color_input is not None:
            device = color_input.device
        if depth_input is not None:
            device = depth_input.device
        if point_cloud_intput is not None:
            device = point_cloud_intput.device
        if robot_state_input is not None:
            device = robot_state_input.device

        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        if timestamp_input is not None:
            timestamp_is_pad = torch.full((timestamp_input.shape[1], timestamp_input.shape[0]), False).to(device)
            timestamp_pos = timestamp_pos.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
            timestamp_start = 0
            src = timestamp_input
            pos = timestamp_pos
            is_pad = timestamp_is_pad
            timestamp_end = src.shape[0]
        else:
            src = torch.tensor([]).to(device)
            pos = torch.tensor([]).to(device)
            is_pad = torch.tensor([]).to(device)

        if robot_state_input is not None:
            robot_state_is_pad = torch.full((robot_state_input.shape[1], robot_state_input.shape[0]), False).to(device)
            robot_state_pos = robot_state_pos.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
            robot_state_start = src.shape[0]
            src = torch.cat([src, robot_state_input], axis=0)
            pos = torch.cat([pos, robot_state_pos], axis=0)
            is_pad = torch.cat([is_pad, robot_state_is_pad], axis=1)
            robot_state_end = src.shape[0]

        if next_action_input is not None:
            next_action_start = src.shape[0]
            next_action_pos = next_action_pos.unsqueeze(1).repeat(1, bs, 1)
            src = torch.cat([src, next_action_input], axis=0)
            pos = torch.cat([pos, next_action_pos], axis=0)
            is_pad = torch.cat([is_pad, next_action_is_pad], axis=1)
            next_action_end = src.shape[0]

        if instruction_input is not None:
            instruction_start = src.shape[0]
            instruction_pos = instruction_pos.unsqueeze(1).repeat(1, bs, 1)
            # instruction_pos_is_pad = torch.full((instruction_input.shape[1], instruction_input.shape[0]), False).to(device)
            pos = torch.cat([pos, instruction_pos], axis=0)
            src = torch.cat([src, instruction_input], axis=0)
            is_pad = torch.cat([is_pad, instruction_pos_is_pad], axis=1)
            instruction_end = src.shape[0]

        if color_input is not None:
            color_start = src.shape[0]
            color_input = color_input.flatten(2).permute(1, 0, 2)
            color_pos = color_pos.unsqueeze(1).repeat(1, bs, 1)
            color_is_pad = torch.full((color_input.shape[1], color_input.shape[0]), False).to(device)
            src = torch.cat([src, color_input], axis=0)
            pos = torch.cat([pos, color_pos], axis=0)
            is_pad = torch.cat([is_pad, color_is_pad], axis=1)
            color_end = src.shape[0]

        if depth_input is not None:
            depth_start = src.shape[0]
            depth_input = depth_input.flatten(2).permute(1, 0, 2)
            depth_pos = depth_pos.unsqueeze(1).repeat(1, bs, 1)
            depth_is_pad = torch.full((depth_input.shape[1], depth_input.shape[0]), False).to(device)
            pos = torch.cat([pos, depth_pos], axis=0)
            src = torch.cat([src, depth_input], axis=0)
            is_pad = torch.cat([is_pad, depth_is_pad], axis=1)
            depth_end = src.shape[0]

        if point_cloud_intput is not None:
            point_cloud_start = src.shape[0]
            point_cloud_intput = point_cloud_intput.permute(1, 0, 2)
            point_cloud_pos = point_cloud_pos.unsqueeze(1).repeat(1, bs, 1)
            point_cloud_is_pad = torch.full((point_cloud_intput.shape[1], point_cloud_intput.shape[0]), False).to(device)
            src = torch.cat([src, point_cloud_intput], axis=0)
            pos = torch.cat([pos, point_cloud_pos], axis=0)
            is_pad = torch.cat([is_pad, point_cloud_is_pad], axis=1)
            point_cloud_end = src.shape[0]

        src = torch.cat([src, torch.zeros_like(query_embed)], axis=0)
        pos = torch.cat([pos, query_embed], axis=0)
        is_pad = torch.cat([is_pad, torch.full((query_embed.shape[1], query_embed.shape[0]), False).to(device)], axis=1)
        hs = torch.zeros_like(src)
        for i in range(self.num_layers):
            hs = self.encoder[i](hs + src + pos, src_key_padding_mask=is_pad)
        hs = hs.transpose(1, 0)

        result = {}
        if timestamp_input is not None:
            result['timestamp'] = hs[:, timestamp_start:timestamp_end, :]
        if robot_state_input is not None:
            result['robot_state'] = hs[:, robot_state_start:robot_state_end, :]
        if next_action_input is not None:
            result['next_action'] = hs[:, next_action_start:next_action_end, :]
        if instruction_input is not None:
            result['instruction'] = hs[:, instruction_start:instruction_end, :]
        if color_input is not None:
            result['color'] = hs[:, color_start:color_end, :]
        # if depth_input is not None:
        #     result['depth'] = hs[:, depth_start:depth_end, :]
        if point_cloud_intput is not None:
            result['point_cloud'] = hs[:, point_cloud_start:point_cloud_end, :]
        result['action'] = hs[:, -query_embed.shape[0]:, :]
        return result


class Transformer_decoder(nn.Module):

    def __init__(self, context_len=None, d_model=512, nhead=8, num_decoder_layers=6, dropout=0.1,
                 use_pos_embd_image=False, query_num=50, use_pos_embd_action=False,
                 self_attention=True, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.decoder = Transformer_BERT(context_len=context_len,
                                            latent_dim=d_model,
                                            num_head=nhead,
                                            num_layer=num_decoder_layers,
                                            dropout_rate=dropout,
                                            use_pos_embd_image=use_pos_embd_image,
                                            use_pos_embd_action=use_pos_embd_action,
                                            query_num=query_num,
                                            self_attention=self_attention)
        else:
            self.decoder = nn.ModuleList([Transformer_BERT(context_len=context_len,
                                                           latent_dim=d_model,
                                                           num_head=nhead,
                                                           num_layer=num_decoder_layers,
                                                           dropout_rate=dropout,
                                                           use_pos_embd_image=use_pos_embd_image,
                                                           use_pos_embd_action=use_pos_embd_action,
                                                           query_num=query_num,
                                                           self_attention=self_attention) for _ in range(num_layers)])

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.self_attention = self_attention

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, bs, query_embed,
                color_input, color_pos,
                depth_input, depth_pos,
                point_cloud_intput, point_cloud_pos,
                robot_state_input, robot_state_pos=None,
                next_action_input=None, next_action_pos=None, next_action_is_pad=None,
                instruction_input=None, instruction_pos=None, instruction_pos_is_pad=None):
        device = None
        if color_input is not None:
            device = color_input.device
        if depth_input is not None:
            device = depth_input.device
        if point_cloud_intput is not None:
            device = point_cloud_intput.device
        if robot_state_input is not None:
            device = robot_state_input.device

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        if robot_state_input is not None:
            robot_state_is_pad = torch.full((robot_state_input.shape[1], robot_state_input.shape[0]), False).to(device)
            robot_state_pos = robot_state_pos.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
            robot_state_start = 0
            src = robot_state_input
            pos = robot_state_pos
            is_pad = robot_state_is_pad
            robot_state_end = src.shape[0]
        else:
            src = torch.tensor([]).to(device)
            pos = torch.tensor([]).to(device)
            is_pad = torch.tensor([]).to(device)

        if next_action_input is not None:
            next_action_start = src.shape[0]
            next_action_pos = next_action_pos.unsqueeze(1).repeat(1, bs, 1)
            src = torch.cat([src, next_action_input], axis=0)
            pos = torch.cat([pos, next_action_pos], axis=0)
            is_pad = torch.cat([is_pad, next_action_is_pad], axis=1)
            next_action_end = src.shape[0]

        if instruction_input is not None:
            instruction_start = src.shape[0]
            instruction_pos = instruction_pos.unsqueeze(1).repeat(1, bs, 1)
            # instruction_pos_is_pad = torch.full((instruction_input.shape[1], instruction_input.shape[0]), False).to(device)
            pos = torch.cat([pos, instruction_pos], axis=0)
            src = torch.cat([src, instruction_input], axis=0)
            is_pad = torch.cat([is_pad, instruction_pos_is_pad], axis=1)
            instruction_end = src.shape[0]

        if color_input is not None:
            color_start = src.shape[0]
            color_input = color_input.flatten(2).permute(1, 0, 2)
            color_pos = color_pos.unsqueeze(1).repeat(1, bs, 1)
            color_is_pad = torch.full((color_input.shape[1], color_input.shape[0]), False).to(device)
            src = torch.cat([src, color_input], axis=0)
            pos = torch.cat([pos, color_pos], axis=0)
            is_pad = torch.cat([is_pad, color_is_pad], axis=1)
            color_end = src.shape[0]

        if depth_input is not None:
            depth_start = src.shape[0]
            depth_input = depth_input.flatten(2).permute(1, 0, 2)
            depth_pos = depth_pos.unsqueeze(1).repeat(1, bs, 1)
            depth_is_pad = torch.full((depth_input.shape[1], depth_input.shape[0]), False).to(device)
            pos = torch.cat([pos, depth_pos], axis=0)
            src = torch.cat([src, depth_input], axis=0)
            is_pad = torch.cat([is_pad, depth_is_pad], axis=1)
            depth_end = src.shape[0]

        if point_cloud_intput is not None:
            point_cloud_start = src.shape[0]
            point_cloud_intput = point_cloud_intput.permute(1, 0, 2)
            point_cloud_pos = point_cloud_pos.unsqueeze(1).repeat(1, bs, 1)
            point_cloud_is_pad = torch.full((point_cloud_intput.shape[1], point_cloud_intput.shape[0]), False).to(device)
            src = torch.cat([src, point_cloud_intput], axis=0)
            pos = torch.cat([pos, point_cloud_pos], axis=0)
            is_pad = torch.cat([is_pad, point_cloud_is_pad], axis=1)
            point_cloud_end = src.shape[0]

        src = torch.cat([src, torch.zeros_like(query_embed)], axis=0)
        pos = torch.cat([pos, query_embed], axis=0)
        if self.num_layers == 1:
            hs = self.decoder(src, pos, query_embed)
        else:
            hs = src
            for i in range(self.num_layers):
                hs = self.decoder[i](hs, pos, query_embed)
        hs = hs.transpose(1, 0)

        result = {}
        if robot_state_input is not None:
            result['robot_state'] = hs[:, robot_state_start:robot_state_end, :]
        if next_action_input is not None:
            result['next_action'] = hs[:, next_action_start:next_action_end, :]
        if instruction_input is not None:
            result['instruction'] = hs[:, instruction_start:instruction_end, :]
        if color_input is not None:
            result['color'] = hs[:, color_start:color_end, :]
        # if depth_input is not None:
        #     result['depth'] = hs[:, depth_start:depth_end, :]
        if point_cloud_intput is not None:
            result['point_cloud'] = hs[:, point_cloud_start:point_cloud_end, :]
        result['action'] = hs[:, -query_embed.shape[0]:, :]
        return result


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # 编码层
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 归一化层
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        
        # 构建多层编码层
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 解码层
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        
        # 构建多层解码层
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, bs, query_embed,
                color_input, color_pos,
                depth_input, depth_pos,
                point_cloud_intput, point_cloud_pos,
                robot_state_input, robot_state_pos=None,
                next_action_input=None, next_action_pos=None, next_action_is_pad=None,
                instruction_input=None, instruction_pos=None, instruction_pos_is_pad=None,
                latent_input=None, latent_pos=None):
        # TODO flatten only when input has H and W
        # if len(src.shape) == 4: # has H and W
        device = None
        if color_input is not None:
            device = color_input.device
        if depth_input is not None:
            device = depth_input.device
        if point_cloud_intput is not None:
            device = point_cloud_intput.device
        if robot_state_input is not None:
            device = robot_state_input.device
        latent_is_pad = torch.full((latent_input.shape[1], latent_input.shape[0]), False).to(device)
        latent_pos = latent_pos.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        latent_start = 0
        src = latent_input
        pos = latent_pos
        is_pad = latent_is_pad
        latent_end = src.shape[0]

        if robot_state_input is not None:
            robot_state_is_pad = torch.full((robot_state_input.shape[1], robot_state_input.shape[0]), False).to(device)
            robot_state_pos = robot_state_pos.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
            robot_state_start = src.shape[0]
            pos = torch.cat([pos, robot_state_pos], axis=0)
            src = torch.cat([src, robot_state_input], axis=0)
            is_pad = torch.cat([is_pad, robot_state_is_pad], axis=1)
            robot_state_end = src.shape[0]

        if next_action_input is not None:
            next_action_start = src.shape[0]
            next_action_pos = next_action_pos.unsqueeze(1).repeat(1, bs, 1)
            src = torch.cat([src, next_action_input], axis=0)
            pos = torch.cat([pos, next_action_pos], axis=0)
            is_pad = torch.cat([is_pad, next_action_is_pad], axis=1)
            next_action_end = src.shape[0]

        if instruction_input is not None:
            instruction_start = src.shape[0]
            instruction_pos = instruction_pos.unsqueeze(1).repeat(1, bs, 1)
            # instruction_pos_is_pad = torch.full((instruction_input.shape[1], instruction_input.shape[0]), False).to(device)
            pos = torch.cat([pos, instruction_pos], axis=0)
            src = torch.cat([src, instruction_input], axis=0)
            is_pad = torch.cat([is_pad, instruction_pos_is_pad], axis=1)
            instruction_end = src.shape[0]

        if color_input is not None:
            color_start = src.shape[0]
            color_input = color_input.flatten(2).permute(1, 0, 2)
            color_pos = color_pos.unsqueeze(1).repeat(1, bs, 1)
            color_is_pad = torch.full((color_input.shape[1], color_input.shape[0]), False).to(device)
            src = torch.cat([src, color_input], axis=0)
            pos = torch.cat([pos, color_pos], axis=0)
            is_pad = torch.cat([is_pad, color_is_pad], axis=1)
            color_end = src.shape[0]

        if depth_input is not None:
            depth_start = src.shape[0]
            depth_input = depth_input.flatten(2).permute(1, 0, 2)
            depth_pos = depth_pos.unsqueeze(1).repeat(1, bs, 1)
            depth_is_pad = torch.full((depth_input.shape[1], depth_input.shape[0]), False).to(device)
            pos = torch.cat([pos, depth_pos], axis=0)
            src = torch.cat([src, depth_input], axis=0)
            is_pad = torch.cat([is_pad, depth_is_pad], axis=1)
            depth_end = src.shape[0]

        if point_cloud_intput is not None:
            point_cloud_start = src.shape[0]
            point_cloud_intput = point_cloud_intput.permute(1, 0, 2)
            point_cloud_pos = point_cloud_pos.unsqueeze(1).repeat(1, bs, 1)
            point_cloud_is_pad = torch.full((point_cloud_intput.shape[1], point_cloud_intput.shape[0]), False).to(device)
            src = torch.cat([src, point_cloud_intput], axis=0)
            pos = torch.cat([pos, point_cloud_pos], axis=0)
            is_pad = torch.cat([is_pad, point_cloud_is_pad], axis=1)
            point_cloud_end = src.shape[0]

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos, src_key_padding_mask=is_pad)
        
        hs = self.decoder(tgt, memory,
                          query_pos=query_embed,
                          pos=pos,
                          memory_key_padding_mask=is_pad)
        result = {}
        if robot_state_input is not None:
            result['robot_state'] = memory[:, robot_state_start:robot_state_end, :]
        if next_action_input is not None:
            result['next_action'] = memory[:, next_action_start:next_action_end, :]
        if instruction_input is not None:
            result['instruction'] = memory[:, instruction_start:instruction_end, :]
        if color_input is not None:
            result['color'] = memory[:, color_start:color_end, :]
        # if depth_input is not None:
        #     result['depth'] = memory[:, depth_start:depth_end, :]
        if point_cloud_intput is not None:
            result['point_cloud'] = memory[:, point_cloud_start:point_cloud_end, :]
        result['action'] = hs.transpose(1, 2)[-1]
        return result


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                pos: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                mask: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output,
                           pos=pos,
                           src_key_padding_mask=src_key_padding_mask,
                           src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                query_pos: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           query_pos=query_pos,
                           pos=pos,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_mask: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    src_mask: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                pos: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, pos, src_key_padding_mask, src_mask)
        return self.forward_post(src, pos, src_key_padding_mask, src_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     query_pos: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    query_pos: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                query_pos: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, query_pos, pos,
                                    tgt_key_padding_mask, memory_key_padding_mask, tgt_mask, memory_mask)
        return self.forward_post(tgt, memory, query_pos, pos,
                                 tgt_key_padding_mask, memory_key_padding_mask, tgt_mask, memory_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True)


def build_original_transformer_encoder(args, output_state_dim):
    feature_dimension = 1
    if args.backbone.startswith("resnet"):
        feature_dimension = 15 * 20
    if not args.use_camera_color:
        feature_dimension = 0
    # context_len = (
    #                 (feature_dimension * len(args.camera_color_names) * args.obs_history_num) +
    #                 ((len(args.camera_point_cloud_names) * args.obs_history_num) if args.use_camera_point_cloud else 0) +
    #                 args.obs_history_num +
    #                 (args.instruction_max_len if args.use_instruction else 0) +
    #                 args.next_action_num +
    #                 args.chunk_size
    #                )  # camera features and proprio
    if args.use_rdt:
        return RDTransformer(args, output_state_dim)
    else:
        return OriginalTransformerEncoder(args)
        # return Transformer_decoder(
        #     context_len=context_len,
        #     d_model=args.hidden_dim,
        #     dropout=args.dropout,
        #     nhead=args.nheads,
        #     num_layers=args.transformer_layer_num
        # )
    # return OriginalTransformerEncoder(
    #     d_model=args.hidden_dim,
    #     dropout=args.dropout,
    #     nhead=args.nheads,
    #     num_encoder_layers=args.enc_layers
    # )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
