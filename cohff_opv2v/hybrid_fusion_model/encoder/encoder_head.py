import torch
import torch.nn as nn
from torch.nn.init import normal_

from mmcv.runner import BaseModule
from mmseg.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding, \
    build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16

from .cross_view_self_attention import CoHFF_CrossView_Self_Attention
from .image_cross_attention import CoHFF_MSDeformableAttention3D


@HEADS.register_module()
class CoHFF_Encoder_Head(BaseModule):

    def __init__(self,
                 positional_encoding=None,
                 p_h=30,
                 p_w=30,
                 p_z=30,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 **kwargs):
        super().__init__()

        self.p_h = p_h
        self.p_w = p_w
        self.p_z = p_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        self.fp16_enabled = False

        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)

        # transformer layers
        self.encoder = build_transformer_layer_sequence(encoder)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.p_embedding_hw = nn.Embedding(self.p_h * self.p_w, self.embed_dims)
        self.p_embedding_zh = nn.Embedding(self.p_z * self.p_h, self.embed_dims)
        self.p_embedding_wz = nn.Embedding(self.p_w * self.p_z, self.embed_dims)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, CoHFF_MSDeformableAttention3D) or isinstance(m, CoHFF_CrossView_Self_Attention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # p queries and pos embeds
        p_queries_hw = self.p_embedding_hw.weight.to(dtype)
        p_queries_zh = self.p_embedding_zh.weight.to(dtype)
        p_queries_wz = self.p_embedding_wz.weight.to(dtype)
        p_queries_hw = p_queries_hw.unsqueeze(0).repeat(bs, 1, 1)
        p_queries_zh = p_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        p_queries_wz = p_queries_wz.unsqueeze(0).repeat(bs, 1, 1)

        p_pos_hw = self.positional_encoding(bs, device, 'z')
        p_pos_zh = self.positional_encoding(bs, device, 'w')
        p_pos_wz = self.positional_encoding(bs, device, 'h')
        p_pos = [p_pos_hw, p_pos_zh, p_pos_wz]
        
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        p_embed = self.encoder(
            [p_queries_hw, p_queries_zh, p_queries_wz],
            feat_flatten,
            feat_flatten,
            p_h=self.p_h,
            p_w=self.p_w,
            p_z=self.p_z,
            p_pos=p_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas,
        )
        
        return p_embed
    

from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
@POSITIONAL_ENCODING.register_module()
class CustomPositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 h, w, z,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__(init_cfg)
        if not isinstance(num_feats, list):
            num_feats = [num_feats] * 3
        self.h_embed = nn.Embedding(h, num_feats[0])
        self.w_embed = nn.Embedding(w, num_feats[1])
        self.z_embed = nn.Embedding(z, num_feats[2])
        self.num_feats = num_feats
        self.h, self.w, self.z = h, w, z

    def forward(self, bs, device, ignore_axis='z'):
        if ignore_axis == 'h':
            h_embed = torch.zeros(1, 1, self.num_feats[0], device=device).repeat(self.w, self.z, 1) # w, z, d
            w_embed = self.w_embed(torch.arange(self.w, device=device))
            w_embed = w_embed.reshape(self.w, 1, -1).repeat(1, self.z, 1)
            z_embed = self.z_embed(torch.arange(self.z, device=device))
            z_embed = z_embed.reshape(1, self.z, -1).repeat(self.w, 1, 1)
        elif ignore_axis == 'w':
            h_embed = self.h_embed(torch.arange(self.h, device=device))
            h_embed = h_embed.reshape(1, self.h, -1).repeat(self.z, 1, 1)
            w_embed = torch.zeros(1, 1, self.num_feats[1], device=device).repeat(self.z, self.h, 1)
            z_embed = self.z_embed(torch.arange(self.z, device=device))
            z_embed = z_embed.reshape(self.z, 1, -1).repeat(1, self.h, 1)
        elif ignore_axis == 'z':
            h_embed = self.h_embed(torch.arange(self.h, device=device))
            h_embed = h_embed.reshape(self.h, 1, -1).repeat(1, self.w, 1)
            w_embed = self.w_embed(torch.arange(self.w, device=device))
            w_embed = w_embed.reshape(1, self.w, -1).repeat(self.h, 1, 1)
            z_embed = torch.zeros(1, 1, self.num_feats[2], device=device).repeat(self.h, self.w, 1)

        pos = torch.cat(
            (h_embed, w_embed, z_embed), dim=-1).flatten(0, 1).unsqueeze(0).repeat(
                bs, 1, 1)
        return pos
