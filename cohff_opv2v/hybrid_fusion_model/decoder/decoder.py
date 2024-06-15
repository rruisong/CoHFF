from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_transformer_layer
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CoHFF_Decoder(TransformerLayerSequence):

    """
    Attention with both self and cross attention.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, p_h, p_w, p_z, 
                 num_points_in_pillar=[4, 32, 32], 
                 num_points_in_pillar_cross_view=[32, 32, 32],
                 return_intermediate=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.p_h, self.p_w, self.p_z = p_h, p_w, p_z
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2] and num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.fp16_enabled = False
        
        self.num_points_cross_view = num_points_in_pillar_cross_view

    @staticmethod
    def get_co_cross_view_ref_points(p_h, p_w, p_z, num_points_in_pillar, num_cav):

        h_ranges = torch.linspace(0.5, p_h-0.5, p_h) / p_h
        w_ranges = torch.linspace(0.5, p_w-0.5, p_w) / p_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, p_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(p_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) 
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) 

        z_ranges = torch.linspace(0.5, p_z-0.5, num_points_in_pillar[2]) / p_z
        z_ranges = z_ranges.unsqueeze(0).expand(p_h*p_w, -1) 
        h_ranges = torch.linspace(0.5, p_h-0.5, p_h) / p_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, p_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) 

        z_ranges = torch.linspace(0.5, p_z-0.5, num_points_in_pillar[2]) / p_z
        z_ranges = z_ranges.unsqueeze(0).expand(p_h*p_w, -1)
        w_ranges = torch.linspace(0.5, p_w-0.5, p_w) / p_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(p_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) 
        
        w_ranges = torch.linspace(0.5, p_w-0.5, num_points_in_pillar[1]) / p_w
        w_ranges = w_ranges.unsqueeze(0).expand(p_z*p_h, -1)
        h_ranges = torch.linspace(0.5, p_h-0.5, p_h) / p_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(p_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)

        z_ranges = torch.linspace(0.5, p_z-0.5, p_z) / p_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, p_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, p_h-0.5, p_h) / p_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(p_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) 

        w_ranges = torch.linspace(0.5, p_w-0.5, num_points_in_pillar[1]) / p_w
        w_ranges = w_ranges.unsqueeze(0).expand(p_z*p_h, -1)
        z_ranges = torch.linspace(0.5, p_z-0.5, p_z) / p_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, p_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        h_ranges = torch.linspace(0.5, p_h-0.5, num_points_in_pillar[0]) / p_h
        h_ranges = h_ranges.unsqueeze(0).expand(p_w*p_z, -1)
        w_ranges = torch.linspace(0.5, p_w-0.5, p_w) / p_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, p_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)

        h_ranges = torch.linspace(0.5, p_h-0.5, num_points_in_pillar[0]) / p_h
        h_ranges = h_ranges.unsqueeze(0).expand(p_w*p_z, -1)
        z_ranges = torch.linspace(0.5, p_z-0.5, p_z) / p_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(p_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)

        w_ranges = torch.linspace(0.5, p_w-0.5, p_w) / p_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, p_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, p_z-0.5, p_z) / p_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(p_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1),
        ], dim=0) 

        for _ in range(num_cav):
            reference_points = torch.cat([
                reference_points,
                torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
                torch.stack([wz_hw, wz_zh, wz_wz], dim=1),
                ], dim=0)
        
        return reference_points.cuda()

    @auto_fp16()
    def forward(self,
                query, # list
                *args,
                p_h=None,
                p_w=None,
                p_z=None,
                p_pos=None, # list
                num_cav=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input p query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
        """
        
        intermediate = []
        bs = query[0].shape[0]

        cross_view_ref_points = self.get_co_cross_view_ref_points(self.p_h, self.p_w, self.p_z, self.num_points_cross_view, num_cav)        
        
        ref_cross_view = cross_view_ref_points.clone().unsqueeze(0).expand(bs, -1, -1, -1, -1)

        for lid, layer in enumerate(self.layers):
            query = layer(
                query,
                *args,
                p_pos=p_pos,
                ref_2d=ref_cross_view,
                p_h=p_h,
                p_w=p_w,
                p_z=p_z,
                num_cav=num_cav,
                **kwargs)      
        if self.return_intermediate:
            intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query