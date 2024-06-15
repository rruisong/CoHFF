from ..encoder.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv import Config
from mmseg.models import builder
import torch
from mmcv import ConfigDict
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmseg.models import HEADS
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule


from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@HEADS.register_module()
class CoHFF_Decoder_Head(BaseModule):

    def __init__(self,
                 positional_encoding=None,
                 p_h=200,
                 p_w=200,
                 p_z=16,
                 co_decoder=None,
                 embed_dims=128,
                 cav_aggregator=True,
                 **kwargs):
        super().__init__()

        self.p_h = p_h
        self.p_w = p_w
        self.p_z = p_z
        self.embed_dims = embed_dims
        self.fp16_enabled = False
        self.cav_aggregator = cav_aggregator

        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)

        # transformer layers
        self.decoder = build_transformer_layer_sequence(co_decoder)


    #@auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, ego_p, cav_p_list):
        bs = ego_p[0].shape[0]
        device= ego_p[0].device
        num_cav = len(cav_p_list) # remove ego
        p_pos_hw_ego = self.positional_encoding(bs, device, 'z')
        p_pos_zh_ego = self.positional_encoding(bs, device, 'w')
        p_pos_wz_ego = self.positional_encoding(bs, device, 'h')

        p_pos_zh_cav_list = []
        p_pos_wz_cav_list = []
        for _ in range(num_cav):
            p_pos_zh_cav = self.positional_encoding(bs, device, 'w')
            p_pos_wz_cav = self.positional_encoding(bs, device, 'h')

            p_pos_zh_cav_list.append(p_pos_zh_cav)
            p_pos_wz_cav_list.append(p_pos_wz_cav)
        
        p_pos = [p_pos_hw_ego, p_pos_zh_ego, p_pos_wz_ego]
        for i in range(num_cav):
            p_pos_zh_cav=p_pos_zh_cav_list[i]
            p_pos_wz_cav=p_pos_wz_cav_list[i]
            p_pos.append(p_pos_zh_cav)
            p_pos.append(p_pos_wz_cav)


        query=[ego_p[0], ego_p[1], ego_p[2]]
        for i in range(num_cav):
            query.append(cav_p_list[i][1])
            query.append(cav_p_list[i][2])

        p_embed = self.decoder(
            query,
            p_h=self.p_h,
            p_w=self.p_w,
            p_z=self.p_z,
            p_pos=p_pos,
            num_cav= num_cav,
        )
        result=[]
        for i in range(len(p_embed)):
            result.append(p_embed[i])

        return result

