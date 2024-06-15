from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmseg.models import SEGMENTORS, builder, build_segmentor
from .decoder.co_pos_encoder import CO_POS_ENCODER
import warnings
import torch
import os, time, argparse, os.path as osp, numpy as np
import numpy as np
from builder import cohff_opv2v_modelbuilder as model_builder
#
@SEGMENTORS.register_module()
class HybridFusionModel(BaseModule):

    def __init__(self,
                 segmentation_net=None,
                 completion_net=None,
                 task_fusion_model =None,
                 co_pos_encoder = None,
                 segmentation_net_path=None,
                 completion_net_path=None,
                 max_connect_cav=None,
                 **kwargs,
                 ):

        super().__init__()
        if segmentation_net:
            self.segmentation_net = build_segmentor(segmentation_net)
        ckpt_path = segmentation_net_path
        assert osp.isfile(ckpt_path)
        resume_from = ckpt_path   
        map_location = 'cpu'
        ckpt = torch.load(resume_from, map_location=map_location)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(ckpt.keys())
        print(self.segmentation_net.load_state_dict(ckpt, strict=False))
        print(f'successfully loaded ckpt') 
            
        if completion_net:
            self.completion_net = build_segmentor(completion_net)

        ckpt_path = completion_net_path
        assert osp.isfile(ckpt_path)
        resume_from = ckpt_path   
        map_location = 'cpu'
        ckpt = torch.load(resume_from, map_location=map_location)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(ckpt.keys())
        print(self.completion_net.load_state_dict(ckpt, strict=False))
        print(f'successfully loaded ckpt')    
        
        if task_fusion_model:
            self.task_fusion_model = builder.build_neck(task_fusion_model)
        if co_pos_encoder:
            self.co_pos_encoder = CO_POS_ENCODER(p_h=co_pos_encoder.p_h, p_w=co_pos_encoder.p_w, p_z=co_pos_encoder.p_z, 
                                                 embed_dims=co_pos_encoder.embed_dims, 
                                                 relative_coord_input_dim=co_pos_encoder.relative_coord_input_dim, 
                                                 relative_angle_input_dim=co_pos_encoder.relative_angle_input_dim)

        self.p_h = 100
        self.p_w = 100
        self.p_z = 8
        self.scale_h = 1
        self.scale_w = 1
        self.scale_z = 1
            
    @auto_fp16(apply_to=('img', 'points'))
    # image input
    def forward(self,
                points=None,
                input_data_dict=None,
                ego_history_imgs=None,
                ego_history_img_metas=None,
                use_grid_mask=None,
                spar_ratio = 0,
                input_voxel = None
        ):
        """Forward training function.
        """
        with torch.no_grad():
            rich_vox =self.completion_net(input_voxel)
            outs = self.segmentation_net(points=None,
                                    input_data_dict=input_data_dict,               
                                    ego_history_imgs=None,
                                    ego_history_img_metas=None,
                                    use_grid_mask=None,
                                    spar_ratio = spar_ratio)
            
            
        predict_occ = torch.argmax(rich_vox, dim=1)
        outs = outs.squeeze(0)
        if torch.is_tensor(predict_occ):
            predict_occ = predict_occ.detach().cpu().numpy()
        if torch.is_tensor(outs):
            outs = outs.detach().cpu().numpy()
        mask = predict_occ == 0
        mask = np.squeeze(mask)
        for i in range(len(outs)):
            outs[i][mask] = 0
        fused_vox = outs
        fused_vox = torch.tensor(fused_vox).to('cuda', non_blocking=True)
        fused_vox = fused_vox.unsqueeze(0)       
        outs = self.task_fusion_model(fused_vox)
        return outs
    
    
def normal(x, mu=0, sigma=5):
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


