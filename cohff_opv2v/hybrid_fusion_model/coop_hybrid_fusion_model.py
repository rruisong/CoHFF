from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmseg.models import SEGMENTORS, builder, build_segmentor
from .decoder.co_pos_encoder import CO_POS_ENCODER
import warnings
#from dataloader.grid_mask import GridMask
from config.load_yaml import task_config
import torch
import os, time, argparse, os.path as osp, numpy as np
import numpy as np
from builder import cohff_opv2v_modelbuilder as model_builder

model_type=task_config["model_type"]

@SEGMENTORS.register_module()
class CoopHybridFusionModel(BaseModule):

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
        self.proj = torch.nn.Linear(512, 128)
        self.max_connect_cav=max_connect_cav
            
    @auto_fp16(apply_to=('img', 'points'))

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


        for i in range(len(outs)):
            bs, _, c = outs[i].shape
            outs[i]=torch.permute(outs[i],(0, 2, 1))
            
            if i == 0:
                outs[i] = torch.reshape(outs[i],(bs, c, self.p_h, self.p_w))
            elif i % 2:
                outs[i] =  torch.reshape(outs[i],(bs, c, self.p_z, self.p_h))
            else:
                outs[i] =  torch.reshape(outs[i],(bs, c, self.p_w, self.p_z))
        p_list_vox = []
        for i in range(len(outs)):
            if i == 0:
                outs[i]=torch.unsqueeze(outs[i],-1)
                outs[i]=torch.permute(outs[i],(0, 1, 3, 2, 4))
                p_vox=outs[i].expand(-1, -1, -1, -1, self.scale_z*self.p_z)
            elif i % 2:
                outs[i]=torch.unsqueeze(outs[i],-1)
                outs[i]=torch.permute(outs[i],(0, 1, 4, 3, 2))
                p_vox=outs[i].expand(-1, -1, self.scale_w*self.p_w, -1, -1)
            else:
                outs[i]=torch.unsqueeze(outs[i],-1)
                outs[i]=torch.permute(outs[i],(0, 1, 2, 4, 3))
                p_vox=outs[i].expand(-1, -1, -1, self.scale_h*self.p_h, -1)
            p_list_vox.append(p_vox)
        
        ego_vox_feature = p_list_vox[0] + p_list_vox[1] + p_list_vox[2]
        cav_vox_list = []
        for i in range(3, len(p_list_vox), 2):
            cav_vox_list.append(p_list_vox[i] + p_list_vox[i + 1])
        

        if len(cav_vox_list)<self.max_connect_cav:
            mask_cav_vox=torch.zeros_like(cav_vox_list[0], device = cav_vox_list[0].device)
            count=len(cav_vox_list)
            while count<self.max_connect_cav:
                cav_vox_list.append(mask_cav_vox)
                count+=1
        cav_vox_feature = torch.cat(cav_vox_list, dim=1)

        if model_type == 1:
            fused_vox = torch.concat((ego_vox_feature, cav_vox_feature), dim=1)
            predict_occ = torch.argmax(rich_vox, dim=1)
            fused_vox = fused_vox.squeeze(0)
            if torch.is_tensor(predict_occ):
                predict_occ = predict_occ.detach().cpu().numpy()
            if torch.is_tensor(fused_vox):
                fused_vox = fused_vox.detach().cpu().numpy()
            mask = predict_occ == 0
            mask = np.squeeze(mask)
            for i in range(len(fused_vox)):
                fused_vox[i][mask] = 0
            fused_vox = fused_vox
            fused_vox = torch.tensor(fused_vox).to('cuda', non_blocking=True)
            fused_vox = fused_vox.unsqueeze(0)
            outs = self.task_fusion_model(fused_vox)
            
        elif model_type == 2:
            rich_vox=torch.permute(rich_vox,(0, 2, 3, 4, 1))
            rich_vox = self.proj(rich_vox)
            rich_vox=torch.permute(rich_vox,(0, 4, 1, 2, 3))
            fused_vox = torch.concat((rich_vox, ego_vox_feature, cav_vox_feature), dim=1)
            outs = self.task_fusion_model(fused_vox)
            
        else:
            print("unknown")
            exit()
        return outs
    
    
def normal(x, mu=0, sigma=5):
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


