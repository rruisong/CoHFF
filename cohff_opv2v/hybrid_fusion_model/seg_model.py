from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmseg.models import SEGMENTORS, builder
from .decoder.co_pos_encoder import CO_POS_ENCODER
import warnings
from opv2v_dataset.grid_mask import GridMask
from config.load_yaml import task_config
import torch
import numpy as np

gi= task_config["gi"]
task_type=task_config["task_type"]
model_type=task_config["model_type"]
pos_emb_type=task_config["pos_emb_type"]


@SEGMENTORS.register_module()
class SegFormer(BaseModule):

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 encoder_head=None,
                 co_decoder_head=None,
                 co_pos_encoder=None,
                 pretrained=None,
                 predictionhead=None,
                 time_decoder_head=None,
                 **kwargs,
                 ):

        super().__init__()
        if encoder_head:
            self.encoder_head = builder.build_head(encoder_head)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)
        if predictionhead:
            self.predictionhead = builder.build_head(predictionhead)
        if co_decoder_head:
            self.co_decoder_head = builder.build_head(co_decoder_head)
        if co_pos_encoder:
            self.co_pos_encoder = CO_POS_ENCODER(p_h=co_pos_encoder.p_h, p_w=co_pos_encoder.p_w, p_z=co_pos_encoder.p_z, embed_dims=co_pos_encoder.embed_dims, relative_coord_input_dim=co_pos_encoder.relative_coord_input_dim, relative_angle_input_dim=co_pos_encoder.relative_angle_input_dim)

        if pretrained is None:
            img_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, use_grid_mask=None):
        """Extract features of images."""
        
        B = img.size(0)
        if img is not None:

            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            if use_grid_mask is None:
                use_grid_mask = self.use_grid_mask
            if use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if hasattr(self, 'img_neck'):
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img', 'points'))
    # image input
    def forward(self,
                points=None,
                input_data_dict=None,
                ego_history_imgs=None,
                ego_history_img_metas=None,
                use_grid_mask=None,
                spar_ratio = 0
        ):
        """Forward training function.
        """

        ego_id = input_data_dict["ego"]

        outs_cav_list = []
        for cav_id in input_data_dict["cav_ids"]:
            if cav_id == ego_id:
                #cav_id = input_data_dict["ego"]
                img_ego = input_data_dict[ego_id]["img"]
                img_metas_ego =input_data_dict[ego_id]["img_metas"]
                img_feats_ego = self.extract_img_feat(img=img_ego, use_grid_mask=use_grid_mask)
                outs_ego = self.encoder_head(img_feats_ego, img_metas_ego) 
                
                ego_rel_pos = input_data_dict[cav_id]["pos"] - input_data_dict[ego_id]["pos"]
                if gi:
                    relative_pos = normal(ego_rel_pos.clone().cpu().numpy()) #.cuda()
                else:
                    relative_pos = ego_rel_pos.clone()
                relative_pos = torch.tensor(relative_pos, dtype=torch.float32, device=img_ego.device)
                outs_ego = self.co_pos_encoder(relative_pos, outs_ego, pos_emb_method=pos_emb_type)
            else: # cav_id
                img_cav=input_data_dict[cav_id]["img"]
                img_metas_cav=input_data_dict[cav_id]["img_metas"]
                img_feats_cav = self.extract_img_feat(img=img_cav, use_grid_mask=use_grid_mask) # image features
                outs_cav = self.encoder_head(img_feats_cav, img_metas_cav) # 3 planes
                
                if task_type==2:
                    spar_ratio = 1 - spar_ratio
                    plane_1 = outs_cav[1]
                    param_list = torch.flatten(plane_1).tolist()
                    tensorlen = len(param_list)
                    torch_param = torch.FloatTensor(param_list)
                    topk = torch.topk(torch.abs(torch_param), max(int(tensorlen * spar_ratio), 1), 0, largest=True, sorted=False)
                    threshold = torch.min(topk[0])
                    outs_cav[1][torch.abs(outs_cav[1]) < threshold] = 0
            
                    plane_2 = outs_cav[2]
                    param_list = torch.flatten(plane_2).tolist()
                    tensorlen = len(param_list)
                    torch_param = torch.FloatTensor(param_list)
                    topk = torch.topk(torch.abs(torch_param), max(int(tensorlen * spar_ratio), 1), 0, largest=True, sorted=False)
                    threshold = torch.min(topk[0])
                    outs_cav[2][torch.abs(outs_cav[2]) < threshold] = 0
                
                cav_rel_pos = input_data_dict[cav_id]["pos"] - input_data_dict[ego_id]["pos"] # rel pos
                    
                if gi:
                    relative_pos = normal(cav_rel_pos.clone().cpu().numpy()) 
                else:
                    relative_pos = cav_rel_pos.clone()
                relative_pos = torch.tensor(relative_pos, dtype=torch.float32, device=img_ego.device)
                outs_cav = self.co_pos_encoder(relative_pos, outs_cav, pos_emb_method=pos_emb_type) # mlp to process gaus and mul 3 planes
                outs_cav_list.append(outs_cav) # 3 planes in one list
        
        # Communication and Fusion
        out_p =self.co_decoder_head(outs_ego, outs_cav_list) # Flatten and concatenate planes and 2 layer self attention -> Embeddings 
        # output is the long list of all planes
        if (model_type == 1 or model_type == 2) and task_type != 1:
            return out_p
        elif model_type == 0 or task_type ==1:
        #----------------
            outs = self.predictionhead(out_p, points)
            return outs
        else:
            print("Unknown")
            exit()
    
def normal(x, mu=0, sigma=5):
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
    
