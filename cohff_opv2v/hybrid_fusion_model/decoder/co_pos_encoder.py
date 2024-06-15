import torch.nn as nn
import torch
from mmcv.runner import force_fp32, auto_fp16
from config.load_yaml import task_config


model_type=task_config["model_type"]

class GateNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(x))
 

class CO_POS_ENCODER(nn.Module):
    def __init__(self, 
                p_h=200,
                p_w=200,
                p_z=16,
                embed_dims=128,
                relative_coord_input_dim=2,
                relative_angle_input_dim=1):
        super(CO_POS_ENCODER, self).__init__()
        self.p_h=p_h
        self.p_w=p_w
        self.p_z=p_z
        self.relative_coord_input_dim = relative_coord_input_dim
        self.relative_angle_input_dim = relative_angle_input_dim
        self.relative_position_coord_encoder = nn.Sequential(
            nn.Linear(self.relative_coord_input_dim, embed_dims),
            nn.Linear(embed_dims, 2 * embed_dims),
            nn.Linear(2 * embed_dims, embed_dims)
        )
        if model_type == 2:
            self.gatenetwork = GateNetwork(embed_dims, embed_dims)
        self.relative_position_angle_encoder = nn.Sequential(nn.Linear(self.relative_angle_input_dim, embed_dims))        
 
 
    @auto_fp16()
    def forward(self,
                relative_pos, cav_output, pos_emb_method):
        cav_output = torch.cat(cav_output, dim=1)
        if model_type == 2:
            gate_weights = self.gatenetwork(cav_output)      
            output_plane = cav_output * gate_weights
 
        relative_pos = relative_pos.view(1, -1)
 
        coord_features = relative_pos[:, :self.relative_coord_input_dim]
        angle_feature = relative_pos[:, -self.relative_angle_input_dim:]

        coord_embed = self.relative_position_coord_encoder(coord_features)
        angle_embed = self.relative_position_angle_encoder(angle_feature)

        if model_type == 2:
            if pos_emb_method == 1:
                outs = output_plane * coord_embed + angle_embed 
                outs = outs + cav_output
            elif pos_emb_method == 2:
                pos_embed = coord_embed + angle_embed
                outs = output_plane * pos_embed
                outs = outs + cav_output
            elif pos_emb_method == 3:
                outs = output_plane + coord_embed + angle_embed
                outs = outs + cav_output
            
        else:
            if pos_emb_method == 1:
                outs = cav_output * coord_embed + angle_embed
            elif pos_emb_method == 2:
                pos_embed = coord_embed + angle_embed
                outs = cav_output * pos_embed
            elif pos_emb_method == 3:
                outs = cav_output + coord_embed + angle_embed
            
        outs = torch.split(outs, [self.p_h*self.p_w, self.p_z*self.p_h, self.p_w*self.p_z], dim=1)              
    
        return outs
 