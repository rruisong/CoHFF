import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS
from mmseg.models import SEGMENTORS, builder
from config.load_yaml import task_config

from torch.autograd import Variable

model_type=task_config["model_type"]

@SEGMENTORS.register_module()
class CompConvModel(BaseModule):
    def __init__(
        self, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,**kwargs,
    ):
        super().__init__()
        out_dims = in_dims if out_dims is None else out_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes

        self.conv3d_layer1=nn.Conv3d(in_channels=1, out_channels=256,kernel_size=3,padding=1)
        self.norm1=nn.LayerNorm([256, 100, 100, 8])
        self.relu1=nn.ReLU()
        self.conv3d_layer2=nn.Conv3d(in_channels=256, out_channels=512,kernel_size=3,padding=1)
        self.norm2=nn.LayerNorm([512, 100, 100, 8])
        self.relu2=nn.ReLU()

    def forward(self, input_voxel, points=None):
            bs, _, _, _ = input_voxel.shape
            input_voxel = input_voxel.unsqueeze(1)
            input_voxel=self.conv3d_layer1(input_voxel)
            input_voxel=self.norm1(input_voxel)
            input_voxel=self.relu1(input_voxel)
            input_voxel=self.conv3d_layer2(input_voxel)
            input_voxel=self.norm2(input_voxel)
            input_voxel=self.relu2(input_voxel)
            if model_type == 2 or model_type == 3:
                return input_voxel
            elif model_type == 0 or model_type == 1:
           
                input_voxel = input_voxel.permute(0, 2, 3, 4, 1)

                input_voxel = self.decoder(input_voxel)
                logits = self.classifier(input_voxel)
                logits = logits.permute(0, 4, 1, 2, 3)
            
                return logits
            else:
                print('Unknown')
                exit()