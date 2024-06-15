import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS


from torch.autograd import Variable

@HEADS.register_module()
class CoHFF_Predictionhead_Conv(BaseModule):
    def __init__(
        self, p_h, p_w, p_z, nbr_classes=12, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=True
    ):
        super().__init__()
        self.p_h = p_h
        self.p_w = p_w
        self.p_z = p_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z


        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        self.conv3d_layer1=nn.Conv3d(in_channels=128, out_channels=256,kernel_size=3,padding=1)
        self.norm1=nn.LayerNorm([256, 100, 100, 8])
        self.relu1=nn.ReLU()
        self.conv3d_layer2=nn.Conv3d(in_channels=256, out_channels=128,kernel_size=3,padding=1)
        self.norm2=nn.LayerNorm([128, 100, 100, 8])
        self.relu2=nn.ReLU()
        self.conv2d_layer1=nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3,padding=1)
        self.norm3=nn.BatchNorm2d(256)
        self.relu3=nn.ReLU()       
        self.conv2d_layer2=nn.Conv2d(in_channels=256, out_channels=128,kernel_size=3,padding=1)
        self.norm4=nn.BatchNorm2d(128)
        self.relu4=nn.ReLU()


    
    def forward(self, p_list, points=None):
        """
        p_list[0]: bs, h*w, c
        p_list[1]: bs, z*h, c
        p_list[2]: bs, w*z, c

        for i in range(len(p_list)):
            bs, _, c = p_list[i].shape
            if i == 0:
                p_list[i] = p_list[i].permute(0, 2, 1).reshape(bs, c, self.p_h, self.p_w)
            elif i % 2:
                p_list[i] = p_list[i].permute(0, 2, 1).reshape(bs, c, self.p_z, self.p_h)
            else:
                p_list[i] = p_list[i].permute(0, 2, 1).reshape(bs, c, self.p_w, self.p_z)
            
            
        """

        for i in range(len(p_list)):
            bs, _, c = p_list[i].shape
            p_list[i]=torch.permute(p_list[i],(0, 2, 1))
            if i == 0:
                p_list[i] = torch.reshape(p_list[i],(bs, c, self.p_h, self.p_w))
                p_list[i] = self.conv2d_layer1(p_list[i])
                p_list[i] = self.norm3(p_list[i])
                p_list[i] = self.relu3(p_list[i])
                p_list[i] = self.conv2d_layer2(p_list[i])
                p_list[i] = self.norm4(p_list[i])
                p_list[i] = self.relu4(p_list[i])
            elif i % 2:
                p_list[i] =  torch.reshape(p_list[i],(bs, c, self.p_z, self.p_h))
                p_list[i] = self.conv2d_layer1(p_list[i])
                p_list[i] = self.norm3(p_list[i])
                p_list[i] = self.relu3(p_list[i])
                p_list[i] = self.conv2d_layer2(p_list[i])
                p_list[i] = self.norm4(p_list[i])
                p_list[i] = self.relu4(p_list[i])
            else:
                p_list[i] =  torch.reshape(p_list[i],(bs, c, self.p_w, self.p_z))
                p_list[i] = self.conv2d_layer1(p_list[i])
                p_list[i] = self.norm3(p_list[i])
                p_list[i] = self.relu3(p_list[i])
                p_list[i] = self.conv2d_layer2(p_list[i])
                p_list[i] = self.norm4(p_list[i])
                p_list[i] = self.relu4(p_list[i])       

        if self.scale_h != 1 or self.scale_w != 1:
            p_list[0]= F.interpolate(
                p_list[0], 
                size=(self.p_h*self.scale_h, self.p_w*self.scale_w),
                mode='bilinear'
            )

        if self.scale_z != 1 or self.scale_h != 1:
            for i in range(len(p_list)):
                if i!=0 and i % 2:
                    p_list[i] = F.interpolate(
                        p_list[i], 
                        size=(self.p_z*self.scale_z, self.p_h*self.scale_h),
                        mode='bilinear'
                    )
        if self.scale_w != 1 or self.scale_z != 1:
            for i in range(len(p_list)):
                if i!=0 and not i % 2:            
                    p_list[i] = F.interpolate(
                        p_list[i], 
                        size=(self.p_w*self.scale_w, self.p_z*self.scale_z),
                        mode='bilinear'
                    )
        
        if points is not None:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3)
            points[..., 0] = torch.div(points[..., 0], (self.p_w*self.scale_w) * 2 - 1) 
            points[..., 1] = torch.div(points[..., 1], (self.p_h*self.scale_h) * 2 - 1)
            points[..., 2] = torch.div(points[..., 2], (self.p_z*self.scale_z) * 2 - 1)
            
            p_list_pts=[]
            for i in range(len(p_list)):
                if i == 0:
                    #sample_loc = torch.rand(1, 128, 200, 200).cuda()
                    sample_loc = Variable(points[:, :, :, [0, 1]].clone())
                    p_pts = F.grid_sample(p_list[0], sample_loc).squeeze(2)
                    
                elif i % 2 == 1:
                    sample_loc = points[:, :, :, [1, 2]].clone()
                    p_pts = F.grid_sample(p_list[i], sample_loc).squeeze(2)
                else:
                    sample_loc = points[:, :, :, [2, 0]].clone()
                    p_pts = F.grid_sample(p_list[i], sample_loc).squeeze(2)  
                p_list_pts.append(p_pts)


            p_list_vox=[]

            for i in range(len(p_list)):
                if i == 0:
                    p_list[i]=torch.unsqueeze(p_list[i],-1)
                    p_list[i]=torch.permute(p_list[i],(0, 1, 3, 2, 4))
                    p_list[i]=p_list[i].expand(-1, -1, -1, -1, self.scale_z*self.p_z)
                    p_vox=p_list[i]
                    
                elif i % 2:
                    p_list[i]=torch.unsqueeze(p_list[i],-1)
                    p_list[i]=torch.permute(p_list[i],(0, 1, 4, 3, 2))
                    p_list[i]=p_list[i].expand(-1, -1, self.scale_w*self.p_w, -1, -1)
                    p_vox=p_list[i]
                    
                else:
                    p_list[i]=torch.unsqueeze(p_list[i],-1)
                    p_list[i]=torch.permute(p_list[i],(0, 1, 2, 4, 3))
                    p_list[i]=p_list[i].expand(-1, -1, -1, self.scale_h*self.p_h, -1)
                    p_vox=p_list[i]
                    
                p_list_vox.append(p_vox)


            # sum of embedings 
            for i in range(len(p_list_vox)):
                fused_vox = torch.sum(torch.stack(p_list_vox),dim=0)

            fused_vox=self.conv3d_layer1(fused_vox)
            fused_vox=self.norm1(fused_vox)
            fused_vox=self.relu1(fused_vox)
            fused_vox=self.conv3d_layer2(fused_vox)
            fused_vox=self.norm2(fused_vox)
            fused_vox=self.norm2(fused_vox)
            fused_vox=fused_vox.flatten(2)
            for i in range(len(p_list_pts)):
                fused_pts = torch.sum(torch.stack(p_list_pts),dim=0)

            fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
            
            fused = fused.permute(0, 2, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits=torch.permute(logits,(0, 2, 1))

            logits_vox = logits[:, :, :(-n)]
            logits_vox=torch.reshape(logits_vox,(bs, self.classes, self.scale_w*self.p_w, self.scale_h*self.p_h, self.scale_z*self.p_z))
           
            logits_pts = logits[:, :, (-n):]
            logits_pts=torch.reshape(logits_pts,(bs, self.classes, n, 1, 1))

            return logits_vox, logits_pts
            
        else:
            
            p_list_vox=[]

            for i in range(len(p_list)):
                if i == 0:
                    p_list[i]=torch.unsqueeze(p_list[i],-1)
                    p_list[i]=torch.permute(p_list[i],(0, 1, 3, 2, 4))
                    p_vox=p_list[i].expand(-1, -1, -1, -1, self.scale_z*self.p_z)
                   
                elif i % 2:
                    p_list[i]=torch.unsqueeze(p_list[i],-1)
                    p_list[i]=torch.permute(p_list[i],(0, 1, 4, 3, 2))
                    p_vox=p_list[i].expand(-1, -1, self.scale_w*self.p_w, -1, -1)
                   
                else:
                    p_list[i]=torch.unsqueeze(p_list[i],-1)
                    p_list[i]=torch.permute(p_list[i],(0, 1, 2, 4, 3))
                    p_vox=p_list[i].expand(-1, -1, -1, self.scale_h*self.p_h, -1)
                    
                p_list_vox.append(p_vox)

            for i in range(len(p_list_vox)):
                if i ==0:
                    fused=p_list_vox[0]
                else:
                    fused=fused+p_list_vox[i]
            fused=self.conv3d_layer1(fused)
            fused=self.norm1(fused)
            fused=self.relu1(fused)
            fused=self.conv3d_layer2(fused)
            fused=self.norm2(fused)
            fused=self.relu2(fused)
            fused = fused.permute(0, 2, 3, 4, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)
        
            return logits
