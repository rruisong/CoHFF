import os, time, argparse, os.path as osp, numpy as np
import torch
from mmcv.image.io import imread

def info_to_device(input_data_info, input_data_dict, cav_id_list):
    for cav_id in cav_id_list:
                
        cav_id_int = int(cav_id)
        input_data_info["cav_ids"].append(cav_id_int)
        input_data_info[cav_id_int] = {}
    
        cav_imgs, cav_img_metas = get_images(cav_id, input_data_dict)
        
        input_data_info[cav_id_int]["img_metas"] = [cav_img_metas]
        input_data_info[cav_id_int]["img"] = cav_imgs
        input_data_info[cav_id_int]["pos"] = input_data_dict[cav_id]["pos"]
        
        ##
        input_data_info[cav_id_int]["pos"] = torch.FloatTensor(input_data_info[cav_id_int]["pos"]).cuda()   
        input_data_info[cav_id_int]["img_metas"][0]["lidar2img"] = torch.FloatTensor(input_data_info[cav_id_int]["img_metas"][0]["lidar2img"]).cuda()
        
        input_data_info[cav_id_int]["img_metas"][0]["img_shape"][0] = torch.FloatTensor(input_data_info[cav_id_int]["img_metas"][0]["img_shape"][0]).cuda()
        input_data_info[cav_id_int]["img_metas"][0]["img_shape"][1] = torch.FloatTensor(input_data_info[cav_id_int]["img_metas"][0]["img_shape"][1]).cuda()
        input_data_info[cav_id_int]["img_metas"][0]["img_shape"][2] = torch.FloatTensor(input_data_info[cav_id_int]["img_metas"][0]["img_shape"][2]).cuda()
        input_data_info[cav_id_int]["img_metas"][0]["img_shape"][3] = torch.FloatTensor(input_data_info[cav_id_int]["img_metas"][0]["img_shape"][3]).cuda()
    
    return input_data_info
    
    
def get_images(ego_id, input_data_dict):
    """Get image data of each cav

    Args:
        ego_id (int): Id of vehicle
        input_data_dict (dict): Input data

    Returns:
        ego_imgs: Image data
        ego_img_metas: Meta info of each images.
    """    
    
    ego_imgs_files = input_data_dict[ego_id]["imgs"]
    ego_img_metas = input_data_dict[ego_id]["img_metas"]
    ego_imgs = []
    for filename in ego_imgs_files:
        ego_imgs.append(
            imread(filename, 'unchanged').astype(np.float32)
        )
    ego_imgs_dict = {'img': ego_imgs, 'lidar2img': ego_img_metas['lidar2img'][0]}
    ego_imgs = ego_imgs_dict['img']
    ego_imgs = [img.transpose(2, 0, 1) for img in ego_imgs]
    ego_imgs = torch.FloatTensor(np.array(ego_imgs)).cuda()
    ego_imgs = ego_imgs.unsqueeze(0)
    return ego_imgs, ego_img_metas

def add_random_mask(tensor, mask):
    """Random mask augmentation to input depth

    Args:
        tensor (Tensor): Input depth
        mask (tensor): Mask tensor

    Returns:
        tensor: Masked tensor
    """    
    tensor_shape = tensor.shape
    mask_shape = mask.shape

    start_row = torch.randint(0, tensor_shape[1] - mask_shape[0], (1,)).item()
    start_col = torch.randint(0, tensor_shape[2] - mask_shape[1], (1,)).item()
    
    tensor[0, start_row:start_row + mask_shape[0], start_col:start_col + mask_shape[1], :] = mask

    return tensor

def get_pvox(ego_id, input_data_dict):
    """Get predicted depth 

    Args:
        ego_id (int): Id of vehicle
        input_data_dict (dict): Input data

    Returns:
        ego_pvox: predicted depth data
    """    
    filename = input_data_dict[ego_id]["imgs"][0]
    p_vox_fn = filename[:-11] + "pvox.npy"
    ego_pvox = np.load(p_vox_fn)
    ego_pvox = torch.FloatTensor(ego_pvox).cuda()
    ego_pvox = ego_pvox.unsqueeze(0)
    return ego_pvox