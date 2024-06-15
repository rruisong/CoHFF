import os, time, argparse, os.path as osp, numpy as np
from tqdm import tqdm
from datetime import datetime
from datetime import date
import copy
import warnings
warnings.filterwarnings("ignore")
import torch

from opv2v_dataset.opv2v_dataset import get_opv2v_label_name

from builder import loss_builder
from builder import cohff_opv2v_modelbuilder as model_builder
from builder import cohff_opv2v_databuilder

from utils.metric_util import MeanIoU
from utils.focal_loss import FocalLoss
from utils.train_util import info_to_device, get_pvox, add_random_mask

from mmcv import Config
from mmcv.runner import build_optimizer
from timm.scheduler import CosineLRScheduler

from config.label_config.opv2v_label_mapping import opv2v_label_mapping_dict
from config.load_yaml import task_config 

import json
from json import JSONEncoder
import pickle

json_types = (list, dict, str, int, float, bool, type(None))

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct

def train_parser():
    parser = argparse.ArgumentParser(description="CoHFF")
    parser.add_argument("--py-config", type=str, default='./config/model_config/cfg_cohff_opv2v.py',
                        help="Configuration py file")
    parser.add_argument('--work-dir', type=str, default='./out/opv2v',
                        help='Output directory')
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    return args

def main():

    args = train_parser()

    task_type=task_config["task_type"]
    supervision=task_config["supervision"]
    log_freq=task_config["log_freq"]
    loss_freq=task_config["loss_freq"]
    save_freq=task_config["save_freq"]
    input_mask = torch.zeros((30, 30, 8))

    # global settings
    today = date.today()
    now = datetime.now().time()
    current_time = today.strftime("%Y%m%d_") + now.strftime("%H%M%S")


    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
    gradient_accumulate_steps = cfg.gradient_accumulate_steps

    torch.cuda.set_device(1)

    os.makedirs(args.work_dir, exist_ok=True)
    cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    my_model = model_builder.build(cfg.model)
    my_model = my_model.to('cuda', non_blocking=True)

    # generate datasets
    CoHFFOpv2v_label_name = get_opv2v_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [CoHFFOpv2v_label_name[x] for x in unique_label]

    train_dataset_loader, val_dataset_loader = \
        cohff_opv2v_databuilder.build(
            train_dataloader_config,
            val_dataloader_config,
            dist=False)
    train_dataloader = train_dataset_loader
    val_dataloader =val_dataset_loader

    CalMeanIou_vox = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_vox_seg = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_vox_occ = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    
    # resume and load
    epoch = 0
    global_iter = 0

    if task_type == 0:
        criterion = FocalLoss(gamma=2)
    elif task_type == 1:
        criterion = loss_builder.build(ignore_label=0)
    if task_type == 2:
        criterion = loss_builder.build(ignore_label=-999)
    
    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataloader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
   
    res_train_loss_list = []
    for epoch in range(max_num_epochs):
        my_model.train()
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        loss_list = []

        data_time_s = time.time()
        time_s = time.time()
        accumulated_loss = 0
        pbar = tqdm(enumerate(train_dataloader))
        for i_iter, [input_data_dict, co_processed_label, ego_processed_label, co_grid_ind, ego_lidar_processed_label, ego_history_data_dict] in pbar:

            input_data_dict = input_data_dict[0]
            if supervision == 0:
                co_processed_label = ego_processed_label
            
            if task_type == 0:
                co_processed_label[co_processed_label!=0] = 1
                
            elif task_type == 1:
                for key in opv2v_label_mapping_dict.keys():
                    co_processed_label[co_processed_label==key] = opv2v_label_mapping_dict[key]

            if task_type == 2:
                for key in opv2v_label_mapping_dict.keys():
                    co_processed_label[co_processed_label==key] = opv2v_label_mapping_dict[key]

            # get images
            input_data_info = {}
            
            ego_id = str(input_data_dict["ego"])
            input_data_info["ego"] = int(ego_id)
            input_data_info["cav_ids"] = []
            
            ego_pvox = get_pvox(ego_id, input_data_dict)
            ego_pvox = add_random_mask(ego_pvox, input_mask)
            
            # get ego history images (History information is not used in this work)
            ego_history_imgs, ego_history_img_metas = None, None
                        
            # pre-selection:
            max_connect_cav=cfg.max_connect_cav
            cav_id_list=[ego_id]
            cav_d_dict={}
            for cav_id in input_data_dict["cav_ids"]:
                if cav_id != ego_id:
                    curr_d = np.linalg.norm(np.array(input_data_dict[cav_id]["pos"][:2]) - np.array(input_data_dict[ego_id]["pos"][:2]))
                    cav_d_dict[cav_id]=curr_d
            sorted_cars = sorted(cav_d_dict.items(), key=lambda x: x[1])
            selected_cars = [car[0] for car in sorted_cars[:max_connect_cav]]
            cav_id_list.extend(selected_cars)
            if len(cav_id_list) == 1:            
                cav_id_list = [ego_id, -1]                
           
            input_data_info = info_to_device(input_data_info, input_data_dict, cav_id_list)       
            
            voxel_label = co_processed_label.type(torch.LongTensor).to('cuda', non_blocking=True)

            # forward + backward + optimize
            data_time_e = time.time()
            if task_type == 0:
                outputs_vox = my_model(ego_pvox)
            if task_type == 1:
                outputs_vox = my_model(points=None,
                                        input_data_dict=input_data_info,               
                                        ego_history_imgs=ego_history_imgs,
                                        ego_history_img_metas=ego_history_img_metas,
                                        use_grid_mask=None,
                                        spar_ratio = 0)
            if task_type == 2:
                outputs_vox = my_model(points=None,
                                    input_data_dict=input_data_info,               
                                    ego_history_imgs=ego_history_imgs,
                                    ego_history_img_metas=ego_history_img_metas,
                                    use_grid_mask=None,
                                    spar_ratio = 0,
                                    input_voxel=ego_pvox)

            if task_type == 0:
                co_processed_label[co_processed_label!=0] = 1
            elif task_type == 1:
                for key in opv2v_label_mapping_dict.keys():
                    co_processed_label[co_processed_label==key] = opv2v_label_mapping_dict[key]
            if task_type == 2:
                for key in opv2v_label_mapping_dict.keys():
                    co_processed_label[co_processed_label==key] = opv2v_label_mapping_dict[key]
            
            loss = criterion(outputs_vox, voxel_label)            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            accumulated_loss += loss.item()
            if (i_iter + 1) % gradient_accumulate_steps == 0 or (i_iter + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
                accumulated_loss = 0
            loss_list.append(loss.item())
            global_iter += 1 
            scheduler.step_update(global_iter)
            time_e = time.time()
            
            if not i_iter % log_freq:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_description('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataloader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
            
            if not i_iter % loss_freq:
                res_train_loss_list.append(loss)
                res_dir = os.path.join(os.path.abspath(args.work_dir), "model_result_"+current_time)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                with open(os.path.join(res_dir, 'loss_record.json'), "w") as jsfile:
                    json.dump(res_train_loss_list, jsfile, cls=PythonObjectEncoder)

        data_time_s = time.time()
        time_s = time.time()
        
        save_dict = True
        if not (epoch % save_freq) or not (epoch % (max_num_epochs-1)):
            if save_dict:
                dict_to_save = {
                        'state_dict': my_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch + 1,
                        'global_iter': global_iter,
                    }
                res_dir = os.path.join(os.path.abspath(args.work_dir), "model_result_"+current_time)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                save_file_name = os.path.join(res_dir, f'epoch_{epoch}_iter_{i_iter}.pth')
                torch.save(dict_to_save, save_file_name)

        my_model.eval()
        val_loss_list = []
        CalMeanIou_vox.reset()
        CalMeanIou_vox_seg.reset()
        CalMeanIou_vox_occ.reset()

        voxel_statistic = {}
        voxel_statistic[-1] = 0
        for i in range(1, 18):
            voxel_statistic[i] = []

        # Validation starts           
        pbar = tqdm(enumerate(val_dataloader))
        with torch.no_grad():
            for i_iter, [input_data_dict, co_processed_label, ego_processed_label, box_corner_list, ego_lidar_processed_label, ego_history_data_dict] in pbar:
                input_data_dict = input_data_dict[0]
                if supervision == 0:
                    co_processed_label = ego_processed_label
                
                if task_type == 0:
                    co_processed_label[co_processed_label!=0] = 1
                    ego_lidar_processed_label[ego_lidar_processed_label!=0] = 1
                    
                elif task_type == 1:
                    for key in opv2v_label_mapping_dict.keys():
                        co_processed_label[co_processed_label==key] = opv2v_label_mapping_dict[key]
                        
                if task_type == 2:
                    for key in opv2v_label_mapping_dict.keys():
                        co_processed_label[co_processed_label==key] = opv2v_label_mapping_dict[key]
                
                # Raw lidar label is used for eval the segmentation performance
                raw_lidar_label = ego_lidar_processed_label.type(torch.FloatTensor).to('cuda', non_blocking=True)

                input_data_info = {}
                
                ego_id = str(input_data_dict["ego"])
                input_data_info["ego"] = int(ego_id)
                input_data_info["cav_ids"] = []
                
                ego_pvox = get_pvox(ego_id, input_data_dict)

                # get ego history images (History information is not used in this work)
                ego_history_imgs, ego_history_img_metas = None, None
                            
                # pre-selection:
                max_connect_cav=cfg.max_connect_cav
                cav_id_list=[ego_id]
                cav_d_dict={}
                for cav_id in input_data_dict["cav_ids"]:
                    if cav_id != ego_id:
                        curr_d = np.linalg.norm(np.array(input_data_dict[cav_id]["pos"][:2]) - np.array(input_data_dict[ego_id]["pos"][:2]))
                        cav_d_dict[cav_id]=curr_d
                sorted_cars = sorted(cav_d_dict.items(), key=lambda x: x[1])
                selected_cars = [car[0] for car in sorted_cars[:max_connect_cav]]
                cav_id_list.extend(selected_cars)
                if len(cav_id_list) == 1:            
                    cav_id_list = [ego_id, -1]            
            
                input_data_info = info_to_device(input_data_info, input_data_dict, cav_id_list)       
                
                voxel_label = co_processed_label.type(torch.LongTensor).to('cuda', non_blocking=True)

    
                data_time_e = time.time()
                if task_type == 0:
                    predict_labels_vox = my_model(ego_pvox)
                if task_type == 1:
                    predict_labels_vox = my_model(points=None,
                                            input_data_dict=input_data_info,               
                                            ego_history_imgs=ego_history_imgs,
                                            ego_history_img_metas=ego_history_img_metas,
                                            use_grid_mask=None,
                                            spar_ratio = 0)
                if task_type == 2:
                    predict_labels_vox = my_model(points=None,
                                        input_data_dict=input_data_info,               
                                        ego_history_imgs=ego_history_imgs,
                                        ego_history_img_metas=ego_history_img_metas,
                                        use_grid_mask=None,
                                        spar_ratio = 0,
                                        input_voxel=ego_pvox)


                loss = criterion(predict_labels_vox, voxel_label)  

                predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)

                # occupancy eval
                occeval_predict_labels_vox = copy.deepcopy(predict_labels_vox)
                occeval_voxel_label = copy.deepcopy(voxel_label)
                
                occeval_predict_labels_vox[occeval_predict_labels_vox!=0] = copy.deepcopy(occeval_voxel_label[occeval_predict_labels_vox!=0])

                # segmentation eval
                segeval_predict_labels_vox = copy.deepcopy(predict_labels_vox)
                segeval_voxel_label = copy.deepcopy(voxel_label)

                # Raw lidar label is used for evaluate segementation performance
                segeval_voxel_label[raw_lidar_label==0] = 0
                segeval_predict_labels_vox[raw_lidar_label==0] = 0

                CalMeanIou_vox_occ._after_step(
                    occeval_predict_labels_vox.squeeze(0).flatten(),
                    occeval_voxel_label.flatten())
                CalMeanIou_vox_seg._after_step(
                    segeval_predict_labels_vox.squeeze(0).flatten(),
                    segeval_voxel_label.flatten())
                CalMeanIou_vox._after_step(
                    predict_labels_vox.squeeze(0).flatten(),
                    voxel_label.flatten())

                val_loss_list.append(loss.detach().cpu().numpy())

                pbar.set_description('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter, loss.item(), np.mean(val_loss_list)))
        
                pbar.update(1)
        if task_type==0:    
            evalocc_val_miou_vox = CalMeanIou_vox_occ._after_epoch(res_dir, "occ_only_res.json")
        if task_type==1:
            evalseg_val_miou_vox = CalMeanIou_vox_seg._after_epoch(res_dir, "seg_only_res.json")
        if task_type==2:
            val_miou_vox = CalMeanIou_vox._after_epoch(res_dir, "occ_seg_res.json")
        
        epoch += 1


if __name__ == '__main__':
    main()
