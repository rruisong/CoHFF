import os, argparse, copy, os.path as osp, numpy as np
from tqdm import tqdm
import torch

from mmcv import Config

import warnings
warnings.filterwarnings("ignore")

from utils.metric_util import MeanIoU

from builder import loss_builder
from opv2v_dataset.opv2v_dataset import get_opv2v_label_name
from opv2v_dataset.opv2v_dataset import plot_vox

from utils.focal_loss import FocalLoss
from utils.train_util import info_to_device, get_pvox

from config.load_yaml import task_config
from config.label_config.opv2v_label_mapping import opv2v_label_mapping_dict


def eval_parser():
    parser = argparse.ArgumentParser(description="CoHFF")
    parser.add_argument('--model-dir', required=True,
                        help='directory of the model for eval')
    parser.add_argument('--model-name', required=True,
                        help='name of the model for eval')
    parser.add_argument("--py-config", type=str, default='./config/model_config/cfg_cohff_opv2v.py',
                        help="Configuration py file")
    parser.add_argument('--work-dir', type=str, default='./out/opv2v',
                        help='Output directory')
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    return args

def main():

    args = eval_parser()

    task_type=task_config["task_type"]
    supervision=task_config["supervision"]
       
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    torch.cuda.set_device(1)

    os.makedirs(args.work_dir, exist_ok=True)
    cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import cohff_opv2v_modelbuilder as model_builder
    
    # Init and load model
    my_model = model_builder.build(cfg.model)
    model_folder = args.model_dir
    args.ckpt_path =os.path.join(model_folder, args.model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
   
    assert osp.isfile(args.ckpt_path)
    cfg.resume_from = args.ckpt_path
    print('ckpt path:', cfg.resume_from)
    map_location = 'cpu'
    ckpt = torch.load(cfg.resume_from, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(ckpt.keys())
    print(my_model.load_state_dict(ckpt, strict=True))
    print(f'successfully loaded ckpt')
    
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()

    # generate datasets
    CoHFFOpv2v_label_name = get_opv2v_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [CoHFFOpv2v_label_name[x] for x in unique_label]
    
    # Load data
    torch.random.seed = 10
    
    from builder import cohff_opv2v_databuilder
    train_dataset_loader, val_dataset_loader = \
        cohff_opv2v_databuilder.build(
            train_dataloader_config,
            val_dataloader_config,
            dist=False,
        )
    my_dataloader = val_dataset_loader
    CalMeanIou_vox = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_vox_seg = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_vox_occ = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    
    if task_type == 0:
        criterion = FocalLoss(gamma=2)
    elif task_type == 1:
        criterion = loss_builder.build(ignore_label=0)
    if task_type == 2:
        criterion = loss_builder.build(ignore_label=-999)

    # eval
    my_model.eval()
    val_loss_list = []
    CalMeanIou_vox.reset()
    CalMeanIou_vox_seg.reset()
    CalMeanIou_vox_occ.reset()

    voxel_statistic = {}
    voxel_statistic[-1] = 0
    for i in range(1, 18):
        voxel_statistic[i] = []
                    
    pbar = tqdm(enumerate(my_dataloader))
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

            orignals = torch.argmax(predict_labels_vox, dim=1)

            orignals = orignals.detach().cpu()
            orignals = orignals.squeeze(0)
            targets = voxel_label.squeeze(0)

            if task_type == 0 or task_type == 2:
                plot_vox(orignals, targets=orignals, ref=1, fig_name="pred", s_idx=input_data_dict[ego_id]["imgs"][0][-44:-11])
            if task_type==1:
                plot_vox(orignals, targets=targets, ref=1, fig_name="seg_only", s_idx=input_data_dict[ego_id]["imgs"][0][-44:-11])
            plot_vox(targets, targets=targets, ref=1, fig_name="gt",s_idx=input_data_dict[ego_id]["imgs"][0][-44:-11])          
            
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

            pbar.set_description('[EVAL] Iter %5d: Loss: %.3f (%.3f)'%(
                    i_iter, loss.item(), np.mean(val_loss_list)))
    
            pbar.update(1)
            
    # Results will be storaged in json-files
    if task_type==0:
        evalocc_val_miou_vox = CalMeanIou_vox_occ._after_epoch(model_folder, "occ_only_res.json")
    if task_type==1:
        evalseg_val_miou_vox = CalMeanIou_vox_seg._after_epoch(model_folder, "seg_only_res.json")
    if task_type==2:
        val_miou_vox = CalMeanIou_vox._after_epoch(model_folder, "occ_seg_res.json")
    

if __name__ == '__main__':
    main()
