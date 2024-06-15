import random
import math
from collections import OrderedDict
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

import json

from mmcv.image.io import imread
from PIL import Image, ImageOps
import open3d as o3d
from opv2v_dataset.utils.box_utils import get_points_in_rotated_box_3d, boxes_to_corners_3d
from opv2v_dataset.utils.augmentor.data_augmentor import DataAugmentor
from opv2v_dataset.utils.yaml_utils import load_yaml
from opv2v_dataset.utils.transformation_utils import x_to_world
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from opv2v_dataset.utils.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class CO_ImagePoint_OPV2V_V2_LANE(Dataset):
    def __init__(self, params, train=True, gps_noise_std = 0):
        self.params = params
        self.train = train
        self.scale_rate=params['scale_rate']
        self.grid_size = np.asarray(params['grid_size'])
        self.fill_label = params['fill_label']
        self.max_volume_space = params['max_volume_space']
        self.min_volume_space = params['min_volume_space']

        self.gps_noise_std = gps_noise_std

        self.pre_processor = None
        self.post_processor = None
        
        if self.scale_rate != 1:
            if train:
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([self.scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([self.scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
        else:
            if train:
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
        self.transforms = transforms

        # if the training/testing include noisy setting
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']

        if 'train_params' not in params or\
                'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)

                    lane_files = os.path.join(cav_path,
                                              timestamp + '_bev_static.png')
                    semantic_file = os.path.join(cav_path,
                                              timestamp + '_semantic' + '.pcd')
                    # semantic_label_file = os.path.join(cav_path,
                    #                           timestamp + '_semantic_label' + '.json')
                    # semantic_label_file = os.path.join(cav_path,
                    # timestamp + '_semantic_label' + '.json')

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['lane'] = \
                        lane_files
                    self.scenario_database[i][cav_id][timestamp]['semantic'] = \
                        semantic_file
                    # self.scenario_database[i][cav_id][timestamp]['semantic_label'] = \
                    #     semantic_label_file
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        
        (input_data_dict, co_processed_label, 
        box_corner_list, co_labels) = self.get_data_wrap(idx, ego_only=False)
        ego_history_data_dict = None
                
        (_, single_label, _, _) = self.get_data_wrap(idx, ego_only=True)
         
        return input_data_dict, co_processed_label, single_label, box_corner_list, co_labels, ego_history_data_dict
        

    def get_data_wrap(self, idx, ego_only=False):
            base_data_dict = self.get_data_info(idx) # Return a input dict with cav information and input pointcloud as well as images
            
            input_data_dict = {"ego": -1,
                            "cav_ids": []}
            init = 0          
            
            for cav_id, cav_content in base_data_dict.items():
                
                if ego_only:
                    if not cav_content["ego"]:
                        continue

                input_data_dict["cav_ids"].append(cav_id)
                input_data_dict[cav_id] = {"imgs": [],
                                        "img_metas": [],
                                        "pos": cav_content['true_ego_pos']}
    
                cav_lidar_pose = cav_content['lidar_pose']
                
                mean = 0        # Mean of the distribution
                std_dev = self.gps_noise_std     # Standard deviation of the distribution
                
                vector_size = 2
                random_vector = np.random.normal(mean, std_dev, vector_size)
                extended_vec1 = np.zeros(3)
                extended_vec1[:len(random_vector)] = random_vector
                extended_vec1 = extended_vec1.reshape((1,3))
                
                cav_lidar_xyz = cav_content['lidar_xyz']
                if  cav_content["ego"]:
                    ego_lidar_xyz = cav_lidar_xyz
                    ego_lidar_xyz = np.add(ego_lidar_xyz, extended_vec1)
                cav_semantic_xyz = cav_content['semantic_xyz']
                cav_semantic_cls = cav_content['semantic_cls']
                cav_camera_files = cav_content['img_filename']
                
                cav_img_metas = {
                    'lidar2img': cav_content['lidar2img'],
                    }
            
                
                cav_transformation_matrix = x_to_world(cav_lidar_pose)
                if cav_content["ego"]:
                    input_data_dict["ego"] = cav_id    
                    ego_transformation_matrix = cav_transformation_matrix
                    ego_inverse_transformation_matrix = np.linalg.inv(ego_transformation_matrix)
                    ego_lane_files = cav_content['lane_filename']

                ego_boxes3d = []
                if cav_content['ego']:
                    ego_pos = np.array(cav_content["true_ego_pos"][:2])
                    for v_id, v_content in cav_content['vehicles'].items():

                        x = v_content['location'][0]
                        y = v_content['location'][1]
                        if np.abs(np.array(ego_pos)[0]-x) > 20 or np.abs(np.array(ego_pos)[1]-y) > 20 :
                            continue
                        z = v_content['location'][2]
                        dx= v_content['extent'][0]
                        dy= v_content['extent'][1]
                        dz= v_content['extent'][2]
                        heading = v_content['angle'][1]
                        ego_boxes3d.append([x, y , z+dz, 2*dx+1, 2*dy+1, 1*dz, heading*np.pi/180])
                    if len(ego_boxes3d) > 0:
                        ego_box_corner = boxes_to_corners_3d(np.array(ego_boxes3d), order="lwh")
                        box_corner_list = []
                        for box_corner in ego_box_corner:
                            box_corner = np.hstack([box_corner, np.ones([box_corner.shape[0], 1])])                     
                            box_corner = np.dot(ego_inverse_transformation_matrix, box_corner.T).T
                            box_corner = np.delete(box_corner, 3, 1)
                            box_corner_list.append(box_corner)
                    else: 
                        print("Empty object in FoV.")
                        box_corner_list = []

                    
                    
                cav_imgs = []
                for filename in cav_camera_files:
                    input_data_dict[cav_id]["imgs"].append(filename)
                    cav_imgs.append(
                        imread(filename, 'unchanged').astype(np.float32)
                    )

                # deal with img augmentations # Rui: NON-OPV2V
                #------------------------
                cav_imgs_dict = {'img': cav_imgs, 'lidar2img': cav_img_metas['lidar2img']}
                for t in self.transforms:
                    cav_imgs_dict = t(cav_imgs_dict)
                cav_imgs = cav_imgs_dict['img']
                cav_imgs = [img.transpose(2, 0, 1) for img in cav_imgs] # Rui: Changable by CONfig

                cav_img_metas['img_shape'] = cav_imgs_dict['img_shape']
                cav_img_metas['lidar2img'] = cav_imgs_dict['lidar2img']
                
                #------------------------

                #cav_lidar_filled = (cav_semantic_xyz) #lidar_filled shape: (n,4)
                cav_label_filled = cav_semantic_cls#.reshape(-1,1)
                
                #transform from lidar coodination to globale coodination
                cav_ones_column = np.ones((cav_semantic_xyz.shape[0], 1))
                cav_lidar_filled = np.hstack((cav_semantic_xyz, cav_ones_column)) #lidar_filled shape: (n,4)
                cav_PC_globale_filled = np.dot(cav_transformation_matrix, cav_lidar_filled.T).T
                cav_PC_globale = cav_PC_globale_filled[:, :-1]# PC_global shape: (n,3)
                

                    
                if init == 0:
                    co_cav_lidar_filled = cav_PC_globale
                    co_cav_label_filled = cav_label_filled
                    init += 1
                else: 
                    co_cav_lidar_filled = np.concatenate((co_cav_lidar_filled, cav_PC_globale), axis=0)
                    co_cav_label_filled = np.concatenate((co_cav_label_filled, cav_label_filled), axis=0)


                input_data_dict[cav_id]["img_metas"] = cav_img_metas
                

            co_cav_label_filled = co_cav_label_filled
            co_ones_column_sorted = np.ones((co_cav_lidar_filled.shape[0], 1))#tempral patch
            co_cav_lidar_filled = np.hstack((co_cav_lidar_filled, co_ones_column_sorted))
            co_cav_lidar_filled= np.dot(ego_inverse_transformation_matrix, co_cav_lidar_filled.T).T
            co_cav_lidar_filled = co_cav_lidar_filled[:, :-1]
            
            #plot_point_cloud_2(co_cav_lidar_filled, co_cav_label_filled)
            
            max_bound = np.asarray(self.max_volume_space ) 
            min_bound = np.asarray(self.min_volume_space)  
            # get grid index
            crop_range = max_bound - min_bound
            cur_grid_size = self.grid_size 
            intervals = crop_range / (cur_grid_size - 1)   

            if (intervals == 0).any(): 
                print("Zero interval!")

            
            co_labels, co_grid_ind_float = self.point_cut(co_cav_label_filled, 
                                                    co_cav_lidar_filled, 
                                                    min_bound, max_bound, intervals)
                
            co_grid_ind = np.floor(co_grid_ind_float).astype(int)

            co_processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
            co_label_voxel_pair = np.concatenate([co_grid_ind, co_labels.reshape(-1,1)], axis=1)
            co_label_voxel_pair = co_label_voxel_pair[np.lexsort((co_grid_ind[:, 0], co_grid_ind[:, 1], co_grid_ind[:, 2])), :]
            co_processed_label = nb_process_label(np.copy(co_processed_label), co_label_voxel_pair)
            
            # raw Lidar for evaluation
            ego_lidar_label_filled = (np.ones((ego_lidar_xyz.shape[0], 1)) * 23).astype(int)
            ego_lidar_label, ego_lidar_grid_ind_float = self.point_cut(ego_lidar_label_filled, 
                                                    ego_lidar_xyz, 
                                                    min_bound, max_bound, intervals)
            ego_lidar_grid_ind = np.floor(ego_lidar_grid_ind_float).astype(int)

            ego_lidar_processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
            ego_lidar_label_voxel_pair = np.concatenate([ego_lidar_grid_ind, ego_lidar_label.reshape(-1,1)], axis=1)
            ego_lidar_label_voxel_pair = ego_lidar_label_voxel_pair[np.lexsort((ego_lidar_grid_ind[:, 0], 
                                                                ego_lidar_grid_ind[:, 1], 
                                                                ego_lidar_grid_ind[:, 2])), :]
            ego_lidar_processed_label = nb_process_label(np.copy(ego_lidar_processed_label), ego_lidar_label_voxel_pair)

            

            return (input_data_dict, co_processed_label, box_corner_list, ego_lidar_processed_label)


    def point_cut(self, ego_labels, ego_points_lidar_sorted, min_bound, max_bound, intervals):
        """### Remove lidar points outside the field of view.
            Args
            ------
                min_bound: The minimum boundary of the field of view
                max_bound: The maximum boundary of the field of view
                intervals: Intervals of each grids
            Return
            ------
                Croped lidar points

        """
        mask = np.where(ego_points_lidar_sorted[: , 2]> min_bound[2])
        ego_points_lidar_sorted = ego_points_lidar_sorted[mask]
        ego_labels = ego_labels[mask]
        mask = np.where(ego_points_lidar_sorted[: , 1]> min_bound[1])
        ego_points_lidar_sorted = ego_points_lidar_sorted[mask]
        ego_labels = ego_labels[mask]
        mask = np.where(ego_points_lidar_sorted[: , 0]> min_bound[0])
        ego_points_lidar_sorted = ego_points_lidar_sorted[mask]
        ego_labels = ego_labels[mask]
        
        mask = np.where(ego_points_lidar_sorted[: , 2]< max_bound[2])
        ego_points_lidar_sorted = ego_points_lidar_sorted[mask]
        ego_labels = ego_labels[mask]
        mask = np.where(ego_points_lidar_sorted[: , 1]< max_bound[1])
        ego_points_lidar_sorted = ego_points_lidar_sorted[mask]
        ego_labels = ego_labels[mask]
        mask = np.where(ego_points_lidar_sorted[: , 0]< max_bound[0])
        ego_points_lidar_sorted = ego_points_lidar_sorted[mask]
        ego_labels = ego_labels[mask]
    
        ego_grid_ind_float = (ego_points_lidar_sorted - min_bound) / intervals
        return ego_labels, ego_grid_ind_float
                
    def get_box_corners(self, base_data_dict):
        """Get box_corners for ego and cooperation, based on the base_data_dict

        Args:
            base_data_dict (dict): _description_
        """        
        
        def batch_boxes_to_corner(cav_boxes_dict):
            """Get the coordinates of the box vertices

            Args:
                cav_boxes_dict (dict)

            Returns:
                cav_box_corner:(N, 8, 3), the 8 corners of the bounding box.
            """
            N = len(cav_boxes_dict)  

            cav_boxes3d = np.zeros((N, 7))  

            for i, box_info in enumerate(cav_boxes_dict):
                x = box_info['x']
                y = box_info['y']
                z = box_info['z']
                dx = box_info['dx']
                dy = box_info['dy']
                dz = box_info['dz']
                heading = box_info['heading']
                cav_boxes3d[i] = [x, y, z, dx, dy, dz, heading]  #boxes3d shape (N,7)

            cav_box_corner = boxes_to_corners_3d(cav_boxes3d, order='lwh')
            return cav_box_corner
        
        cav_boxes_dict = []
        for cav_id, cav_content in base_data_dict.items():
            curr_cav_boxes=[]
            for v_id, v_content in cav_content['vehicles'].items():
                x = v_content['location'][0]
                y = v_content['location'][1]
                z = v_content['location'][2]
                dx= v_content['extent'][0]
                dy= v_content['extent'][1]
                dz= v_content['extent'][2]
                heading = v_content['angle'][1]
                curr_cav_boxes.append({'x': x, 'y':y , 'z': z+dz, 'dx': 2*dx, 'dy':2*dy, 'dz': 2*dz, 'heading': heading*np.pi/180})
            if cav_content["ego"]:
                ego_box_corner = batch_boxes_to_corner(curr_cav_boxes)
            
            cav_boxes_dict.extend(curr_cav_boxes)   
        cav_box_corner = batch_boxes_to_corner(cav_boxes_dict) 
        
        return cav_box_corner, ego_box_corner
    
    # step 1: Index of datapoint in all scenarios
    def get_data_info(self, idx):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
        """
        # standard protocal modified from SECOND.Pytorch

        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
    
        input_dict = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            input_dict[cav_id] = OrderedDict()
            input_dict[cav_id]['ego'] = cav_content['ego']
            cur_params = load_yaml(cav_content[timestamp_key]['yaml'])
            input_dict[cav_id]['lidar_pose']=cur_params['lidar_pose']
            input_dict[cav_id]['true_ego_pos']=cur_params['true_ego_pos']
            input_dict[cav_id]["pts_filename"] = cav_content[timestamp_key]['lidar']
            input_dict[cav_id]['img_filename'] = cav_content[timestamp_key]['camera0']
            input_dict[cav_id]['lane_filename'] = cav_content[timestamp_key]['lane']
            camera_dict = {}
            for key, value in cur_params.items():
                if key.startswith('camera'):
                    camera_dict[key] = value # RUi: Camera images
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in  camera_dict.items(): 
                lidar2cam_rt = np.linalg.inv(cam_info['extrinsic'])
                intrinsic = np.array(cam_info['intrinsic'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt) 
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            
            input_dict[cav_id]['lidar2img'] = lidar2img_rts
            input_dict[cav_id]['cam_intrinsic'] = cam_intrinsics
            input_dict[cav_id]['lidar2cam'] = lidar2cam_rts

            pcd = o3d.io.read_point_cloud(cav_content[timestamp_key]['lidar'])
            
            pcd_xyz = np.asarray(pcd.points)
            
            
            pcd_semantic= o3d.io.read_point_cloud(cav_content[timestamp_key]['semantic'])
            pcd_semantic_xyz = np.asarray(pcd_semantic.points)
            pcd_semantic_c = np.asarray(pcd_semantic.colors)
            pcd_semantic_c = (pcd_semantic_c[:,2] * 255).astype(int)
            input_dict[cav_id]['semantic_xyz'] = pcd_semantic_xyz # pxd_xyz shape: (n,4)
            input_dict[cav_id]['semantic_cls'] = pcd_semantic_c
            input_dict[cav_id]['lidar_xyz'] = pcd_xyz # pxd_xyz shape: (n,3)
            input_dict[cav_id]['vehicles'] = cur_params['vehicles']


        return input_dict
    
    def load_camera_files(self, cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]    

    def return_timestamp_key(self, scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key    
    
    def extract_timestamps(self, yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

def custom_collate_fn(data):
    
    ego_input2stack = [d[0] for d in data]
    co_label2stack = np.stack([d[1] for d in data]).astype(int)
    ego_processed_label_stack = np.stack([d[2] for d in data]).astype(float)
    co_grid_ind_stack = np.stack([d[3] for d in data]).astype(float)
    co_point_label = np.stack([d[4] for d in data]).astype(int)
    ego_truncated_boxes_corner3d_stack = [d[5] for d in data]
    
    return ego_input2stack, \
    torch.from_numpy(co_label2stack),\
    torch.from_numpy(ego_processed_label_stack),\
    torch.from_numpy(co_grid_ind_stack),\
    torch.from_numpy(co_point_label),\
    ego_truncated_boxes_corner3d_stack
 

def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1

    if np.any(np.concatenate((counter[:2], counter[2:] != 0))):
         counter[2] = 0
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def plot_box(box_corner):
    """plot the bounding box of vehicles

    Args:
        box_corner (np.array): (N, 8, 3), the 8 corners of the bounding box.
    """    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for box in box_corner:
        x = box[:, 0]
        y = box[:, 1]
        z = box[:, 2]

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  
            [4, 5], [5, 6], [6, 7], [7, 4], 
            [0, 4], [1, 5], [2, 6], [3, 7]   
        ]
        for edge in edges:
            ax.plot(x[edge], y[edge], z[edge], color='b')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()        

def plot_point_and_box(points, boxes):
    """###plot lidar points and bounding box of vehicles
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x, y, z, c='b', marker='o',s=5)

    for box in boxes:
    
        x = box[:, 0]
        y = box[:, 1]
        z = box[:, 2]

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  
            [4, 5], [5, 6], [6, 7], [7, 4],  
            [0, 4], [1, 5], [2, 6], [3, 7]   
        ]
        for edge in edges:
            ax.plot(x[edge], y[edge], z[edge], color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([200, 200, 16])

    plt.show()


def plot_vox(voxels, targets, ref =1, fig_name="test", s_idx=-1):
    """Plot 3D voxel scene

    Args:
        voxels (np.array or Tensor): Input voxel data
        targets (np.array or Tensor): Occupancy ground truth, only used for plot semantic result, otherwise should be same as the input voxel data
        ref (int, optional): Defaults to 1.
        fig_name (str, optional): _Saved file name.
        s_idx (int, optional): . Defaults to -1.

    """    

    if torch.is_tensor(voxels):
        voxels = voxels.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
        
    if ref == 0:
        with open('visual_result.npy', 'wb') as f:
            np.save(f, voxels)
            np.save(f, targets)
    ax = plt.figure().add_subplot(projection='3d')

    colors_ref = ['gray', 'red', 'brown', 'pink', 'purple', 
                  'cyan', 'magenta', 'green', 'blue', 'black', 
                  'navy', 'olive', 'maroon', 'teal']
    
    mask = (targets!=0) 
    colors = np.empty(voxels.shape, dtype=object)

    for i in range(len(colors_ref)):
        def rgb_to_hex(rgb):
            return '%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
        colors[voxels==i] = colors_ref[i]
    ax.voxels(mask, facecolors=colors)
    ax.grid(False)
    ax.set_box_aspect([200, 200, 16])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=25, azim=-45, roll=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent pane for X-axis
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent pane for Y-axis
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent pane for Y-axis
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent color for X-axis line
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent color for Y-axis line
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent color for Y-axis line


    plt.savefig(fig_name+".png", dpi=600)


def get_opv2v_label_name(label_mapping):
    """Load label mapping from yaml file

    Args:
        label_mapping (yaml): Class idex and its corresponding class label name

    Returns:
        opv2v_label_name (dict): Class idex and its corresponding class label name
    """    
    with open(label_mapping, 'r') as stream:
        opv2vyaml = yaml.safe_load(stream)
    opv2v_label_name = dict(opv2vyaml['opv2v_label_name'])

    return opv2v_label_name

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


