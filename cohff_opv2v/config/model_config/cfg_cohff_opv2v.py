
from config.model_config.cfg_com_net import model as completion_net
from config.model_config.cfg_seg_net import model as segmentation_net
from config.load_yaml import task_config


_base_ = [
    '../train_config/opv2v_dataset.py',
    '../train_config/optimizer.py',
]

task_type = task_config["task_type"]
model_type=task_config["model_type"]

vox_range = task_config["vox_range"]#
# optional: 51.2, 51.2, 3, -51.2, -51.2, -5

_dim_ = 128
_time_dim_ = _dim_
num_heads = 8
_pos_dim_ = [48, 48, 32]
#_pos_dim_ = [96, 96, 64]
#_pos_dim_ = [192, 192, 128]
_ffn_dim_ = _dim_*3
_num_levels_ = 4
_num_cams_ = 4
max_connect_cav=task_config["max_connect_cav"]

h_ = task_config["h"]
w_ = task_config["w"]
z_ = task_config["z"]
encoder_layers = 5
decoder_layer = 1
time_decoder_layer=2

grid_size = [h_, w_, z_]

nbr_class = task_config["nbr_class"]


if task_type == 0:
    model=completion_net

if task_type == 1:
    model=segmentation_net

if task_type == 2:
    if model_type == 0:
        hybrid_type_="HybridFusionModel"
        task_fusion_in_dims = 18
    elif model_type == 1:
        hybrid_type_ = "CoopHybridFusionModel"
        task_fusion_in_dims = (max_connect_cav+1)*_dim_
    elif model_type == 2: 
        hybrid_type_ = "CoopHybridFusionModel"
        task_fusion_in_dims = (max_connect_cav+2)*_dim_
    else:
        print("unknown")
        exit()

    model = dict(
        type=hybrid_type_,
        segmentation_net=segmentation_net,
        completion_net=completion_net,
        segmentation_net_path=task_config["segmentation_net_path"],
        completion_net_path=task_config["completion_net_path"],
        max_connect_cav=max_connect_cav,
        task_fusion_model = dict(
            type='TaskFusionConvModel',
            nbr_classes=18, 
            in_dims=task_fusion_in_dims, 
            hidden_dims=1024,
            out_dims=None
        ),
        co_pos_encoder=dict(
            p_h=h_,
            p_w=w_,
            p_z=z_,
            embed_dims = _dim_,
            relative_coord_input_dim = 2,
            relative_angle_input_dim = 1
        )
    )

