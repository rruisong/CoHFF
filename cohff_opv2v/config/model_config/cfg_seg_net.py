from config.load_yaml import task_config

vox_range = task_config["vox_range"]#
#51.2, 51.2, 3, -51.2, -51.2, -5

_dim_ = 128
_time_dim_ = _dim_
num_heads = 8
_pos_dim_ = [48, 48, 32]
_ffn_dim_ = _dim_*3
_num_levels_ = 4
_num_cams_ = 4

h_ = task_config["h"] #
w_ = task_config["w"] #
z_ = task_config["z"] #
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [h_*scale_h, w_*scale_w, z_*scale_z]
encoder_layers = 5
decoder_layer = 1
time_decoder_layer=2
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
num_anchors = 16
attn_points = 32
init = 0

nbr_class = task_config["nbr_class"] #

_cav_aggregator_= True

time_self_layer=dict(
    type = 'CoHFF_Time_Decoder_Layer',
    attn_cfgs=[
        dict(
            type='CoHFF_Time_Cross_Attention',
            p_h=h_,
            p_w=w_,
            p_z=z_,
            num_anchors=num_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=attn_points,
            init_mode=init,
            cav_aggregator=_cav_aggregator_
            )
        ],
        feedforward_channels=_ffn_dim_,
        ffn_dropout=0.1,
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
        cav_aggregator=_cav_aggregator_                
)
#'self_attn', 'norm', 

self_cross_layer = dict(
    type='CoHFF_Encoder_Layer',
    attn_cfgs=[
        dict(
            type='CoHFF_CrossView_Self_Attention',
            p_h=h_,
            p_w=w_,
            p_z=z_,
            num_anchors=num_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=attn_points,
            init_mode=init,
        ),
        dict(
            type='ImageCrossAttention',
            pc_range=vox_range,
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='CoHFF_MSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                p_h=h_,
                p_w=w_,
                p_z=z_,
            ),
            embed_dims=_dim_,
            p_h=h_,
            p_w=w_,
            p_z=z_,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)

self_layer = dict(
    type='CoHFF_Encoder_Layer',
    attn_cfgs=[
        dict(
            type='CoHFF_CrossView_Self_Attention',
            p_h=h_,
            p_w=w_,
            p_z=z_,
            num_anchors=num_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=attn_points,
            init_mode=init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)

co_self_layer=dict(
    type = 'CoHFF_Decoder_Layer',
    attn_cfgs=[
        dict(
            type='CoHFF_Decoder_CrossView_Self_Attention',
            p_h=h_,
            p_w=w_,
            p_z=z_,
            num_anchors=num_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=attn_points,
            init_mode=init,
            cav_aggregator=_cav_aggregator_
            )
        ],
        feedforward_channels=_ffn_dim_,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'ffn', 'norm'),
        cav_aggregator=_cav_aggregator_                
)


model = dict(
    type='SegFormer',
    use_grid_mask=True,
    predictionhead = dict(
        type='CoHFF_Predictionhead',
        p_h=h_,
        p_w=w_,
        p_z=z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    img_backbone=dict(
        type='ResNet',
        depth=101,# 101 layers
        num_stages=4, 
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    encoder_head=dict(
        type='CoHFF_Encoder_Head',
        p_h=h_,
        p_w=w_,
        p_z=z_,
        pc_range=vox_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        encoder=dict(
            type='CoHFF_Encoder',
            p_h=h_,
            p_w=w_,
            p_z=z_,
            num_layers=encoder_layers,
            pc_range=vox_range,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer,
                self_cross_layer,
                self_cross_layer,
                self_layer,
                self_layer,
            ]),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=h_,
            w=w_,
            z=z_
        )),
    co_decoder_head=dict(
        type = 'CoHFF_Decoder_Head',
        p_h=h_,
        p_w=w_,
        p_z=z_,
        embed_dims=_dim_,
        cav_aggregator=_cav_aggregator_,
        co_decoder=dict(
            type='CoHFF_Decoder',
            p_h=h_,
            p_w=w_,
            p_z=z_,
            num_layers=decoder_layer,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                co_self_layer
            ]),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=h_,
            w=w_,
            z=z_
        )),
    co_pos_encoder=dict(
        p_h=h_,
        p_w=w_,
        p_z=z_,
        embed_dims = _dim_,
        relative_coord_input_dim = 2,
        relative_angle_input_dim = 1
        )
)