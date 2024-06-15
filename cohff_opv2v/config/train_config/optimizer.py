from config.load_yaml import task_config as task_config_optimizer

optimizer = dict(
    type=task_config_optimizer["optimizer_type"],
    lr=task_config_optimizer["lr"],
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=task_config_optimizer["weight_decay"]
)

grad_max_norm = task_config_optimizer["grad_max_norm"]

