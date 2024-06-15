from config.load_yaml import task_config

h=task_config["h"]
w=task_config["w"]
z=task_config["z"]
vox_range = task_config["vox_range"]
max_volume_space=vox_range[:3]
min_volume_space=vox_range[3:]

dataset_params = dict(
    ignore_label = -99,
    label_mapping = "./config/label_config/opv2v_label_name.yaml"
)

train_data_loader = dict(
    root_dir = task_config["train_dataset_dir"],
    fill_label = 0,
    grid_size = [h, w, z],
    max_volume_space = max_volume_space,
    min_volume_space = min_volume_space,
    scale_rate = 1,
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)

val_data_loader = dict(
    validate_dir = task_config["val_dataset_dir"],
    fill_label = 0,
    grid_size = [h, w, z],
    max_volume_space = max_volume_space,
    min_volume_space = min_volume_space,
    scale_rate = 1,
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



max_epochs = task_config["max_epochs"]
gradient_accumulate_steps = task_config["gradient_accumulate_steps"]