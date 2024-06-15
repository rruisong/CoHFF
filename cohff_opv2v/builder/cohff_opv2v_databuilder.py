import torch
from opv2v_dataset.opv2v_dataset import CO_ImagePoint_OPV2V_V2_LANE, custom_collate_fn

def build (train_dataloader_config,
          val_dataloader_config,
          dist=False,
          gps_noise_std = 0
    ):
    """Build dataloader

    Args:
        train_dataloader_config (dict): Train dataloder config
        val_dataloader_config (dict): Validation dataloader config
        dist (bool, optional): Whether to use distributed training. Defaults to False.
        gps_noise_std (int, optional): Standard deviation of GPS noise. Defaults to 0.

    Returns:
        train_dataset_loader: Dataloader of train set
        val_dataset_loader: Dataloader of validation set
    """    

    train_dataset = CO_ImagePoint_OPV2V_V2_LANE(params=train_dataloader_config, train=True, gps_noise_std = gps_noise_std)

    val_dataset = CO_ImagePoint_OPV2V_V2_LANE(params=val_dataloader_config,train=False, gps_noise_std = gps_noise_std)
    
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None
        
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=custom_collate_fn,
                                                       shuffle=False if dist else train_dataloader_config["shuffle"],
                                                       sampler=sampler,
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=custom_collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=val_dataloader_config["num_workers"])                                                           

    return train_dataset_loader, val_dataset_loader