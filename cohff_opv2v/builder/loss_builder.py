import torch

def build(ignore_label=0):
    """Build loss function

    Args:
        ignore_label (int, optional): Classes to ignore when calculating loss. Defaults to 0.

    Returns:
        ce_loss_func: cross entropy loss function
    """    
    ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
    return ce_loss_func
