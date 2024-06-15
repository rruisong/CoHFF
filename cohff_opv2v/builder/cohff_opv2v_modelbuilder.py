from hybrid_fusion_model import *
from mmseg.models import build_segmentor

def build(model_config):
    """Build model

    Args:
        model_config (dict): Model config

    Returns:
        model
    """    
    model = build_segmentor(model_config)
    return model
