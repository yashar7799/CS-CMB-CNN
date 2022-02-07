# from .unet_pretrain import unet_pretrain_args

ARGUMENTS = dict(
    # unet_pretrain=unet_pretrain_args
)

def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()