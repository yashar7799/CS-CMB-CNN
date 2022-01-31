"""
This module load user selected model.

this modules contain following Functions:
    -   load_model
"""

from .densenet import UNet as Unet
from .unet_pretrain import UNet as UnetPR
from .unet_batch_normalization import UNet as UnetBN
from .deeplabv3plus_resnet import DeepLabV3Plus as deeplabv3plus_resnet

MODELS = dict(
    densenet=densenet,
    efficientnet=efficientnet,
    inception=inception,
    mobilenet=mobilenet,
    nasnet=nasnet,
    resnet=resnet,
    vgg=vgg,
    xception=xception
)


def load_model(model_name, **kwargs):
    """ Get model
       Parameters
       ----------
       model_name  string, user selected model name

       Returns
       -------
       model     tensorflow model instance
       """
    return MODELS[model_name](**kwargs).get_model()
