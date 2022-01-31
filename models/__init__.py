"""
This module load user selected model.

this modules contain following Functions:
    -   load_model
"""

from .mobilenet import MobileNetV2
from .efficientnet import EfficientNetB0
from .resnet import ResNet18, ResNet50, ResNet50V2
from .densenet import DenseNet121

MODELS = dict(
    densenet=DenseNet121,
    efficientnet=EfficientNetB0,
    # inception=inception,
    mobilenet=MobileNetV2,
    # nasnet=nasnet,
    resnet18=ResNet18,
    resnet50=ResNet50,
    resnet50v2=ResNet50V2
    # vgg=vgg,
    # xception=xception
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
