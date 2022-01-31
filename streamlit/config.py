"""
This file contains two dictionary to add models' weights path
    and also models' arguments like input_shape.
"""

MODELS_TO_ADDR = { 
    "unet": '../weights/unet/v1/weights.h5'
}

MODELS_TO_ARGS = {
    "unet": {
        'input_shape': (None, None, 3)
    },
}