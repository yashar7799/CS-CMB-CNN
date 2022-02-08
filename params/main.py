"""
Set main parameters before execution
"""

from argparse import ArgumentParser


def main_args():
    """
    Parameters
    ----------
    None

    Returns
    -------
    Returns some hyper-parameters which are common among all the models.
    """
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='model1', help='model name.', required=True)
    parser.add_argument('--epochs', type=int, default=5, help='define number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='define batch size')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes; this should be same as the number of classes of the dataset you are using')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='define dropout rate to use between fc layers')

    parser.add_argument('--input_shape', type=tuple, default=(256, 256, 1), help='desired input shape to feed the model with')

    parser.add_argument('--pretrain', dest='pretrain', action='store_true', help='pass this arg if you want to load weights of a pretrained model.')
    parser.add_argument('--path_to_pretrain', type=str, default=None, help='path of pretrain .h5 weights file')
    
    parser.add_argument('--weights_base_folder', type=str, default='./weights', help='this is the base folder that all the weights will be saved there')

    parser.add_argument('--dataset_dir', type=str, default='./dataset/CS-CMB-CNN-data', help='dataset directory, this directory should contain train, val & test folders', required=True)

    parser.add_argument('--mlflow-source', type=str, default='./mlruns', help='The mlflow direcotry')
    parser.add_argument('--run-ngrok', dest='run_ngrok', action='store_true', help="Run ngrok for colab!")
    parser.add_argument('--no-run-ngrok', dest='run_ngrok', action='store_false', help="Don't run ngrok for colab!")

    parser.add_argument('--augmentation', dest='augmentation', action='store_true', help='pass this arg if you want augmentations')
    parser.add_argument('--augmentation_p', type=float, default=0.5, help='augmentation probability')

    parser.add_argument('--multiprocessing', dest='multiprocessing', action='store_true', help="Run model.fit with multi-processing")
    parser.add_argument('--no-multiprocessing', dest='multiprocessing', action='store_false', help="Run model.fit without multi-processing")
    parser.add_argument('--workers', type=int, default=1, help="number of workers for model.fit")

    parser.add_argument('--early_stopping_patience', type=int, default=6, help='early stopping patience')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, use this if you dont use any lr_scheduler')

    parser.add_argument('--plateau_reduce_lr_scheduler', dest='reduce_lr_scheduler', action='store_true', help="use plateau_reduce_lr_scheduler")
    parser.add_argument('--plateau_reduce_initial_lr', type=float, default=0.001, help='initilal learning rate for plateau_reduce_lr_scheduler')
    parser.add_argument('--plateau_reduce_min_lr', type=float, default=0.0001, help='lower bound on the learning rate for plateau_reduce_lr_scheduler')
    parser.add_argument('--plateau_reduce_patience', type=int, default=4, help='number of epochs with no improvement after which learning rate will be reduced.')

    parser.add_argument('--warmup_lr_scheduler', dest='warmup_lr_scheduler', action='store_true', help="use warmup_lr_scheduler")
    parser.add_argument('--warmup_max_lr', type=float, default=0.1, help='maximum lr that warmup_lr_scheduler will reach to')
    parser.add_argument('--warmup_epoch', type=int, default=3, help='number of epoch to increase the lr to warmup_max_lr')

    parser.add_argument('--cosine_decay_lr_scheduler', dest='cosine_decay_lr_scheduler', action='store_true', help="use cosine_decay_lr_scheduler")
    parser.add_argument('--cosine_decay_initial_lr', type=float, default=0.1, help='cosine_decay_lr_scheduler initial learning_rate')
    parser.add_argument('--cosine_decay_alpha', type=float, default=0.001, help='minimum learning_rate value as a fraction of cosine_decay_initial_lr.')

    parser.add_argument('--tb_log_dir', type=str, default='./logs', help='The TensorBoard directory')

    parser.set_defaults(multiprocessing=True, warmup_lr_scheduler=False, reduce_lr_scheduler=False, cosine_decay_lr_scheduler=False)

    return parser
