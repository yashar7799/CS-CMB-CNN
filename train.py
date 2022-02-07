"""
This is the main module of this project ; here we difine a function to start training process.
"""

import sys
import tensorflow as tf
from utils.metrics import *
from datetime import datetime
from os.path import join
from models import load_model
# from params import get_args
from params.main import main_args
from data_handler.data_loader import DataGenerator
from data_handler.data_creator import DataCreator
from tensorflow.keras.optimizers import Adam
from utils.callbacks import get_callbacks
from utils.mlflow_handler import MLFlowHandler
from utils.plots import get_plots
from utils.utils import get_gpu_grower

get_gpu_grower()


def train():
    """
    this function is to start trainning process of the desired model.
    it should run in below format:
        python train.py -[option] [value] --[option] [value] ...
    to see available options and arguments see Readme.md file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = main_args()
    print(f"Arguments: {args}")

    id_ = model_name + "_" + \
        str(datetime.now().date()) + "_" + str(datetime.now().time())
    weight_path = join('weights', id_) + ".h5"
    mlflow_handler = MLFlowHandler(model_name=model_name,
                                   warmup=args.warmup_lr_scheduler,
                                   run_name=id_,
                                   mlflow_source=args.mlflow_source,
                                   run_ngrok=args.run_ngrok)
    mlflow_handler.start_run(args)

    data = DataCreator()
    partition, labels = data.partitioning(partitioning_base_folder= args.dataset_dir)

    train_loader = DataGenerator(partition['train'], labels=labels, batch_size=args.batch_size, dim=(args.input_shape[0], args.input_shape[1]), n_channels=args.input_shape[2], n_classes=args.n_classes)
    val_loader = DataGenerator(partition['val'], labels=labels, batch_size=args.batch_size, dim=(args.input_shape[0], args.input_shape[1]), n_channels=args.input_shape[2], n_classes=args.n_classes)
    test_loader = DataGenerator(partition['test'], labels=labels, batch_size=args.batch_size, dim=(args.input_shape[0], args.input_shape[1]), n_channels=args.input_shape[2], n_classes=args.n_classes)

    model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, dropout=args.dropout_rate)

    print("Loading Model is Done!")

    if args.pretrain:
        model.load_weights(args.path_to_pretrain)
        print('pretrain weights loaded.')
    model.summary()

    checkpoint, warmup, early_stopping, tensorboard, reduce_lr = get_callbacks(model_path=weight_path,
                                                                               early_stopping_p=args.early_stopping_p,
                                                                               tb_log_dir=args.tb_dir,
                                                                               epochs=args.epochs,
                                                                               sample_count=len(train_loader) * args.batch_size,
                                                                               batch_size=args.batch_size,
                                                                               warmup_epoch=args.warmup_epoch,
                                                                               warmup_max_lr=args.warmup_max_lr,
                                                                               model_name=model_name,
                                                                               plateau_min_lr=args.plateau_min_lr,
                                                                               plateau_reduce_patience=args.plateau_reduce_patience)
    callbacks = [tensorboard, checkpoint, early_stopping, mlflow_handler.mlflow_logger]
    if args.warmup_lr_scheduler:
        callbacks.append(warmup)
        print('activate warmup_lr_scheduler.')
    elif args.cosine_decay_lr_scheduler:
        learning_rate_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.cosine_decay_initial_learning_rate, decay_steps=args.epochs, alpha=args.cosine_decay_alpha, name='None')
        opt = Adam(learning_rate=learning_rate_scheduler)
        print('activate cosine_decay_lr_scheduler.')
    elif args.reduce_lr_scheduler:
        callbacks.append(reduce_lr)
        opt = Adam(learning_rate=args.learning_rate)
        print('activate reduce_lr_scheduler.')
    else:
        opt = Adam(learning_rate=args.learning_rate)
        print(f'No lr_scheduler, learning rate fixed at: {args.learning_rate}')
    loss = bce_dice_loss
    model.compile(optimizer=opt, loss=loss, metrics=['acc', dice_coef])
    model.fit(x=train_loader,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=val_loader,
              validation_batch_size=args.val_batch_size,
              callbacks=callbacks,
              use_multiprocessing=args.multiprocessing,
              workers=args.workers,
              )

    print("Training Model is Done!")

    get_plots(model, val_loader, mlflow_handler)
    mlflow_handler.end_run(weight_path)


if __name__ == '__main__':
    train()
