Classification of Cosmic Strings on the CMB Map Using CNNs‬‬ (CNNs)
=====


This repository is a complete model training loop (implemented in TensorFlow) to train multi-class models to do a classification task to recognize the existence of the ‫‪Cosmic‬‬ ‫‪String tension (g_mu) in a given patch of the skymap.

Note: This project is done as my Bachelor's project under the supervision of [Dr. S.M.S Movahed](https://scholar.google.com/citations?user=uhy9JDAAAAAJ&hl=en); this project is a reproduced version of [this paper](https://www.researchgate.net/profile/Motahare-Torki/publication/352054373_Planck_Limits_on_Cosmic_String_Tension_Using_Machine_Learning/links/60c06cdb458515bfdb556da4/Planck-Limits-on-Cosmic-String-Tension-Using-Machine-Learning.pdf) (deep learning part); so if you want to know more about the problem and theoretical aspects, I highly suggest you check out the mentioned paper.
-----
Here, classes are different values of g_mu that you gave to the data creation part of this repo to make the classes.

To use this repo you should first clone it and do the following two steps:
1. data creation
2. training

follow [this](https://github.com/yashar7799/CS-CMB-CNN/blob/master/CS_CMB_CNN_notebook.ipynb) notebook to know how to go through these steps and use the repo.

## parameters
you can add these parameters to the train.py code as below:  
* parameters inside a [] are optional.
<pre>

usage: train.py [-h] --model MODEL [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                --n_classes N_CLASSES [--dropout_rate DROPOUT_RATE]
                [--input_shape INPUT_SHAPE] [--pretrain]
                [--path_to_pretrain PATH_TO_PRETRAIN]
                [--weights_base_folder WEIGHTS_BASE_FOLDER] --dataset_dir
                DATASET_DIR [--mlflow_source MLFLOW_SOURCE] [--run_ngrok]
                [--no_run_ngrok] [--ngrok_auth_token NGROK_AUTH_TOKEN]
                [--augmentation] [--augmentation_p AUGMENTATION_P]
                [--multiprocessing] [--no_multiprocessing] [--workers WORKERS]
                [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                [--learning_rate LEARNING_RATE]
                [--plateau_reduce_lr_scheduler]
                [--plateau_reduce_initial_lr PLATEAU_REDUCE_INITIAL_LR]
                [--plateau_reduce_factor PLATEAU_REDUCE_FACTOR]
                [--plateau_reduce_min_lr PLATEAU_REDUCE_MIN_LR]
                [--plateau_reduce_patience PLATEAU_REDUCE_PATIENCE]
                [--warmup_lr_scheduler] [--warmup_max_lr WARMUP_MAX_LR]
                [--warmup_epoch WARMUP_EPOCH] [--cosine_decay_lr_scheduler]
                [--cosine_decay_initial_lr COSINE_DECAY_INITIAL_LR]
                [--cosine_decay_alpha COSINE_DECAY_ALPHA]
                [--tb_log_dir TB_LOG_DIR]

description of each parameter:

  -h, --help            show this help message and exit
  --model MODEL         model name.
  --epochs EPOCHS       define number of training epochs
  --batch_size BATCH_SIZE
                        define batch size
  --n_classes N_CLASSES
                        number of classes; this should be same as the number
                        of classes of the dataset you are using
  --dropout_rate DROPOUT_RATE
                        define dropout rate to use between fc layers
  --input_shape INPUT_SHAPE
                        desired input shape to feed the model with
  --pretrain            pass this arg if you want to load weights of a
                        pretrained model.
  --path_to_pretrain PATH_TO_PRETRAIN
                        path of pretrain .h5 weights file
  --weights_base_folder WEIGHTS_BASE_FOLDER
                        this is the base folder that all the weights will be
                        saved there
  --dataset_dir DATASET_DIR
                        dataset directory, this directory should contain
                        train, val & test folders
  --mlflow_source MLFLOW_SOURCE
                        The mlflow direcotry
  --run_ngrok           pass this arg if you want to run train.py in colab!
  --no_run_ngrok        pass this arg if you want to run train.py locally!
  --ngrok_auth_token NGROK_AUTH_TOKEN
                        an authentication token that ngrok gives it to you
  --augmentation        pass this arg if you want augmentations
  --augmentation_p AUGMENTATION_P
                        augmentation probability
  --multiprocessing     Run model.fit with multi-processing
  --no_multiprocessing  Run model.fit without multi-processing
  --workers WORKERS     number of workers for model.fit
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        early stopping patience
  --learning_rate LEARNING_RATE
                        learning rate, use this if you dont use any
                        lr_scheduler
  --plateau_reduce_lr_scheduler
                        use plateau_reduce_lr_scheduler
  --plateau_reduce_initial_lr PLATEAU_REDUCE_INITIAL_LR
                        initilal learning rate for plateau_reduce_lr_scheduler
  --plateau_reduce_factor PLATEAU_REDUCE_FACTOR
                        factor by which the learning rate will be reduced;
                        new_lr = previous_lr * factor
  --plateau_reduce_min_lr PLATEAU_REDUCE_MIN_LR
                        lower bound on the learning rate for
                        plateau_reduce_lr_scheduler
  --plateau_reduce_patience PLATEAU_REDUCE_PATIENCE
                        number of epochs with no improvement after which
                        learning rate will be reduced.
  --warmup_lr_scheduler
                        use warmup_lr_scheduler
  --warmup_max_lr WARMUP_MAX_LR
                        maximum lr that warmup_lr_scheduler will reach to
  --warmup_epoch WARMUP_EPOCH
                        number of epoch to increase the lr to warmup_max_lr
  --cosine_decay_lr_scheduler
                        use cosine_decay_lr_scheduler
  --cosine_decay_initial_lr COSINE_DECAY_INITIAL_LR
                        cosine_decay_lr_scheduler initial learning_rate
  --cosine_decay_alpha COSINE_DECAY_ALPHA
                        minimum learning_rate value as a fraction of
                        cosine_decay_initial_lr.
  --tb_log_dir TB_LOG_DIR
                        The TensorBoard directory
<pre>
