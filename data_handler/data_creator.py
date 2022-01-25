from ast import Str
import numpy as np
import healpy as hp
import ccgpack as ccg
import os
import shutil
from glob import glob
from PIL import Image
import cv2


class DataCreator():

    """
    Labels
    ------
    0
    1e-5
    5e-6
    1e-6
    5e-7
    1e-7
    5e-8
    1e-8
    5e-9
    1e-9   
    """

    def __init__(self, base_folder='../dataset', write_base_folder='../dataset'):
        self.base_folder = base_folder
        self.write_base_folder = write_base_folder
    def download(self):
        
        # download string maps:
        os.system(f'gdown --id 15iwucSA5yqqEf-QtdcNJt2Dviv2L9fny -c -O {self.base_folder}/map1n_allz_rtaapixlw_2048_1.fits')
        os.system(f'gdown --id 1DNaOMEk1zzX_SzEn7Q6YqF91R3W31jRo -c -O {self.base_folder}/map1n_allz_rtaapixlw_2048_2.fits')
        os.system(f'gdown --id 1kL3-MsBGlOiWw4XdUGYrocISh8gejvR0 -c -O {self.base_folder}/map1n_allz_rtaapixlw_2048_3.fits')

        # download gaussian maps:
        os.system(f'gdown --id 1atZ0F99rzmsKt3NdnCiOm17mc9B6U7qT -c -O {self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0000.fits')
        os.system(f'gdown --id 1VWiXbsYh6M8HkuhorZuonQdXaDW6l3zl -c -O {self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0001.fits')
        os.system(f'gdown --id 1XzMrbXe6hWJVZ0paMXm1UYy8fTj7q162 -c -O {self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0002.fits')
        os.system(f'gdown --id 1Fb7Yj4Pok-k8mSnH2b9pdS07gHofcwxB -c -O {self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0003.fits')
        os.system(f'gdown --id 1KFIGKLee-OBrG7t5Gwk_uuTE0RegdsQ4 -c -O {self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits')

    def partitioning(self, val_ratio=0.15):

        for folder in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:
            os.makedirs(os.path.join(self.write_base_folder, str(folder)), exist_ok=True)

        # train data:
        for i in [1, 2]:
            for j in [0, 1, 2, 3]:
                for g_mu in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:

                    string_map = hp.read_map(f'{self.base_folder}/map1n_allz_rtaapixlw_2048_{i}.fits', nest=1)
                    gaussian_map = hp.read_map(f'{self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_000{j}.fits', nest=1)

                    cmb_with_string = gaussian_map + g_mu*string_map

                    cmb_with_string_patchs = ccg.sky2patch(cmb_with_string, 8)

                    for k in range(768):
                        if not os.path.isfile(f'{self.write_base_folder}/{g_mu}/{k}_{i}_{j}_{g_mu}.png'):
                            array = cmb_with_string_patchs[k]
                            array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                            image = Image.fromarray(array)
                            image.save(f'{self.write_base_folder}/{g_mu}/{k}_{i}_{j}_{g_mu}.png')


        for folder in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:
            os.makedirs(os.path.join(self.write_base_folder, 'test', str(folder)), exist_ok=True)

        # test data:
        for g_mu in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:

            string_map = hp.read_map(f'{self.base_folder}/map1n_allz_rtaapixlw_2048_3.fits', nest=1)
            gaussian_map = hp.read_map(f'{self.base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits', nest=1)

            cmb_with_string = gaussian_map + g_mu*string_map

            cmb_with_string_patchs = ccg.sky2patch(cmb_with_string, 8)

            for i in range(768):
                if not os.path.isfile(f'{self.write_base_folder}/test/{g_mu}/{i}_3_4_{g_mu}.png'):
                    array = cmb_with_string_patchs[i]
                    array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                    image = Image.fromarray(array)
                    image.save(f'{self.write_base_folder}/test/{g_mu}/{i}_3_4_{g_mu}.png')


        folders = ['0', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '5e-8', '1e-8', '5e-9', '1e-9']
        partition = {'train':[], 'val':[], 'test':[]}
        labels = {}

        for folder in folders:

            train_files, val_files = train_val_spliter(self.write_base_folder, folder, val_ratio)

            for train in train_files:
                partition['train'].append(train)
                if folder == '0':
                    labels[train] = '0'
                elif folder == '1e-5':
                    labels[train] = '1'
                elif folder == '5e-6':
                    labels[train] = '2'
                elif folder == '1e-6':
                    labels[train] = '3'
                elif folder == '5e-7':
                    labels[train] = '4'
                elif folder == '1e-7':
                    labels[train] = '5'
                elif folder == '5e-8':
                    labels[train] = '6'
                elif folder == '1e-8':
                    labels[train] = '7'
                elif folder == '5e-9':
                    labels[train] = '8'
                else :
                    labels[train] = '9'


            for val in val_files:
                partition['val'].append(val)
                if folder == '0':
                    labels[val] = '0'
                elif folder == '1e-5':
                    labels[val] = '1'
                elif folder == '5e-6':
                    labels[val] = '2'
                elif folder == '1e-6':
                    labels[val] = '3'
                elif folder == '5e-7':
                    labels[val] = '4'
                elif folder == '1e-7':
                    labels[val] = '5'
                elif folder == '5e-8':
                    labels[val] = '6'
                elif folder == '1e-8':
                    labels[val] = '7'
                elif folder == '5e-9':
                    labels[val] = '8'
                else :
                    labels[val] = '9'

            test_dirs = np.array(glob(os.path.join(self.write_base_folder, 'test', folder, '*')))

            test_folder = os.path.join(self.write_base_folder, folder, 'test')

            os.makedirs(test_folder, exist_ok=True)

            for test_dir in test_dirs:
                shutil.move(test_dir, test_folder)

            test_files = [f'{self.write_base_folder}/{folder}/test/{name}' for name in os.listdir(test_folder)]

            for test in test_files:
                partition['test'].append(test)
                if folder == '0':
                    labels[test] = '0'
                elif folder == '1e-5':
                    labels[test] = '1'
                elif folder == '5e-6':
                    labels[test] = '2'
                elif folder == '1e-6':
                    labels[test] = '3'
                elif folder == '5e-7':
                    labels[test] = '4'
                elif folder == '1e-7':
                    labels[test] = '5'
                elif folder == '5e-8':
                    labels[test] = '6'
                elif folder == '1e-8':
                    labels[test] = '7'
                elif folder == '5e-9':
                    labels[test] = '8'
                else :
                    labels[test] = '9'
        
        # print out train/val/test counts:

        print('Classes and train/val/test counts:\n')
        
        for label in ['0', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '5e-8', '1e-8', '5e-9', '1e-9']:

            n_train = len(os.listdir(os.path.join(self.write_base_folder, f'{label}','train')))
            n_val = len(os.listdir(os.path.join(self.write_base_folder, f'{label}','val')))
            n_test = len(os.listdir(os.path.join(self.write_base_folder, f'{label}','test')))

            print(f'{label} >>> train: {n_train} | val: {n_val} | test: {n_test}')

        return partition, labels










def train_val_spliter(base_folder, folder, val_ratio):
    dirs = np.array(glob(os.path.join(base_folder, folder, '*')))

    train_folder = os.path.join(base_folder, folder, 'train')
    val_folder = os.path.join(base_folder, folder, 'val')

    os.mkdir(train_folder)
    os.mkdir(val_folder)
    
    np.random.shuffle(dirs)
    train_dirs, val_dirs = np.split(dirs, [int(len(dirs)* (1 - val_ratio))])
    for train_dir in train_dirs:
        shutil.move(train_dir, train_folder)
    for val_dir in val_dirs:
        shutil.move(val_dir, val_folder)

    train_files = [f'{base_folder}/{folder}/train/{name}' for name in os.listdir(train_folder)]
    val_files = [f'{base_folder}/{folder}/val/{name}' for name in os.listdir(val_folder)]

    return train_files, val_files