import numpy as np
import healpy as hp
import ccgpack as ccg
import os
import shutil
from glob import glob
from PIL import Image


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

    def __init__(self, download_base_folder='./dataset/raw_data', create_base_folder='/content/drive/MyDrive/CS-CMB-CNN-data', partitioning_base_folder='./dataset'):
        
        self.download_base_folder = download_base_folder
        self.create_base_folder = create_base_folder
        self.partitioning_base_folder = partitioning_base_folder

    def download(self):
        
        os.makedirs(f'{self.download_base_folder}', exist_ok=True)

        # download string maps:
        if not os.path.isfile(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_1.fits'):
            os.system(f'gdown --id 15iwucSA5yqqEf-QtdcNJt2Dviv2L9fny -O {self.download_base_folder}/map1n_allz_rtaapixlw_2048_1.fits')
        if not os.path.isfile(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_2.fits'):
            os.system(f'gdown --id 1DNaOMEk1zzX_SzEn7Q6YqF91R3W31jRo -O {self.download_base_folder}/map1n_allz_rtaapixlw_2048_2.fits')
        if not os.path.isfile(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_3.fits'):
            os.system(f'gdown --id 1kL3-MsBGlOiWw4XdUGYrocISh8gejvR0 -O {self.download_base_folder}/map1n_allz_rtaapixlw_2048_3.fits')

        # download gaussian maps:
        if not os.path.isfile(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0000.fits'):
            os.system(f'gdown --id 1atZ0F99rzmsKt3NdnCiOm17mc9B6U7qT -O {self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0000.fits')
        if not os.path.isfile(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0001.fits'):
            os.system(f'gdown --id 1VWiXbsYh6M8HkuhorZuonQdXaDW6l3zl -O {self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0001.fits')
        if not os.path.isfile(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0002.fits'):
            os.system(f'gdown --id 1XzMrbXe6hWJVZ0paMXm1UYy8fTj7q162 -O {self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0002.fits')
        if not os.path.isfile(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0003.fits'):
            os.system(f'gdown --id 1Fb7Yj4Pok-k8mSnH2b9pdS07gHofcwxB -O {self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0003.fits')
        if not os.path.isfile(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits'):
            os.system(f'gdown --id 1KFIGKLee-OBrG7t5Gwk_uuTE0RegdsQ4 -O {self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits')

    def create(self):

        for folder in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:
            os.makedirs(os.path.join(self.create_base_folder, 'train_and_val', str(folder)), exist_ok=True)

        # train & val data: mixed at first; will be seperated later at partitioning method.
        for s in [1, 2]:
            for g in [0, 1, 2, 3]:

                string_map = hp.read_map(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_{s}.fits', nest=1)
                gaussian_map = hp.read_map(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_000{g}.fits', nest=1)

                for g_mu in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:

                    cmb_with_string = gaussian_map + g_mu * string_map

                    cmb_with_string_patchs = ccg.sky2patch(cmb_with_string, 8)

                    for n in range(768):
                        if not os.path.isfile(f'{self.create_base_folder}/train_and_val/{g_mu}/{n}_{s}_{g}_{g_mu}.png'):
                            array = cmb_with_string_patchs[n]
                            array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                            image = Image.fromarray(array)
                            image.save(f'{self.create_base_folder}/train_and_val/{g_mu}/{n}_{s}_{g}_{g_mu}.png')


        for folder in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:
            os.makedirs(os.path.join(self.create_base_folder, 'test', str(folder)), exist_ok=True)

        # test data: completely seperated from raw data.
        string_map = hp.read_map(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_3.fits', nest=1)
        gaussian_map = hp.read_map(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits', nest=1)

        for g_mu in [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]:

            cmb_with_string = gaussian_map + g_mu * string_map

            cmb_with_string_patchs = ccg.sky2patch(cmb_with_string, 8)

            for n in range(768):
                if not os.path.isfile(f'{self.create_base_folder}/test/{g_mu}/{n}_3_4_{g_mu}.png'):
                    array = cmb_with_string_patchs[n]
                    array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                    image = Image.fromarray(array)
                    image.save(f'{self.create_base_folder}/test/{g_mu}/{n}_3_4_{g_mu}.png')

    def partitioning(self, val_ratio=0.15):

        folders = ['0', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '5e-8', '1e-8', '5e-9', '1e-9']
        partition = {'train':[], 'val':[], 'test':[]}
        labels = {}

        for folder in folders:

            dirs = np.array(glob(os.path.join(self.create_base_folder, 'train_and_val', folder, '*')))
            test_dirs = np.array(glob(os.path.join(self.create_base_folder, 'test', folder, '*')))

            shutil.rmtree(os.path.join(self.partitioning_base_folder, folder), ignore_errors=True)

            train_folder = os.path.join(self.partitioning_base_folder, folder, 'train')
            val_folder = os.path.join(self.partitioning_base_folder, folder, 'val')
            test_folder = os.path.join(self.partitioning_base_folder, folder, 'test')

            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)

            np.random.shuffle(dirs)
            np.random.shuffle(test_dirs)
            train_dirs, val_dirs = np.split(dirs, [int(len(dirs)* (1 - val_ratio))])
            for train_dir in train_dirs:
                shutil.copy(train_dir, train_folder)
            for val_dir in val_dirs:
                shutil.copy(val_dir, val_folder)
            for test_dir in test_dirs:
                shutil.copy(test_dir, test_folder)

            train_files = np.array(glob(os.path.join(train_folder, '*')))
            val_files = np.array(glob(os.path.join(val_folder, '*')))
            test_files = np.array(glob(os.path.join(test_folder, '*')))

            for train in train_files:
                partition['train'].append(train)
                if folder == '0':
                    labels[train] = 0
                elif folder == '1e-5':
                    labels[train] = 1
                elif folder == '5e-6':
                    labels[train] = 2
                elif folder == '1e-6':
                    labels[train] = 3
                elif folder == '5e-7':
                    labels[train] = 4
                elif folder == '1e-7':
                    labels[train] = 5
                elif folder == '5e-8':
                    labels[train] = 6
                elif folder == '1e-8':
                    labels[train] = 7
                elif folder == '5e-9':
                    labels[train] = 8
                else :
                    labels[train] = 9

            for val in val_files:
                partition['val'].append(val)
                if folder == '0':
                    labels[val] = 0
                elif folder == '1e-5':
                    labels[val] = 1
                elif folder == '5e-6':
                    labels[val] = 2
                elif folder == '1e-6':
                    labels[val] = 3
                elif folder == '5e-7':
                    labels[val] = 4
                elif folder == '1e-7':
                    labels[val] = 5
                elif folder == '5e-8':
                    labels[val] = 6
                elif folder == '1e-8':
                    labels[val] = 7
                elif folder == '5e-9':
                    labels[val] = 8
                else :
                    labels[val] = 9

            for test in test_files:
                partition['test'].append(test)
                if folder == '0':
                    labels[test] = 0
                elif folder == '1e-5':
                    labels[test] = 1
                elif folder == '5e-6':
                    labels[test] = 2
                elif folder == '1e-6':
                    labels[test] = 3
                elif folder == '5e-7':
                    labels[test] = 4
                elif folder == '1e-7':
                    labels[test] = 5
                elif folder == '5e-8':
                    labels[test] = 6
                elif folder == '1e-8':
                    labels[test] = 7
                elif folder == '5e-9':
                    labels[test] = 8
                else :
                    labels[test] = 9

        # print out train/val/test counts:

        print('Classes and train/val/test counts:\n')
        
        for label in ['0', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '5e-8', '1e-8', '5e-9', '1e-9']:

            n_train = len(os.listdir(os.path.join(self.partitioning_base_folder, label, 'train')))
            n_val = len(os.listdir(os.path.join(self.partitioning_base_folder, label, 'val')))
            n_test = len(os.listdir(os.path.join(self.partitioning_base_folder, label, 'test')))

            print(f'{label} >>> train: {n_train} | val: {n_val} | test: {n_test}')

        return partition, labels