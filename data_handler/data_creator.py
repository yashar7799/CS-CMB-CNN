import numpy as np
import healpy as hp
import ccgpack as ccg
import os
import shutil
from glob import glob
from PIL import Image
import gdown


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

    def __init__(self):

        pass

    def download(self, download_base_folder:str = './dataset/raw_data'):
        
        self.download_base_folder = download_base_folder

        os.makedirs(f'{download_base_folder}', exist_ok=True)

        # download string maps:
        if not os.path.isfile(f'{download_base_folder}/map1n_allz_rtaapixlw_2048_1.fits'):
            gdown.download(id='15iwucSA5yqqEf-QtdcNJt2Dviv2L9fny', output=f'{download_base_folder}/map1n_allz_rtaapixlw_2048_1.fits')
        if not os.path.isfile(f'{download_base_folder}/map1n_allz_rtaapixlw_2048_2.fits'):
            gdown.download(id='1DNaOMEk1zzX_SzEn7Q6YqF91R3W31jRo', output=f'{download_base_folder}/map1n_allz_rtaapixlw_2048_2.fits')
        if not os.path.isfile(f'{download_base_folder}/map1n_allz_rtaapixlw_2048_3.fits'):
            gdown.download(id='1kL3-MsBGlOiWw4XdUGYrocISh8gejvR0', output=f'{download_base_folder}/map1n_allz_rtaapixlw_2048_3.fits')

        # download gaussian maps:
        if not os.path.isfile(f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0000.fits'):
            gdown.download(id='1atZ0F99rzmsKt3NdnCiOm17mc9B6U7qT', output=f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0000.fits')
        if not os.path.isfile(f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0001.fits'):
            gdown.download(id='1VWiXbsYh6M8HkuhorZuonQdXaDW6l3zl', output=f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0001.fits')
        if not os.path.isfile(f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0002.fits'):
            gdown.download(id='1XzMrbXe6hWJVZ0paMXm1UYy8fTj7q162', output=f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0002.fits')
        if not os.path.isfile(f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0003.fits'):
            gdown.download(id='1Fb7Yj4Pok-k8mSnH2b9pdS07gHofcwxB', output=f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0003.fits')
        if not os.path.isfile(f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits'):
            gdown.download(id='1KFIGKLee-OBrG7t5Gwk_uuTE0RegdsQ4', output=f'{download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits')

    def create(self, g_mu_values:list = [0, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9], val_ratio:float = 0.15, create_base_folder:str = '/content/drive/MyDrive/CS-CMB-CNN-data'):

        folders = []

        for g_mu in g_mu_values:
            if g_mu == 0:
                folders.append('0')
            else:
                l = str(g_mu).split('-')
                folders.append(l[0] + '_' + l[1])

        for folder in folders:
            os.makedirs(os.path.join(create_base_folder, 'train', folder), exist_ok=True)
            os.makedirs(os.path.join(create_base_folder, 'val', folder), exist_ok=True)
            os.makedirs(os.path.join(create_base_folder, 'test', folder), exist_ok=True)

        # train & val data: mixed at first; will be seperated later at partitioning method.
        for s in [1, 2]:
            for g in [0, 1, 2, 3]:

                string_map = hp.read_map(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_{s}.fits', nest=1)
                gaussian_map = hp.read_map(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_000{g}.fits', nest=1)

                for g_mu, g_mu_str in zip(g_mu_values, folders):

                    cmb_with_string = gaussian_map + g_mu * string_map

                    cmb_with_string_patchs = ccg.sky2patch(cmb_with_string, 8)

                    train_array, val_array = np.split(cmb_with_string_patchs, [int(len(cmb_with_string_patchs)* (1 - val_ratio))])

                    for n in range(len(train_array)):
                        if not os.path.isfile(f'{create_base_folder}/train/{g_mu_str}/{n}_{s}_{g}_{g_mu_str}.png'):
                            array = train_array[n]
                            array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                            image = Image.fromarray(array)
                            image.save(f'{create_base_folder}/train/{g_mu_str}/{n}_{s}_{g}_{g_mu_str}.png')

                    for n in range(len(val_array)):
                        if not os.path.isfile(f'{create_base_folder}/val/{g_mu_str}/{n}_{s}_{g}_{g_mu_str}.png'):
                            array = val_array[n]
                            array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                            image = Image.fromarray(array)
                            image.save(f'{create_base_folder}/val/{g_mu_str}/{n}_{s}_{g}_{g_mu_str}.png')

        # test data: completely seperated from raw data.
        string_map = hp.read_map(f'{self.download_base_folder}/map1n_allz_rtaapixlw_2048_3.fits', nest=1)
        gaussian_map = hp.read_map(f'{self.download_base_folder}/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_0004.fits', nest=1)

        for g_mu, g_mu_str in zip(g_mu_values, folders):

            cmb_with_string = gaussian_map + g_mu * string_map

            cmb_with_string_patchs = ccg.sky2patch(cmb_with_string, 8)

            for n in range(len(cmb_with_string_patchs)):
                if not os.path.isfile(f'{create_base_folder}/test/{g_mu_str}/{n}_3_4_{g_mu_str}.png'):
                    array = cmb_with_string_patchs[n]
                    array = ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')
                    image = Image.fromarray(array)
                    image.save(f'{create_base_folder}/test/{g_mu_str}/{n}_3_4_{g_mu_str}.png')

    def partitioning(self, partitioning_base_folder='./dataset'):

        train_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'train')))
        val_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'val')))
        test_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'test')))

        if not train_classes == val_classes == test_classes:
            raise FileNotFoundError('data is not completely ready!\ncheck that you run data creation correctly.')

        folders = train_classes
        partition = {'train':[], 'val':[], 'test':[]}
        labels = {}

        for folder in folders:

            train_files = np.array(glob(os.path.join(partitioning_base_folder, 'train', folder, '*')))
            val_files = np.array(glob(os.path.join(partitioning_base_folder, 'val', folder, '*')))
            test_files = np.array(glob(os.path.join(partitioning_base_folder, 'test', folder, '*')))

            for train in train_files:
                partition['train'].append(train)
                labels[train] = folder

            for val in val_files:
                partition['val'].append(val)
                labels[val] = folder

            for test in test_files:
                partition['test'].append(test)
                labels[test] = folder


        # print out train/val/test counts:

        print('Classes and train/val/test counts:\n')
        
        for label in folders:

            n_train = len(os.listdir(os.path.join(partitioning_base_folder, label, 'train')))
            n_val = len(os.listdir(os.path.join(partitioning_base_folder, label, 'val')))
            n_test = len(os.listdir(os.path.join(partitioning_base_folder, label, 'test')))

            print(f'{label} >>> train: {n_train} | val: {n_val} | test: {n_test}')

        return partition, labels