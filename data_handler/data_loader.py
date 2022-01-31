import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, smart_resize
from tensorflow import image as im
from PIL import Image
from albumentations import (
    RandomBrightness, RandomContrast, Sharpen, Emboss, PiecewiseAffine,
    ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, Flip, OneOf, Compose, ElasticTransform 
)

class DataGenerator(Sequence):

    """
    Generates data on-the-fly for model training.
    Should use data_creator module first, to download and create proper datasets.
    """

    def __init__(self, list_IDs, labels, batch_size=32, dim=(256, 256), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'

        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
 
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def preprocess_image(self, image_path):
        """
        Parameters
        ----------
        image: image to be processed

        Returns
        -------
        preprocessed image ready for learning and prediction.
        """
        image = Image.open(image_path)
        image = img_to_array(image)
        image = smart_resize(image, self.dim)
        image = im.per_image_standardization(image)
        return image

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)




        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.preprocess_image(ID)

            # Store class
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)

    # @staticmethod
    # def aug_func(p=0.5):
    #     return Compose([Flip(p=0.5),
    #                     GaussNoise(p=0.2),
    #                     OneOf([
    #                         MotionBlur(p=.2),
    #                         MedianBlur(blur_limit=3, p=.1),
    #                         Blur(blur_limit=3, p=.1),
    #                     ], p=0.2),
    #                     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
    #                     OneOf([
    #                         OpticalDistortion(p=0.3),
    #                         GridDistortion(p=.1),
    #                         PiecewiseAffine(p=0.3),
    #                         ElasticTransform(p=0.3, alpha=20, sigma=60 * 0.05, alpha_affine=60 * 0.03)
    #                     ], p=0.2),
    #                     OneOf([
    #                         Sharpen(),
    #                         Emboss(),
    #                         RandomContrast(),
    #                         RandomBrightness(),
    #                     ], p=0.3),
    #                     HueSaturationValue(p=0.3),
    #                     ], p=p)
