import os
import glob
import numpy as np

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

CAT_DATA_PATH = '.\\cat\\data\\'
DOG_DATA_PATH = '.\\dog\\data\\'

CAT_TEST_PATH  = os.path.join( CAT_DATA_PATH, 'test\\')
CAT_TRAIN_PATH = os.path.join( CAT_DATA_PATH, 'train\\')
CAT_VAL_PATH   = os.path.join( CAT_DATA_PATH, 'val\\')

DOG_TEST_PATH  = os.path.join( DOG_DATA_PATH, 'test\\')
DOG_TRAIN_PATH = os.path.join( DOG_DATA_PATH, 'train\\')
DOG_VAL_PATH   = os.path.join( DOG_DATA_PATH, 'val\\')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cat_images_path, dog_images_path):
        self.cat_images = self.get_image_paths(cat_images_path)
        self.dog_images = self.get_image_paths(dog_images_path)
        self.images_paths = self.cat_images + self.dog_images
    
    def __getitem__(self, idx):
        image = self.load_as_tensor(idx)
        label = self.get_label(idx)
        return image, label
    
    def __len__(self):
        return len(self.images_paths)

    def get_image_paths(self, path_to_images):
        return [path for path in glob.glob(os.path.join(path_to_images, '*\\*.jpg'))]

    def load_as_tensor(self, idx):
        return transforms.ToTensor()(cv2.imread(self.images_paths[idx], cv2.IMREAD_COLOR))
    
    def get_label(self, idx):
        path = self.images_paths[idx]
        label = path.split('\\')[1]
        return 0 if label == 'cat' else 1
    
