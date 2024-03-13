from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)  # spiltext() a.npy to a and .npy
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')  # show npy num

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, npy):
        img_nd = npy

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + '/' + idx + '.*')
        img_file = glob(self.imgs_dir + '/' + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0])
        mask = np.load(mask_file[0])
        # img = Image.open(img_file[0])
        img = np.load(img_file[0])

        # assert img.size == mask.size, \
        assert img.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'id':idx
        }
