# By Yuxiang Sun, Jul. 3, 2021
# Email: sun.yuxiang@outlook.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
import cv2
from mypath import Path
from collections import OrderedDict

from torchvision import transforms
from dataloaders import custom_transforms_thermal as tr

class MF_dataset(Dataset):
    NUM_CLASSES = 9

    def __init__(self, args, split, base_dir=Path.db_root_dir('rgb_thermal_dataset'), input_h=480, input_w=640):
        super(MF_dataset, self).__init__()
        self.modality = ['rgb', 'thermal']
        assert args.dataset_modality.split(',') == self.modality
        for modality in args.use_modality.split(','):
            assert modality in self.modality

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(base_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = base_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.n_data    = len(self.names)
        self.args = args

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

    def read_image(self, name, folder, normalize_term=1):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(PIL.Image.open(file_path)).astype(np.float32) / normalize_term
        if len(image.shape) < 3: image = np.expand_dims(image, axis=2)
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image_data = self.read_image(name, 'images', 255)
        sample = OrderedDict([('rgb', image_data[:,:,:3]), ('thermal', image_data[:,:,3:]), ('label', self.read_image(name, 'labels'))])
        for split in self.split:
            if split == "train":
                ret = self.transform_tr(sample)
            
            elif split == 'val':
                ret = self.transform_val(sample)
            elif split == 'test':
                ret = self.transform_val(sample)

        for modality in self.modality:
            if not modality in self.args.use_modality.split(','):
                del ret[modality]
        return ret

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomFlip(prob=0.5),
            tr.RandomCrop(crop_rate=0.1, prob=1.0),
            tr.Resize(w=640, h=480),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(w=640, h=480),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return self.n_data