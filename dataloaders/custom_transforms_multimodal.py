import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['rgb']
        img = np.array(img).astype(np.float32)
        img -= self.mean
        img /= self.std

        sample['rgb'] = img

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key, item in sample.items():
            item = np.array(item).astype(np.float32).transpose((2, 0, 1))
            item = torch.from_numpy(item).float()
            sample[key] = item

        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # # img = img[:,::-1]
            for key, item in sample.items():
                item = item[:,::-1]
                sample[key] = item

        return sample

class RandomGaussianBlur(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            radius = random.random()
            # # img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            # img = cv2.GaussianBlur(img, (0,0), radius)
            for key, item in sample.items():
                if key == 'label' or key == 'mask' or key == 'nir_mask':
                    continue
                item = cv2.GaussianBlur(item, (0,0), radius)
                # GaussianBlur will squeeze dimension if the dimension is 1.
                if len(item.shape) < 3:
                    item = np.expand_dims(item, axis=2)

                sample[key] = item

        return sample

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        rgb = sample['rgb']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        h, w = rgb.shape[:2]
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            
        # random crop crop_size
        # w, h = img.size
        h, w = rgb.shape[:2]

        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        x1 = random.randint(0, max(0, ow - self.crop_size))
        y1 = random.randint(0, max(0, oh - self.crop_size))

        for key, item in sample.items():
            if key == 'label' or key == 'mask' or key == 'nir_mask':
                item = cv2.resize(item  ,(ow,oh), interpolation=cv2.INTER_NEAREST)
            else:
                item = cv2.resize(item  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
            # resize will squeeze dimension if the dimension is 1.
            if len(item.shape) < 3:
                item = np.expand_dims(item, axis=2)
            if short_size < self.crop_size:
                item_ = np.zeros((oh+padh,ow+padw, item.shape[-1]))
                item_[:oh,:ow] = item
                item = item_
            item = item[y1:y1+self.crop_size, x1:x1+self.crop_size]

            sample[key] = item

        return sample

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        rgb = sample['rgb']

        # w, h = img.size
        h, w = rgb.shape[:2]

        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # nir = nir.resize((ow, oh), Image.BILINEAR)

        # center crop
        # w, h = img.size
        # h, w = img.shape[:2]
        x1 = int(round((ow - self.crop_size) / 2.))
        y1 = int(round((oh - self.crop_size) / 2.))
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # nir = nir.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        for key, item in sample.items():
            if key == 'label' or key == 'mask' or key == 'nir_mask':
                item = cv2.resize(item  ,(ow,oh), interpolation=cv2.INTER_NEAREST)
            else:
                item = cv2.resize(item  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
            # resize will squeeze dimension if the dimension is 1.
            if len(item.shape) < 3:
                item = np.expand_dims(item, axis=2)

            item = item[y1:y1+self.crop_size, x1:x1+self.crop_size]

            sample[key] = item

        return sample