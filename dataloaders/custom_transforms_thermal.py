import numpy as np
from PIL import Image
import torch
import cv2
#from ipdb import set_trace as st


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
    

class RandomFlip():
    def __init__(self, prob=0.5):
        #super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            for key, item in sample.items():
                item = item[:,::-1]
                sample[key] = item

        return sample


class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        #super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            w, h, c = list(sample.values())[0].shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            for key, item in sample.items():
                item = item[w1:w2, h1:h2]
                sample[key] = item

        return sample
    

class Resize():
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

    def __call__(self, sample):
        for key, item in sample.items():
            if key == 'label':
                item = cv2.resize(item  ,(self.w,self.h), interpolation=cv2.INTER_NEAREST)
            else:
                item = cv2.resize(item  ,(self.w,self.h), interpolation=cv2.INTER_LINEAR)
            # resize will squeeze dimension if the dimension is 1.
            if len(item.shape) < 3:
                item = np.expand_dims(item, axis=2)
            sample[key] = item

        return sample


class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        #super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            w, h, c = list(sample.values())[0].shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            for key, item in sample.items():
                item[w1:w2, h1:h2] = 0
                sample[key] = item

        return sample


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        #super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            for key, item in sample.items():
                if key == 'label':
                    continue
                item = (item * bright_factor).astype(item.dtype)
                sample[key] = item

        return sample


class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        #super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            w, h, c = list(sample.values())[0].shape

            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w,h,c)
            )
            for key, item in sample.items():
                if key == 'label':
                    continue
                item = (item + noise).clip(0,255).astype(item.dtype)
                sample[key] = item

        return sample