import numpy as np
import torch

import random
from torchvision.transforms import functional as F
import numbers     

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        t,  c,  height,  width = img.shape
        th,  tw = output_size
        if width == tw and height == th:
            return 0, 0, height, width

        i = random.randint(0, height - th) if height!=th else 0
        j = random.randint(0, width - tw) if width!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        i, j, height, width = self.get_params(imgs, self.size)
        imgs = imgs[:, :, i:i+height, j:j+width]
        return imgs

    def __repr__(self):
        #return printed info
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, tensor):
        return tensor.sub_(self.mean).div_(self.std)
            
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        t, c, height, width = imgs.shape
        th, tw = self.size
        i = int(np.round((height - th) / 2.))
        j = int(np.round((width - tw) / 2.))

        return imgs[:, :, i:i+th, j:j+tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            return torch.from_numpy(np.flip(imgs, axis=3).copy())
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
