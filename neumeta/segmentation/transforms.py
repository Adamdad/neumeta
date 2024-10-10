import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import ImageFilter

#color mapping
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

def pad_to_divisible_by_32(img, size, fill=0):
    """
    Adjust the padding of an image to ensure that its dimensions are divisible by 32.
    The image will be padded if it is smaller than the specified size.
    """
    # Get original width and height
    ow, oh = img.size

    # Pad width and height to be divisible by 32
    padw = (32 - ow % 32) 
    padh = (32 - oh % 32) 

    # Pad the image if necessary
    if padw > 0 or padh > 0:
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, antialias=True)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class Resize:
    def __init__(self, size):
       
        self.size = size

    def __call__(self, image, target):
        # image = pad_to_divisible_by_32(image, self.size)
        image = F.resize(image, self.size,  antialias=True)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        image = pad_to_divisible_by_32(image, self.size)
        target = pad_to_divisible_by_32(target, self.size, fill=255)
        return image, target
    

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __init__(self, normalize=True) -> None:
        self.normalize = normalize
    def __call__(self, image, target):
        if self.normalize:
            image = F.to_tensor(image)
        else:
            image = F.pil_to_tensor(image)
        
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToDtype:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target):
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
class RandomGaussianBlur:
    def __init__(self, radius_range):
        self.radius_range = radius_range

    def __call__(self, image, target):
        if random.random() < 0.5:
            radius = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius))
        return image, target
    
class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target