from config import config

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import torchvision.transforms as transforms

def get_training_aug_basic():
    train_transform = [
        album.Resize(height=config.image_height, width=config.image_width, interpolation=cv2.INTER_CUBIC, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)

def get_training_aug_basic_scratch():
    train_transform = [
        album.Resize(height=config.image_height, width=config.image_width, interpolation=cv2.INTER_CUBIC, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.Normalize(config.image_mean, config.image_std)
    ]
    return album.Compose(train_transform)

def get_validation_aug_basic():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.Resize(height=config.image_height, width=config.image_width, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return album.Compose(test_transform)

def get_validation_aug_basic_scratch():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.Resize(height=config.image_height, width=config.image_width, interpolation=cv2.INTER_CUBIC, always_apply=True),
        album.Normalize(config.image_mean, config.image_std)
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)