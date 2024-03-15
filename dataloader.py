import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape
from datasets.BaseDataset import BaseDataset
from visualize import *
from einops import rearrange, reduce, repeat

def random_mirror(img, gt=None):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)

    return img, gt

def normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img


def random_scale(img, gt=None, scales=None):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if gt is not None:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale

class TrainPre(object):
    def __init__(self):
        self.img_mean = None
        self.img_std = None

    def __call__(self, img, gt=None):
        img, gt = random_mirror(img, gt)

        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        else:
            p_gt = None

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = {}

        return p_img, p_gt, extra_dict

class ValPre(object):
    def __call__(self, img, gt):
        extra_dict = {}
        return img, gt, extra_dict

def get_train_loader(engine, dataset, train_source, augmentation, class_rgb_values, times=1, preprocessing=None):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': train_source,
                    'eval_source': config.eval_source}

    train_dataset = dataset(data_setting, "train", preprocessing, augmentation, class_rgb_values, config.max_samples*times)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler


class Kvasir(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None, augmentation=None, class_rgb_values=None,
                 file_length=None, training=True, pseudo_label=False):
        self.istraining = training
        super(Kvasir, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.class_rgb_values = class_rgb_values

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]

        if self.istraining:
            workdir = self._train_source
        else:
            workdir = self._eval_source
        datadir = workdir.split('/')[-1].split('.')[0]       # train or train_aug or xxx
        if 'train' in datadir:
            datadir = 'TrainDataset/Kvasir'
        elif 'test' in datadir:
            datadir = 'TestDataset/Kvasir'

        sp = names.strip().split('\t')
        if 'test' in datadir:
            img_path = os.path.join(self._img_path, datadir, sp[0] + '.png')
        else:
            img_path = os.path.join(self._img_path, datadir, 'images', sp[0] + '.png')

        if './' not in names:
            gt_path = os.path.join(self._gt_path, datadir, 'masks', sp[0] + '.png')
        else:       # self-training file
            gt_path = sp[1]

        img, gt = self._fetch_data(img_path, gt_path)

        # if gt is not None:
        #     gt = np.uint8(gt)

        # img = img[:, :, ::-1]


        # img = torch.from_numpy(np.ascontiguousarray(img)).float()
        # gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
 
        
        gt = one_hot_encode(gt, self.class_rgb_values).astype('float')

        if self.augmentation is not None:
            sample = self.augmentation(image=img, mask=gt)
            img, gt = sample['image'], sample['mask']

        if self.preprocess is not None:
            sample = self.preprocess(image=img, mask=gt)
            img, gt = sample['image'], sample['mask']
    
        else: # self.preprocess is None
            img = torch.from_numpy(img).float()
            gt = torch.from_numpy(gt).float()

            img =  rearrange(img, 'h w c -> c h w')
            gt =  rearrange(gt, 'h w c -> c h w') # 512, 608, 2 -> 

        item_name = names.strip()
        output_dict = dict(data=img, fn=str(item_name), label=gt)
        return output_dict

    def _fetch_data(self, img_path, gt_path=None, dtype=None):
        # img = self._open_image(img_path)
        # if gt_path is not None:
            # gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        return img, gt
        

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 21
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @classmethod
    def get_class_names(*args):
        return ['background', 'polyp']

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name

    def _get_file_names(self, split_name, train_extra=False):
        # assert split_name in ['train', 'val']
        source = self._train_source
        if not self.istraining:
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            name = item.strip()
            file_names.append(name)
        return file_names