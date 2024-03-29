#!/home/user/miniconda/bin/python
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from prep import *
from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from dataloader import Kvasir
from dataloader import ValPre
# from unet import UNet
import segmentation_models_pytorch as smp


try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

logger = get_logger()

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from PIL import Image

default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

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
    return class_colors[1:]


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, select_classes, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        pred = self.whole_eval(img, config.image_width, config.image_height, select_classes, device) ##
        
        if config.encoder_weight is None:
            label = label.detach().cpu().numpy() 

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        # if self.save_path is not None:
        #     ensure_dir(self.save_path)
        #     ensure_dir(self.save_path+'_color')

        # fn = name + '.png'

        # # 'save colored result'
        # result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        # class_colors = get_class_colors()
        # palette_list = list(np.array(class_colors).flat)
        # if len(palette_list) < 768:
        #     palette_list += [0] * (768 - len(palette_list))
        # result_img.putpalette(palette_list)
        # result_img.save(os.path.join(self.save_path+'_color', fn))

        # 'save raw result'
        cv2.imwrite(os.path.join(self.save_path, fn), pred)
        # logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, mean_DICE, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)    

        # print(len(self.dataset.get_class_names()))
        result_line = print_iou(iu, mean_DICE, mean_pixel_acc,
                                self.dataset.get_class_names(), True)
        if azure:
            mean_IU = np.nanmean(iu)*100
            run.log(name='Test/Val-mIoU', value=mean_IU)
        return result_line, np.nanmean(iu)*100, mean_DICE*100

def eval_per_epoch(epoch, ngpus, val_dataset, snapshot_dir):
    # all_dev = parse_devices('0-' + str(ngpus-1))
    all_dev = parse_devices('0')
    
    select_classes = ['polyp', 'background']
    class_rgb_values = [[255, 255, 255], [0, 0, 0]]
    ENCODER = config.encoder_network
    ACTIVATION = 'sigmoid' 

    if config.network == 'unet':
        network = smp.Unet(
            encoder_name=ENCODER, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'unet++':
        network = smp.UnetPlusPlus(
            encoder_name=ENCODER, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'pspnet':
        network = smp.PSPNet(
            encoder_name=ENCODER, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'deeplabv3+':
        network = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'linknet':
        network = smp.Linknet(
            encoder_name=ENCODER, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    dataset = val_dataset

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
            config.image_std, network,
            config.eval_scale_array, config.eval_flip,
            all_dev, False, None, False)

        val_miou, val_mdice = segmentor.epoch_run(snapshot_dir, str(epoch), select_classes)
    return val_miou, val_mdice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    val_data_setting = {'img_root': config.img_root_folder,
                'gt_root': config.gt_root_folder,
                'train_source': config.train_source,
                'eval_source': config.eval_source}

    select_classes = ['polyp', 'background']
    class_rgb_values = [[255, 255, 255], [0, 0, 0]]
    select_class_indices = [select_classes.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    val_dataset = Kvasir(val_data_setting, 'val', 
    augmentation=get_validation_aug_basic_scratch(), 
    class_rgb_values=select_class_rgb_values, training=False)

    val_miou, val_mdice = eval_per_epoch(args.epochs, 8, val_dataset)

    # network = UNet(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d)
    # data_setting = {'img_root': config.img_root_folder,
    #                 'gt_root': config.gt_root_folder,
    #                 'train_source': config.train_source,
    #                 'eval_source': config.eval_source}
    # val_pre = ValPre()
    # dataset = Kvasir(data_setting, 'val', val_pre, training=False)

    # with torch.no_grad():
    #     segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
    #                              config.image_std, network,
    #                              config.eval_scale_array, config.eval_flip,
    #                              all_dev, args.verbose, args.save_path,
    #                              args.show_image)
    #     segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
    #                   config.link_val_log_file)
