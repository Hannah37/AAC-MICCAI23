# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import math
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
if os.getenv('volna') is not None:
    C.volna = os.environ['volna']

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'Anti-Adv-Aug'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]

C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath('log')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))

C.log_dir_link = osp.join(C.abs_dir, 'log')

# snapshot dir that stores checkpoints
if os.getenv('snapshot_dir'):
    C.snapshot_dir = osp.join(os.environ['snapshot_dir'], "snapshot")
else:
    C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

""" Data Dir and Weight Dir """
C.dataset_path = '/root/hyunacho/medidata_80-10-10/'
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
# C.pretrained_model = C.volna + 'DATA/pytorch-weight/resnet50_v1c.pth'

""" Path Config """
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir, 'furnace'))

''' Experiments Setting '''
C.train_source = osp.join(C.dataset_path, "kvasir_train.txt")
C.eval_source = osp.join(C.dataset_path, "kvasir_test.txt")
C.is_test = False
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

C.scheduler = False
C.max_rampdown_epochs = 120

C.eps = 0.001
C.attack_step = 10
C.cr_weight=0.01

C.network = str(os.environ['network'])
C.encoder_network = 'mobilenet_v2'
C.encoder_weight = None

''' Image Config '''
C.num_classes = 2
C.background = 0
C.image_mean = np.array([0.557, 0.322, 0.236])  
C.image_std = np.array([0.307, 0.215, 0.178])
C.image_height = 512
C.image_width = 608
C.num_train_imgs = 800 
C.num_eval_imgs = 100


"""Train Config"""
if os.getenv('learning_rate'):
    C.lr = float(os.environ['learning_rate'])
else:
    C.lr = 0.005

if os.getenv('batch_size'):
    C.batch_size = int(os.environ['batch_size'])
else:
    C.batch_size = 16

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4


C.nepochs = int(os.environ['nepochs'])
C.val_epoch = C.nepochs

C.max_samples = C.num_train_imgs     # Define the iterations in an epoch
C.cold_start = 0
C.niters_per_epoch = int(math.ceil(C.max_samples * 1.0 // C.batch_size))
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.warm_up_epoch = 0

''' Eval Config '''
# C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] 
C.eval_flip = False

"""Display Config"""
if os.getenv('snapshot_iter'):
    C.snapshot_iter = int(os.environ['snapshot_iter'])
else:
    C.snapshot_iter = 2

C.record_info_iter = 20
C.display_iter = 50