from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from prep import *
from dataloader import get_train_loader
from dataloader import Kvasir, ValPre
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR, WarmUpPolyLR_eps
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist
from einops import rearrange, reduce, repeat
from eval import SegEvaluator, eval_per_epoch
from utils.pyt_utils import parse_devices
from torchvision.utils import save_image

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics as smp_metrics

import warnings
warnings.filterwarnings(action='ignore') 

import wandb

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '29500'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    rank = dist.get_rank()
    if rank == 0:
        wandb.init(project="segaug_miccai24")
        wandb.run.name = "aac"
        wandb.run.save()
        wandb.config.update({"attack_step": config.attack_step})
        wandb.config.update({"cr_weight": config.cr_weight})
        wandb.config.update({"network": config.network})

    # config network and criterion
    criterion = smp.losses.DiceLoss('binary')

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm

    # define and init the model
    select_classes = ['polyp', 'background']
    class_rgb_values = [[255, 255, 255], [0, 0, 0]]
    ENCODER = config.encoder_network
    ENCODER_WEIGHTS = config.encoder_weight
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    if config.network == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'pspnet':
        model = smp.PSPNet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )
    elif config.network == 'linknet':
        model = smp.Linknet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(select_classes), 
            activation=ACTIVATION,
        )


    select_class_indices = [select_classes.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    val_data_setting = {'img_root': config.img_root_folder,
                'gt_root': config.gt_root_folder,
                'train_source': config.train_source,
                'eval_source': config.eval_source}

    train_loader, train_sampler = get_train_loader(engine, Kvasir, train_source=config.train_source, 
                                            augmentation=get_training_aug_basic_scratch(),
                                            class_rgb_values=select_class_rgb_values)

    train_loader_aug, train_sampler_aug = get_train_loader(engine, Kvasir, train_source=config.train_source, 
                                augmentation=get_training_aug_basic_scratch(),
                                class_rgb_values=select_class_rgb_values)

    val_dataset = Kvasir(val_data_setting, 'val', 
    augmentation=get_validation_aug_basic_scratch(), 
    class_rgb_values=select_class_rgb_values, training=False)


    # define the learning rate
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=base_lr)
        ])

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch

    eps = config.eps

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            rank = dist.get_rank()
            model.cuda()
            model = DDP(model.to(rank), find_unused_parameters=True, device_ids=[rank], output_device=[rank]) 


    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)

    if engine.continue_state_object:
        engine.restore_checkpoint() 

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)

        sum_sup_loss, sum_ptb_loss, sum_cr_loss = 0, 0, 0
        sum_delta_ptb_origin, sum_delta_origin_rev = 0, 0
        sum_W = [0]*(config.attack_step+1)

        imgs_climb_noise_all, imgs_adv_noise_all, imgs_gt_all = [], [], []

        for idx in pbar:
            optimizer.zero_grad()
    
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = next(iter(dataloader))
            imgs = minibatch['data']
            gts = minibatch['label']
            imgs = imgs.cuda(non_blocking=True)
            b, c, h, w = imgs.shape # b 3 h w

            imgs_min = torch.min(imgs).item()
            imgs_max = torch.max(imgs).item()

            rank = dist.get_rank()
            gts = gts.to(rank) # b 2 h w
            gts_pgd = gts.clone().detach().to(rank)
            gts_ = gts[:, select_classes.index('polyp'), :, :].clone().detach().cuda(non_blocking=True)
            gts_ = repeat(gts_, 'b h w -> b c h w', c=c)

            model.eval()
            model.zero_grad()
            imgs_adv = imgs.clone().detach()
            imgs_climb = imgs.clone().detach()
            
            imgs_adv_all = []
            for step in range(config.attack_step):
                """ get anti-adversarial perturbations """
                imgs_climb.requires_grad = True
                pred_climb = model(imgs_climb)
                # pred_climb = torch.where(gts_pgd[:, select_classes.index('polyp'), :, :] == 1.0, pred_climb, gts_pgd)

                loss_climb = -criterion(pred_climb, gts_pgd)
                dist.all_reduce(loss_climb, dist.ReduceOp.SUM)
                loss_climb = loss_climb / engine.world_size

                loss_climb.backward()
                
                grad_climb = imgs_climb.grad.detach()
                sum_grad_climb = torch.norm(grad_climb, p=1)
                grad_cilmb = grad_climb.sign_()

                MU_climb = eps * grad_climb # batch/ngpus, 3, 512, 512
                imgs_climb.requires_grad = False
                model.zero_grad()

                """ get adversarial perturbations """
                imgs_adv.requires_grad = True
                pred_adv = model(imgs_adv)

                loss_adv = criterion(pred_adv, gts_pgd)
                dist.all_reduce(loss_adv, dist.ReduceOp.SUM)
                loss_adv = loss_adv / engine.world_size
                
                loss_adv.backward()
                
                grad_adv = imgs_adv.grad.detach()
                sum_grad_adv = torch.norm(grad_adv, p=1)
                grad_adv = grad_adv.sign_()

                MU_adv = eps * grad_adv # batch/ngpus, 3, 512, 512
                imgs_adv.requires_grad = False
                model.zero_grad()

                with torch.no_grad():
                    climb_noise = torch.where(gts_ == 1.0, MU_climb, torch.tensor(0.).cuda()).detach()
                    adv_noise = torch.where(gts_ == 1.0, MU_adv, torch.tensor(0.).cuda()).detach()

                    imgs_climb += climb_noise
                    imgs_adv += adv_noise

                    imgs_adv = imgs_adv.detach()
                    imgs_adv = torch.clamp(imgs_adv, imgs_min, imgs_max)

                    imgs_climb = imgs_climb.detach()
                    imgs_climb = torch.clamp(imgs_climb, imgs_min, imgs_max)

                    imgs_adv_all.append(imgs_adv)

            torch.cuda.empty_cache()
            with torch.autograd.set_detect_anomaly(True):
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                imgs_adv_all = torch.stack(imgs_adv_all).cuda(non_blocking=True)
                imgs_adv_all = rearrange(imgs_adv_all, 'a n b h w -> (a n) b h w')

                imgs_adv = imgs_adv.clone().detach().cuda(non_blocking=True)
                imgs_climb = imgs_climb.clone().detach().cuda(non_blocking=True)
                imgs_all = torch.cat([imgs, imgs_climb, imgs_adv_all])

                pred_all = model(imgs_all)

                pred_ori = pred_all[:b]
                pred_rev = pred_all[b:2*b]

                with torch.no_grad():
                    rev_loss = criterion(pred_rev, gts)
                    dist.all_reduce(rev_loss, dist.ReduceOp.SUM)
                    rev_loss = rev_loss / engine.world_size

                    """ consistency regularization """
                    # _, max_rev = torch.max(pred_rev, dim=1)
                    # max_rev = max_rev.long()
                    argmax_rev = torch.argmax(pred_rev, dim=1)
                    max_rev = torch.zeros_like(pred_rev).scatter_(1, argmax_rev.unsqueeze(1), 1.)    

                sup_origin_loss = criterion(pred_ori, gts)
                dist.all_reduce(sup_origin_loss, dist.ReduceOp.SUM)
                sup_origin_loss = sup_origin_loss / engine.world_size

                w = (sup_origin_loss / (sup_origin_loss + rev_loss)).item()
                sum_W[0] += w

                cr_loss = w * criterion(pred_ori, max_rev)

                ptb_loss = 0
                for i in range(config.attack_step):
                    pred_ptb = pred_all[(2+i)*b:(3+i)*b]

                    ptb_step_loss = criterion(pred_ptb, gts)
                    dist.all_reduce(ptb_step_loss, dist.ReduceOp.SUM)
                    ptb_step_loss = ptb_step_loss / engine.world_size 
                    ptb_loss = ptb_loss + ptb_step_loss

                    w = (ptb_step_loss / (ptb_step_loss + rev_loss)).item()
                    sum_W[i+1] += w

                    cr_loss = cr_loss + w * criterion(pred_ptb, max_rev)

                dist.all_reduce(cr_loss, dist.ReduceOp.SUM)
                cr_loss = cr_loss / engine.world_size 

                cr_loss = cr_loss / (config.attack_step + 1)
                ptb_loss = ptb_loss / config.attack_step

                """ update parameters """  
                curr_lr = optimizer.param_groups[0]['lr']

                if (rev_loss < sup_origin_loss) & (cr_loss > 0.0):
                    loss = sup_origin_loss + ptb_loss + config.cr_weight * cr_loss
                else:
                    cr_loss = torch.tensor(0)
                    loss = sup_origin_loss + ptb_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            model.eval()
            with torch.no_grad():
                imgs_gt_all.append(gts)
                imgs_climb_noise_all.append(climb_noise)
                imgs_adv_noise_all.append(adv_noise)

                delta_ptb_origin = ptb_step_loss - sup_origin_loss # if delta > 0: origin is easier than ptb
                delta_origin_rev = sup_origin_loss - rev_loss # if delta > 0: rev is easier to predict and can be used as pseudo-labels to unlabeld data 

                dist.all_reduce(delta_origin_rev, dist.ReduceOp.SUM)
                delta_origin_rev = delta_origin_rev / engine.world_size

                dist.all_reduce(delta_ptb_origin, dist.ReduceOp.SUM)
                delta_ptb_origin = delta_ptb_origin / engine.world_size


            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % curr_lr \
                        + ' sup_loss=%.2f' % sup_origin_loss.item() \
                        + ' ptb_loss=%.4f' % ptb_loss.item() \
                        + ' cr_loss=%.4f' % cr_loss.item()
                            
            sum_sup_loss += sup_origin_loss.item()
            sum_ptb_loss += ptb_loss.item()
            sum_cr_loss += cr_loss.item()
            sum_delta_ptb_origin += delta_ptb_origin.item()
            sum_delta_origin_rev += delta_origin_rev.item()

            pbar.set_description(print_str, refresh=False)

        if rank == 0:
            engine.save_and_link_checkpoint(os.path.join(config.snapshot_dir, 'aac'),
                                config.log_dir,
                                config.log_dir_link)
            val_miou, val_mdice = eval_per_epoch(epoch, engine.world_size, val_dataset, os.path.join(config.snapshot_dir, 'aac'))
            wandb.log({"val_mIoU": val_miou,
                        "val_mDice": val_mdice,
                        "train_loss_sup": sum_sup_loss / len(pbar),
                        "train_loss_aug": sum_ptb_loss/ len(pbar),
                        "cr_loss": sum_cr_loss / len(pbar)})
       
            
        end_time = time.time()