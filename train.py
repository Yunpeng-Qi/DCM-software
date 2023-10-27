import sys
from tokenize import group

sys.path.append('.')

import argparse
import datetime
import glob
import timm
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import einops
# import models as MODELS
import models
import numpy as np
import PIL.Image as Image
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm
from data_compression.datasets.datasets import ImageLMDBDataset, data_prefetcher
from models.utils.lpips import LPIPS
from models.utils.mylib import (generate_local_region_msk,
                                generate_random_group_msk,
                                generate_random_qmap, load_coco_labels,
                                load_img, parse_instance, quality2lambda,
                                visualize_bitmap, write_log)
from models.utils.pytorch_msssim import MSSSIMLoss, ms_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

torch.backends.cudnn.benchmark=True
torch.set_num_threads(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # deug时设置
# Common Setting
parser = argparse.ArgumentParser()
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--model', type=str, default='ours_meanscalehyper')
parser.add_argument('--total-iteration', type=int, default=2000000)
parser.add_argument('--saving-iteration', type=int, default=0)
parser.add_argument('--eval-interval', type=int, default=10000)
parser.add_argument('--saving-interval', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--lmbda', type=int, default=1024)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--N', type=int, default=128)
parser.add_argument('--M', type=int, default=192)
# ChARM settings
parser.add_argument('--char_layernum', type=int, default=3)
parser.add_argument('--char_embed', type=int, default=128)
parser.add_argument("--transform-channels", type=int, nargs='+',
                    default=[128, 128, 128, 192],help="Transform channels.")
parser.add_argument("--hyper-channels", type=int, nargs='+',
                    default=None, help="Transform channels.")
parser.add_argument("--depths_char", type=int, nargs='+',
                    default=[1,1,1],help="Depth of GroupSwinBlocks in ChARM.")
parser.add_argument("--num_heads_char", type=int, nargs='+',
                    default=[8,8,8],help="Head num of GroupSwinBlocks in ChARM.")

parser.add_argument('--patch-size', type=int, default=256)
parser.add_argument('--train-set', type=str, default='/data1/datasets/Imagenet')
parser.add_argument('--eval-set', type=str, default='/data/datasets/kodak/images')
parser.add_argument('--eval-folders', action='store_true', 
    help='there are folders in the args.eval_set')
parser.add_argument('--save', '-s', default='./logs/cube_train', type=str, help='directory for saving')
parser.add_argument('--metric', type=str, nargs='+',
                    default=['mse'], choices=['mse', 'msssim', 'lpips'])
parser.add_argument('--scheduler', type=str, default='multistep')
parser.add_argument('--multistep-milestones', type=int, nargs='+', default=[1800000])
parser.add_argument('--multistep-gamma', type=float, default=0.1)
parser.add_argument('--resume', type=str, default='')

parser.add_argument('--save-result', type=str, default='')
parser.add_argument('--save-qmap', type=str, default='')
parser.add_argument('--reset-rdo', action='store_true', help='reset the rdo to +inf.')

parser.add_argument('--soft-then-hard', action='store_true')
parser.add_argument('--soft-then-hard-start-iteration', type=int, default=0)
parser.add_argument('--freeze-transform', action='store_true')
parser.add_argument('--start-joint-training-iteration', type=int, default=-1)

# Transformer based Transform Coding
parser.add_argument('--swin-disable-norm', action='store_true',
                    help='do not use any normalization in the transformation.')

# GroupViT
parser.add_argument('--only-rec-fg', action='store_true')
parser.add_argument('--groupvit-save-group-msk', type=str, default='')
parser.add_argument('--groupvit-load-group-msk', type=str, default='')
parser.add_argument("--groups_tobe_decode", type=int, nargs='+',
                    default=[0, 0, 0], help="group idxs to be decoded.")

# Variable Rate Training
parser.add_argument('--vbr-training', action='store_true')
parser.add_argument('--low-lmbda', type=float, default=4, help="Lowest lambda for rate-distortion tradeoff.")
parser.add_argument('--high-lmbda', type=float, default=10, help="Highest lambda for rate-distortion tradeoff.")
parser.add_argument('--MSE-weight', type=float, default=1, metavar='N', help='Lambda (default: 128)')
parser.add_argument('--MSSSIM-weight', type=float, default=1, metavar='N', help='Lambda (default: 128)')
parser.add_argument('--GAN-weight', type=float, default=1, metavar='N', help='Lambda (default: 128)')
parser.add_argument('--LPIPS-weight', type=float, default=10, metavar='N', help='Lambda (default: 128)')

# Task-Driven Setting
parser.add_argument('--task-driven', action='store_true')
parser.add_argument('--td-det-weights', type=str, 
                    default='./../detectron2/ckpts/model_final_68b088.pkl')
parser.add_argument('--td-det-cfg-file', type=str, 
                    default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
parser.add_argument('--td-det-threshold', type=float, default=0.05)
parser.add_argument('--td-lmbda-feat', type=int, default=32)
parser.add_argument('--td-feat-ratio', type=float, nargs='+', 
                    default=[1,1,1,1,1], help='the ratio of different layers.')

# Task-Driven Bit Allocation Setting
parser.add_argument('--task-driven-bit-allocation', action='store_true')
parser.add_argument('--TDBA-lmbda', type=float, default=8)
parser.add_argument('--TDBA-iterations', type=int, default=10)

# Predicted Task-Driven Bit Allocation Setting
parser.add_argument('--predicted-task-driven-bit-allocation', action='store_true')
parser.add_argument('--PTDBA-lmbda', type=float, default=8)
parser.add_argument('--PTDBA-epoch', type=int, default=1)
parser.add_argument('--PTDBA-qg-resume', type=str, default='')  # what?

# Feature Compression
parser.add_argument('--feature-compression', action='store_true')
parser.add_argument('--fc-inference-dir', type=str, default='') # feature compression inference dir
parser.add_argument('--fc-ssic', action='store_true')           # understand SSIC
parser.add_argument('--fc-det-load-result', type=str, default='')

# Analysis Setting
parser.add_argument('--visualize-bit-allocation', type=str, default='',
                    help='path to the bit allocation saving directory.')

# ddp training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()

models = {
    'RDT_CheckerCube': models.MIK_codecs.RDT_CheckerCube,
    'ours_groupswin_channelar': models.ours_vit.GroupChARTTC,
}

vbr_models = {
    # "ours_spatialvbr_imgcomnet": models.CLIC_based_codecs.Spatially_adaptive_ImgComNet,
    # 'ours_spatialvbr_meanscalehyper': models.vbr.SpatiallyVariabeRateMeanScaleHyper,
    # 'ours_spatialvbr_groupswin_TfChARM': models.vbr.SpatiallyVariableRateGroupSwinTfChARM,
}

class DefaultTrainer():
    def __init__(self):
        self.build_logger()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_loader = self.build_train_loader()
        self.eval_ps = self.load_eval_ps()
        self.loss_fn = self.build_loss_fn()
        
        if args.resume:
            self.resume(args.resume)
        else:
            self.best_rdo = float('+inf')
            self.start_iteration = 1

    def build_logger(self):
        assert len(args.metric) == 1
        args.metric = args.metric[0]
        if args.metric == 'msssim':
            args.save = os.path.join(args.save, args.model+'_msssim', str(args.lmbda))
        elif args.model == 'ours_groupswin_TfChARM':   
            args.save = os.path.join(args.save, args.model, str(args.lmbda),
                                     f'ChAR-Layer{args.char_layernum}' +
                                     f'Embed{args.char_embed}' +
                                     'Depth'+str(''.join(str(t) for t in args.depths_char)) +
                                     'NumHeads'+str(''.join(str(t) for t in args.num_heads_char)))

        os.makedirs(args.save, exist_ok=True)
        self.p_log = os.path.join(
            args.save,
            '{}.txt'.format(str(datetime.datetime.now()).replace(':', '-')[:-7]))
        write_log(self.p_log, str(args).replace(', ', ',\n\t') + '\n')
        
    def log(self, content):
        return write_log(self.p_log, content)

    def ddp_training(self):
        # todo : code of this part is unusable. Debug it in the future. 
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        ngpus_per_node = torch.cuda.device_count()
        args.rank = args.rank * ngpus_per_node 

        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        self.model = torch.nn.parallel.DistributedDataParallel(
            # self.model, find_unused_parameters=True)
            self.model)

    def build_model(self):
        if 'ours' in args.model:
            if 'swin' in args.model:
                if args.swin_disable_norm:
                    norm_layer = nn.Identity
                    self.log('disable the layer normalization in transformation.')
                else:
                    norm_layer = nn.LayerNorm
                if 'TfChARM' in args.model:
                    model = models[args.model](norm_layer=norm_layer,
                                               char_layernum=args.char_layernum,
                                               depths_char=args.depths_char,
                                               num_heads_char=args.num_heads_char,
                                               char_embed=args.char_embed)
                else:
                    model = models[args.model](norm_layer=norm_layer)
            elif 'group' in args.model:
                model = models[args.model](args.hyper_channels)
            else:
                model = models[args.model](args.transform_channels, args.hyper_channels)
        elif 'elic' in args.model:
            model = models[args.model]()
        else:
            # model = models[args.model](args.N, args.M)
            model = models[args.model](args.N)
        model.train()
        model.cuda()
        self.log('\n'+str(model)+'\n\n')
        return model

    def build_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        return optimizer

    def build_scheduler(self):
        assert args.scheduler in ['multistep', 'cos']
        if args.scheduler == 'multistep':
            scheduler = MultiStepLR(
                self.optimizer, 
                milestones=args.multistep_milestones, 
                gamma=args.multistep_gamma)
        elif args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(self.optimizer, args.total_iteration)
        else:
            raise NotImplementedError
        self.log('scheduler: {}\n'.format(scheduler))
        
        return scheduler

    def build_train_loader(self):

        if args.lmbda > 512:
            args.train_set = '/data/datasets/Flickr2K_HR_lmdb'
        else:
            args.train_set = '/data/datasets/Coco'
        self.log('training dataset path: {}\n'.format(
            args.train_set
        ))

        # dataset = ImageLMDBDataset(args.train_set, is_training=True, patch_size=args.patch_size)
        dataset = ImageLMDBDataset(args.train_set, crop_size=args.patch_size)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        return train_loader

    def load_eval_ps(self):
        eval_ps = sorted(glob.glob(os.path.join(args.eval_set, '*.png')))
        if eval_ps == []:
            eval_ps = sorted(glob.glob(os.path.join(args.eval_set, '*.jpg')))
        return eval_ps

    def build_loss_fn(self):
        if args.metric == 'mse':
            loss_fn = nn.MSELoss().cuda()
        elif args.metric == 'msssim':
            loss_fn = MSSSIMLoss(1.0, True).cuda()
        # loss_fn = {}
        # for metric in args.metric:
        #     if metric == 'mse':
        #         loss_fn[metric] = nn.MSELoss().cuda()
        #     if metric == 'msssim':
        #         loss_fn[metric] = MSSSIMLoss(1.0, True).cuda()
        #     if metric == 'lpips':
        #         raise NotImplementedError
        return loss_fn

    def train(self):
        self.log('pre evaluation on entire images:\n')
        self.eval()
        print('pre evaluation on partial images:\n')
        self.eval_partial() # Note: only for debugging
        if 'group' in args.model:
            self.eval(eval_fg=True)
            self.log('\n')

        prefetcher = data_prefetcher(self.train_loader)

        self.model.train()
        for iteration in range(self.start_iteration, args.total_iteration + 1):
            #fetch data
            frames = prefetcher.next()
            if frames is None:
                prefetcher = data_prefetcher(self.train_loader)
                frames = prefetcher.next()

            # train one step
            with torch.autograd.set_detect_anomaly(True):
                if 'group' in args.model:
                    b,c,h,w = frames[0].shape
                    msk = generate_random_group_msk(b,h,w,16)
                    for bi in range(b):
                        if random.random() > 0.5:
                            msk[bi, ...] = 0
                    
                    if not args.soft_then_hard:
                        res = self.model(frames[0], noisy=True, msk=msk)
                    else:
                        if iteration > args.soft_then_hard_start_iteration:
                            res = self.model(frames[0], noisy=False, msk=msk)
                else:
                    if not args.soft_then_hard:
                        res = self.model(frames[0], noisy=True)
                    else:
                        if iteration > args.soft_then_hard_start_iteration:
                            res = self.model(frames[0], noisy=False)

            ## calculate loss
            loss = self.calculate_loss(frames[0], res)

            # optimize
            self.optimizer.zero_grad()
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if iteration < (args.total_iteration * 0.9):
                eval_interval = args.eval_interval * 10
            else:
                eval_interval = args.eval_interval
            # eval_interval = args.eval_interval      # ! debug

            if iteration % eval_interval == 0:
                rdo = self.eval(iteration)
                if 'group' in args.model:
                    rdo += self.eval(iteration, eval_fg=True)
                    self.log('\n')

                ## save best model
                if rdo < self.best_rdo:
                    self.best_rdo = rdo
                    self.log('Best model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    if iteration >= args.saving_iteration:
                        self.save_ckpt(iteration)

            if args.saving_interval:
                if (iteration+1) % args.saving_interval == 0:
                    self.log('Save model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    self.save_ckpt(iteration, '{}.pth'.format(iteration+1))

            if args.soft_then_hard:
                if args.soft_then_hard_start_iteration == iteration:
                    self.log('-------------------------------\n')
                    self.log('Start hard training ! \n')
                    self.log('-------------------------------\n')
                    self.model.soft_then_hard()
                    self.show_learnable_params()

            if iteration == args.start_joint_training_iteration:
                self.log('-------------------------------\n')
                self.log('Start joint training ! \n')
                self.log('-------------------------------\n')
                for p in self.model.parameters():
                    p.requires_grad = True
                self.show_learnable_params()

    def eval(self, iteration=None, eval_fg=False):
        self.model.eval()
        torch.cuda.empty_cache()
        log = {
            'bpp':0,
            'bpp_y':0,
            'bpp_side':0,
            'psnr': 0,
            'ms_ssim':0,
        }
        
        with torch.no_grad():
            for input_p in self.eval_ps:
                torch.cuda.empty_cache()
                ## forward
                x, hx, wx = load_img(input_p, padding=True, factor=64)
                x = x.cuda()

                if eval_fg:
                    if args.groupvit_load_group_msk:
                        msk = Image.open(os.path.join(args.groupvit_load_group_msk, input_p.split('/')[-1])).convert('L')
                        msk = np.asarray(msk).astype(np.uint8)
                        msk[msk!=0] = 1
                        _, msk, _, _ = cv2.connectedComponentsWithStats(
                            msk, connectivity=4)
                        msk = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)
                    else:
                        b,c,h,w = x.shape
                        msk = generate_random_group_msk(b,h,w,16)
                        if args.groupvit_save_group_msk:
                            os.makegdirs(args.groupvit_save_group_msk, exist_ok=True)
                            torchvision.utils.save_image(
                                msk.float()/msk.max(), os.path.join(
                                args.groupvit_save_group_msk, input_p.split('/')[-1]))
                    res = self.model(x, noisy=False, msk=msk, only_rec_fg=eval_fg)
                else:
                    if 'group' in args.model:
                        b,c,h,w = x.shape
                        msk = generate_random_group_msk(b,h,w,16)
                        res = self.model(x, noisy=False, msk=msk)
                    else:
                        res = self.model(x, noisy=False)
                loss = self.calculate_loss(x, res)

                x = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                x_hat = res['x_hat'][:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                if eval_fg:
                    fg_msk = msk.float()
                    fg_msk[fg_msk!=0] = 1
                    fg_msk = F.interpolate(fg_msk, size=(hx, wx), mode='nearest').cuda() # [1,1,hx,wx]
                    # psnr = 20 * np.log10(255.) - 10 * torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / (hx* wx))
                    # print(hx, wx, hx*wx, fg_msk.shape, fg_msk.sum(), ((x - x_hat) ** 2).shape)
                    # print(torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / fg_msk.sum()))
                    # print(torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / (hx* wx)))
                    # raise
                    psnr = 20 * np.log10(255.) - 10 * torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / (fg_msk.sum()*3))
                else:
                    psnr = 20 * np.log10(255.) - 10 * torch.log10(((x - x_hat) ** 2).mean())
                msssim = ms_ssim(x, x_hat, data_range=255).item()

                if args.save_result:
                    os.makedirs(args.save_result, exist_ok=True)
                    p_save = os.path.join(args.save_result, input_p.split('/')[-1][:-4]+'.png')
                    torchvision.utils.save_image(x_hat/255, p_save)
                    self.log('{} -> {}\n'.format(input_p, p_save))

                log = self.update_log(log, loss, psnr, msssim)

            for key in log.keys():
                log[key] /= len(self.eval_ps)

        self.display_log(log, iteration)
        self.model.train()

        rdo = self.calculate_rdo(log)

        return rdo

    def eval_partial(self, iteration=None):
        self.model.eval()
        torch.cuda.empty_cache()
        log = {
            'bpp': 0,
            'bpp_y': 0,
            'bpp_side': 0,
            'psnr': 0,
            'ms_ssim': 0,
        }

        with torch.no_grad():
            for input_p in self.eval_ps:
                torch.cuda.empty_cache()
                ## forward
                x, hx, wx = load_img(input_p, padding=True, factor=64)
                x = x.cuda()

                if args.groupvit_load_group_msk:
                    msk = Image.open(
                        os.path.join(args.groupvit_load_group_msk,
                                     input_p.split('/')[-1])).convert('L')
                    msk = np.asarray(msk).astype(np.uint8)
                    msk[msk != 0] = 1
                    _, msk, _, _ = cv2.connectedComponentsWithStats(
                        msk, connectivity=4)
                    group_mask = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(x.device)
                if args.groups_tobe_decode == [0, 0, 0]:
                    group_idxs = group_mask.unique().tolist()
                    group_idxs = group_idxs[1:]
                elif args.groups_tobe_decode is not None:
                    for group_idx in args.groups_tobe_decode:
                        assert group_idx in group_mask.unique()
                    group_idxs = sorted(args.groups_tobe_decode)
                else:
                    group_idxs = group_mask.unique().tolist()
                    if -1 in group_idxs:
                        group_idxs.remove(-1)
                res = self.model(x, noisy=False)
                loss = self.calculate_loss(x, res)

                x = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                x_hat = res['x_hat'][:, :, :hx, :wx].mul(255).round().clamp(0,
                                                                            255)
                # decoded region psnr
                msk_trans = torch.ones_like(group_mask).to(x.device)
                msk_trans[~torch.isin(group_mask,
                                      torch.tensor(group_idxs).to(
                                          x.device))] = 0
                msk_trans = F.interpolate(msk_trans.float(), size=(hx, wx),
                                          mode='nearest').cuda()  # [1,1,hx,wx]
                psnr = 20 * np.log10(255.) - 10 * torch.log10(
                    (((x - x_hat) ** 2) * msk_trans).sum() / (
                                msk_trans.sum() * 3))
                msssim = ms_ssim(x, x_hat, data_range=255).item()

                if args.save_result:
                    os.makedirs(args.save_result, exist_ok=True)
                    p_save = os.path.join(args.save_result,
                                          input_p.split('/')[-1][:-4] + '.png')
                    torchvision.utils.save_image(x_hat / 255, p_save)
                    self.log('{} -> {}\n'.format(input_p, p_save))

                log = self.update_log(log, loss, psnr, msssim)

            for key in log.keys():
                log[key] /= len(self.eval_ps)

        self.display_log(log, iteration)
        self.model.train()

        rdo = self.calculate_rdo(log)

        return rdo


    def calculate_rdo(self, log):
        ## calculate rdo
        if args.metric == 'mse':
            rdo = log['bpp'] + 1 / (10 ** (log['psnr'] / 10.)) * args.lmbda
        else:
            assert args.metric == 'msssim'
            rdo = log['bpp'] + (1 - log['ms_ssim']) * args.lmbda
        return rdo

    def calculate_loss(self, x, res):
        loss = {}        
        loss = self.calculate_dist_loss(x, res, loss)
        loss = self.calculate_bpp_loss(x, res, loss)
        loss['loss'] = args.lmbda * loss['dist_loss'] + loss['bpp_loss']
        return loss

    def calculate_dist_loss(self, x, res, loss):
        x_hat = res['x_hat']
        loss['dist_loss'] = self.loss_fn(x, x_hat)
        return loss

    def calculate_bpp_loss(self, x, res, loss):
        b, _, h, w = x.shape
        n_pixels = b*h*w
        loss['bpp_y'] = res['bits']['y'] / n_pixels
        if ('z' in res['bits'].keys()):
            loss['bpp_side'] = res['bits']['z'] / n_pixels
            loss['bpp_loss'] = loss['bpp_y'] + loss['bpp_side']
        else:
            loss['bpp_loss'] = loss['bpp_y']
        return loss

    def save_ckpt(self, iteration, name=None):
        if name:
            filename = name
        else:
            filename = 'debug.pth'
        try:
            self.model.fix_tables() ## fix cdf tables
        except:
            self.log('error occured when self.model.fix_tables()')

        if args.multiprocessing_distributed and torch.cuda.device_count() > 1:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save({
            'best_rdo': self.best_rdo,
            'iteration': iteration,
            'parameters': state_dict
        }, os.path.join(args.save, filename))

    def resume(self, p_ckpt):
        ckpt = torch.load(p_ckpt)
        if 'best_rdo' in list(ckpt.keys()):
            self.best_rdo = ckpt['best_rdo']
        else:
            self.best_rdo = float('+inf')
            self.log('no best rdo loaded, set it as +inf.\n')

        if 'iteration' in list(ckpt.keys()):
            self.start_iteration = ckpt['iteration']
        else:
            self.start_iteration = 0
            self.log('no iteration loaded, set it as 0.\n')
        if args.reset_rdo:
            self.best_rdo = float('+inf')
            self.log('reset rdo to +inf.\n')
            self.start_iteration = 0
            self.log('reset iteration to 0.\n')
        if 'elic' in args.model:
            msg = self.model.load_state_dict(ckpt["params"], strict=False)
        else:
            msg = self.model.load_state_dict(ckpt['parameters'], strict=False)
        # self.log('resume the ckpt from : {} and the message is {}\n'.format(
        #     p_ckpt, msg
        # ))
        self.scheduler.step(self.start_iteration)
        self.log('resume info:\nbeginning lr: {:.6f}, best_rdo: {:.3f}\n\n'.format(
            self.optimizer.param_groups[0]['lr'], self.best_rdo))

    def update_log(self, log, loss, psnr, msssim):
        log['bpp'] += loss['bpp_loss'].item()
        log['bpp_y'] += loss['bpp_y'].item()
        if 'bpp_side' in loss.keys():
            log['bpp_side'] += loss['bpp_side'].item()
        log['psnr'] += psnr.item()
        log['ms_ssim'] += msssim
        return log

    def display_log(self, log, iter=None, n_blankline=1):
        if iter:
            self.log('iteration: {}\t'.format(iter))
        for k,v in log.items():
            self.log('{}: {:>6.5f}  '.format(k, v))
        for i in range(n_blankline+1):
            self.log('\n')

    def show_loss(self, loss, iteration, interval=100):
        if iteration % interval == 0:
            self.log('Loss: \t')
            for k,v in loss.items():
                self.log('{}: {:>7.6f}\t'.format(k, v))
            self.log('\n')

    def show_learnable_params(self):
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.log("Parameters to be updated: ")
        for each in enabled:
            self.log('\t{}\n'.format(str(each)))
        self.log('\n')
        

class TaskDrivenTrainer(DefaultTrainer):
    def __init__(self):
        args.save = os.path.join(args.save, args.model, str(args.td_lmbda_feat))    # td_lmbda_feat = 32
        args.td_feat_ratio = [each/sum(args.td_feat_ratio) for each in args.td_feat_ratio]

        self.build_logger()
        self.model = self.build_model()
        self.task_model = self.build_task_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_loader = self.build_train_loader()
        self.eval_ps = self.load_eval_ps()
        self.loss_fn = self.build_loss_fn()

        self.best_rdo = float('+inf')
        self.start_iteration = 1
        self.pixel_mean = torch.tensor([103.530, 116.280, 123.675]) # ImageNet Normalization
        self.pixel_std = torch.tensor([57.375, 57.120, 58.395])

        assert args.resume != None

        # resume model.
        ckpt = torch.load(args.resume)
        msg = self.model.load_state_dict(ckpt['parameters'], strict=False)
        self.log('resume the ckpt from : {} and the message is {}\n\n'.format(
            args.resume, msg))

        # freeze task model
        for param in self.task_model.model.parameters():
            param.requires_grad = False

        self.show_learnable_params()

    def build_optimizer(self):
        optimizer = optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.task_model.model.parameters()}
            ], 
            lr=args.lr
        )
        return optimizer

    def build_task_model(self):
        self.log('building task model ... ')

        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg_file = args.td_det_cfg_file
        cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.td_det_threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = args.td_det_weights
        detector = DefaultPredictor(cfg)
        self.log('building task done.\n')

        # demo
        # import cv2
        # img = cv2.imread('logs/samples/coco/000000000025.jpg')
        # predictions = detector(img)
        # instances = predictions['instances'].to("cpu")
        # print(instances)
        # raise

        return detector

    def get_feat(self, task_model, x, feat_type='stem'):   
        '''
            task model: a task model in detectron2 
            x: BxCxHxW, range in [0, 255]
            preprocess including normalize, 
            remember that the channel order of detectron2 is 'BGR'
            'Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.'
        '''
        x = x.flip(dims=[1])    

        b,_,h,w = x.shape
        pixel_mean = einops.repeat(self.pixel_mean, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        pixel_std = einops.repeat(self.pixel_std, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        x = (x - pixel_mean) / pixel_std

        if feat_type == 'stem':
            feat = task_model.model.backbone.bottom_up.stem(x)
        elif feat_type == 'p':
            feat = task_model.model.backbone(x)
        else:
            raise NotImplementedError

        return feat

    def calculate_loss(self, x, res):
        loss = {}        
        loss = self.calculate_dist_loss(x, res, loss)
        loss = self.calculate_feat_loss(x, res, loss)
        loss = self.calculate_bpp_loss(x, res, loss)
        loss['loss'] = args.lmbda * loss['dist_loss'] \
            + args.td_lmbda_feat * loss['feat_loss'] \
            + loss['bpp_loss'] 
        
        return loss

    def calculate_feat_loss(self, x, res, loss):
        x_hat = res['x_hat']

        feats = self.get_feat(self.task_model, x *255, 'p')  
        # feats = self.get_feat(self.task_model, x)
        for k,v in feats.items():
            feats[k] = v.detach()
        feat_hats = self.get_feat(self.task_model, x_hat *255, 'p')  
        # feat_hats = self.get_feat(self.task_model, x_hat)

        loss['feat_loss'] = torch.tensor(0.).to(x_hat.device)

        if len(feats.keys()) != len(args.td_feat_ratio):
            raise ValueError('Invalid params. feats n: {}, td_feat_ratio: {}'.format(
                len(feats.keys()), len(args.td_feat_ratio)
            ))

        for k, ratio in zip(feats.keys(), args.td_feat_ratio):
            feat = feats[k]
            feat_hat = feat_hats[k]
            feat = torch.clamp(feat, min=-255.5, max=256.49)
            feat_hat = torch.clamp(feat_hat, min=-255.5, max=256.49)
            loss['feat_loss'] += nn.MSELoss()(feat, feat_hat) * ratio

        return loss

    def calculate_rdo(self, log):
        ## calculate rdo
        rdo = log['bpp'] + log['feat_loss'] * args.td_lmbda_feat
        return rdo

    def update_log(self, log, loss, psnr, msssim):
        if 'feat_loss' not in log.keys():
            log['feat_loss'] = 0
            log['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
        log['bpp'] += loss['bpp_loss'].item()
        log['bpp_y'] += loss['bpp_y'].item()
        if 'bpp_side' in loss.keys():
            log['bpp_side'] += loss['bpp_side'].item()
        log['psnr'] += psnr.item()
        log['ms_ssim'] += msssim
        log['feat_loss'] += loss['feat_loss'].item()
        log['lr'] += self.optimizer.state_dict()['param_groups'][0]['lr']

        return log


class TaskDrivenBitAllocation(TaskDrivenTrainer):
    '''
    This works for Bit Allocation for Spatial Variable Rate Compression Models.
    '''
    def __init__(self):
        args.td_feat_ratio = [each/sum(args.td_feat_ratio) for each in args.td_feat_ratio]

        self.build_logger()
        self.model = self.build_model()
        self.task_model = self.build_task_model()
        self.eval_ps = self.load_eval_ps()
        self.loss_fn = self.build_loss_fn()

        self.pixel_mean = torch.tensor([103.530, 116.280, 123.675])
        self.pixel_std = torch.tensor([57.375, 57.120, 58.395])

        assert args.resume != None

        # resume model.
        ckpt = torch.load(args.resume)
        msg = self.model.load_state_dict(ckpt['parameters'], strict=False)
        self.log('resume the ckpt from : {} and the message is {}\n\n'.format(
            args.resume, msg))

    def build_model(self):
        if args.model == 'ours_spatialvbr_imgcomnet':
            args.N, args.M = 192, 320
            model = vbr_models['ours_spatialvbr_imgcomnet'](args)
        else:
            model = vbr_models[args.model](args.transform_channels, args.hyper_channels)
        model.train()
        model.cuda()
        self.log('\n'+str(model)+'\n\n')
        return model

    def load_eval_ps(self):
        eval_ps = sorted(glob.glob(os.path.join(args.eval_set, '*.png')))
        if not eval_ps:
            eval_ps = sorted(glob.glob(os.path.join(args.eval_set, '*.jpg')))
        return eval_ps

    def eval(self, iteration=None):
        self.model.eval()
        torch.cuda.empty_cache()
        log = {
            'bpp':0,
            'bpp_y':0,
            'bpp_side':0,
            'psnr': 0,
            'ms_ssim':0,
        }
        
        for input_p in tqdm.tqdm(self.eval_ps):
            if args.save_qmap:
                p_save = os.path.join(args.save_qmap, input_p.split('/')[-1][:-4]+'.png')
                if os.path.exists(p_save):
                    continue

            torch.cuda.empty_cache()
            ## forward
            x, hx, wx = load_img(input_p, padding=True, factor=64)
            if x.shape[1] != 3:
                self.log('{} with invalid shape of: {}, ' \
                        'take the first 3 channels to be an RGB image.\n'.format(
                    input_p, x.shape))
                x = x[:,:3,...]

            # generate qmap
            x = x.cuda()
            qmap = self.generate_qmap(x)
        
            with torch.no_grad():
                x, qmap = x.cuda(), qmap.cuda()

                res = self.model(x, qmap, noisy=False)

                loss = {}        
                loss = self.calculate_dist_loss(x, res, loss)
                loss = self.calculate_bpp_loss(x, res, loss)

                x = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                x_hat = res['x_hat'][:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                psnr = 20 * np.log10(255.) - 10 * torch.log10(((x - x_hat) ** 2).mean())
                msssim = ms_ssim(x, x_hat, data_range=255).item()

                if args.save_result:
                    os.makedirs(args.save_result, exist_ok=True)
                    p_save = os.path.join(args.save_result, input_p.split('/')[-1][:-4]+'.png')
                    torchvision.utils.save_image(x_hat/255, p_save)
                    self.log('{} -> {}\n'.format(input_p, p_save))

                if args.save_qmap:
                    os.makedirs(args.save_qmap, exist_ok=True)
                    p_save = os.path.join(args.save_qmap, input_p.split('/')[-1][:-4]+'.png')
                    # torchvision.utils.save_image(qmap, p_save)
                    torchvision.utils.save_image(qmap[:, :, :hx, :wx], p_save)

            # update log
            log['bpp'] += loss['bpp_loss'].item()
            log['bpp_y'] += loss['bpp_y'].item()
            if 'bpp_side' in loss.keys():
                log['bpp_side'] += loss['bpp_side'].item()
            log['psnr'] += psnr.item()
            log['ms_ssim'] += msssim
            self.log('eval: bpp: {:.4f}, dist_loss: {:.4f}\n\n'.format(
                loss['bpp_loss'].item(), loss['dist_loss'].item()
            ))

        for key in log.keys():
            log[key] /= len(self.eval_ps)

        self.display_log(log, iteration)
        self.model.train()

        return 

    def generate_qmap(self, x):
        # . same value
        q_factor = 0.3
        x = x.cuda()
        qmap = torch.tensor(q_factor).repeat(1, 1, x.size(2), x.size(3)).float().cuda()

        # preparation
        self.model.qmap = nn.Parameter(qmap)
        optimizer = optim.Adam(    # 这里理论上是不是直接写成{'params': self.model.qmap()}就可以
            [
                {'params': self.model.parameters()},
                {'params': self.task_model.model.parameters()}
            ], lr=0.05
        )

        # freeze codec and task model 
        for param in self.task_model.model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.qmap.requires_grad = True

        # update qmap
        iterations = args.TDBA_iterations
        self.log('iteration\tdist\tbpp\tloss\n')

        for iteration in range(iterations):
            res = self.model(x, self.model.qmap, noisy=False)
            x_hat = res['x_hat'].clamp(0, 1)    # 0~1

            x_tmp = self.detectron2_resize(x).detach() # resize to (800, 1333), which is consistent with detectron2
            x_hat = self.detectron2_resize(x_hat)
            # x_tmp = x

            feats = self.get_feat(self.task_model, x_tmp*255, 'p')
            for k,v in feats.items():
                feats[k] = v.detach()
            feat_hats = self.get_feat(self.task_model, x_hat*255, 'p')

            dist_loss = []
            for k, ratio in zip(feats.keys(), args.td_feat_ratio):
                feat = feats[k]
                feat_hat = feat_hats[k]
                dist_loss.append(nn.MSELoss()(feat, feat_hat) * ratio)

            loss = {}
            lmbda = args.TDBA_lmbda # default: 8
            loss['dist_loss'] = sum(dist_loss)
            loss = self.calculate_bpp_loss(x, res, loss)
            loss['loss'] = lmbda * loss['dist_loss'] + loss['bpp_loss']

            self.log('{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                iteration, loss['dist_loss'], loss['bpp_loss'], loss['loss']))

            # optimize
            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()
            self.model.qmap.data = self.model.qmap.data.clamp(0, 1)
            
        qmap = self.model.qmap.detach()
        self.log('qmap_max: {:.4f}, qmap_min: {:.4f}\n'.format(
            qmap.max().item(), qmap.min().item()))

        return qmap

    def detectron2_resize(self, x):
        _,_,h_ori,w_ori = x.shape
        size, max_size = 800, 1333
        scale = size * 1.0 / min(h_ori, w_ori)
        if h_ori < w_ori:
            newh, neww = size, scale * w_ori
        else:
            newh, neww = scale * h_ori, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(h_ori, w_ori)

        x = F.interpolate(x, scale_factor=scale, mode='bilinear')
        _,_,h,w = x.shape
        factor = 32
        dh = factor * math.ceil(h / factor) - h
        dw = factor * math.ceil(w / factor) - w
        x = F.pad(x, (0, dw, 0, dh))
        return x


# -----------------------------------
# todo: reorganize the code
# -----------------------------------

class QmapGeneratorUNet(nn.Module):
    def __init__(self, model='resnet18'):
        super(QmapGeneratorUNet, self).__init__()
        
        # Load the pre-trained ResNet18 model with features_only=True
        self.encoder = timm.create_model(model, features_only=True, pretrained=True)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Define the skip connections
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(64, 64, kernel_size=1)
        self.skip_conv4 = nn.Conv2d(64, 32, kernel_size=1)
        
        # Use pixel shuffle to upsample 
        self.final_conv = nn.Conv2d(96, 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # Encoder
        feats = self.encoder(x)
        feats = {
            'f32x': feats[-1],
            'f16x': feats[-2],
            'f8x': feats[-3],
            'f4x': feats[-4],
            'f2x': feats[-5],
        }

        # Decoder
        x = self.relu1(self.bn1(self.conv1(feats['f32x'])))
        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv1(feats['f16x'])], dim=1)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv2(feats['f8x'])], dim=1)
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv3(feats['f4x'])], dim=1)

        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv4(feats['f2x'])], dim=1)

        # Output
        x = self.final_conv(x)
        x = self.pixel_shuffle(x)
        x = torch.sigmoid(x)
        return x

# build a new UNet with resnet50 as pretrained encoder 
# from here: https://github.com/rawmarshmellows/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        return self.sigmoid(x)

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CustomMSCOCO(Dataset):
    '''
    This load the images of MS COCO with no labels. 
    Mainly used for the training of task driven bit-allocation.
    '''
    def __init__(self, data_dir, crop_size=256, subset_ratio=1.0, split='train'):
        # assert 'train2017' in data_dir
        self.data_dir = data_dir    
        self.crop_size = crop_size
        self.files = os.listdir(data_dir)
        self.files = [f for f in self.files if f.endswith('.jpg') or f.endswith('.png')]
        self.files = self.files[:int(len(self.files)*subset_ratio)]
        self.split = split

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        image = Image.open(file_path).convert('RGB')
        image = transforms.ToTensor()(image)

        scale_factor = torch.Tensor([self.get_detectron2_scale_factor(image.unsqueeze(0))])

        if self.split == 'train':
            if min(image.shape) < self.crop_size:
                pad_h = max(self.crop_size - image.shape[1], 0)
                pad_w = max(self.crop_size - image.shape[2], 0)
                image = F.pad(image, (0, pad_w, 0, pad_h), 'constant', 0)
            image = transforms.RandomCrop(self.crop_size)(image)

        return image, scale_factor

    def __len__(self):
        return len(self.files)

    def detectron2_resize(self, x):
        _,_,h_ori,w_ori = x.shape
        size, max_size = 800, 1333
        scale = size * 1.0 / min(h_ori, w_ori)
        if h_ori < w_ori:
            newh, neww = size, scale * w_ori
        else:
            newh, neww = scale * h_ori, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(h_ori, w_ori)

        x = F.interpolate(x, scale_factor=scale, mode='bilinear')
        _,_,h,w = x.shape
        factor = 64
        dh = factor * math.ceil(h / factor) - h
        dw = factor * math.ceil(w / factor) - w
        x = F.pad(x, (0, dw, 0, dh))
        return x

    def get_detectron2_scale_factor(self, x):
        _,_,h_ori,w_ori = x.shape
        size, max_size = 800, 1333
        scale = size * 1.0 / min(h_ori, w_ori)
        if h_ori < w_ori:
            newh, neww = size, scale * w_ori
        else:
            newh, neww = scale * h_ori, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(h_ori, w_ori)

        # x = F.interpolate(x, scale_factor=scale, mode='bilinear')
        # _,_,h,w = x.shape
        # factor = 64
        # dh = factor * math.ceil(h / factor) - h
        # dw = factor * math.ceil(w / factor) - w
        # x = F.pad(x, (0, dw, 0, dh))
        return scale


class PredictedTaskDrivenBitAllocation(TaskDrivenBitAllocation):
    '''
    Use a model to predict the qmap.
    '''
    def __init__(self):
        args.td_feat_ratio = [each/sum(args.td_feat_ratio) for each in args.td_feat_ratio]

        self.build_logger()
        self.model = self.build_model()
        self.task_model = self.build_task_model()
        # self.eval_ps = self.load_eval_ps()
        self.loss_fn = self.build_loss_fn()

        self.pixel_mean = torch.tensor([103.530, 116.280, 123.675])
        self.pixel_std = torch.tensor([57.375, 57.120, 58.395])

        assert args.resume != None

        # resume model.
        ckpt = torch.load(args.resume)
        msg = self.model.load_state_dict(ckpt['parameters'], strict=False)
        self.log('resume the ckpt for the codec from : {} and the message is {}\n\n'.format(
            args.resume, msg))
        
        # build qmap generator and loader    
        self.crop_size = 256
        # self.qmap_generator = QmapGeneratorUNet().cuda()
        self.qmap_generator = UNetWithResnet50Encoder().cuda()
        if args.PTDBA_qg_resume:
            ckpt = torch.load(args.PTDBA_qg_resume)
            msg = self.qmap_generator.load_state_dict(ckpt['parameters'], strict=False)
            self.log('resume the ckpt for qmap generator from : {} and the message is {}\n\n'.format(
                args.PTDBA_qg_resume, msg))

        train_dataset = CustomMSCOCO('/data1/datasets/coco_new/train2017',
                                    crop_size=self.crop_size)
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
        eval_dataset = CustomMSCOCO('/data1/datasets/coco_new/val2017', 
                                    crop_size=self.crop_size, subset_ratio=0.01, split='val')
        self.eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        if args.eval_set:
            self.eval_ps = self.load_eval_ps()

        # build optimizer
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

    def build_optimizer(self):
        optimizer = optim.Adam(
            [
                {'params': self.qmap_generator.parameters()},
            ], 
            lr=args.lr
        )
        return optimizer

    def build_scheduler(self):
        assert args.scheduler in ['multistep', 'cos']
        if args.scheduler == 'multistep':
            scheduler = MultiStepLR(
                self.optimizer, 
                milestones=[args.multistep_milestones[0] * len(self.train_loader)], 
                gamma=args.multistep_gamma)
        elif args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(self.optimizer, args.PTDBA_epoch * len(self.train_loader))
        else:
            raise NotImplementedError
        self.log('scheduler: {}\n'.format(scheduler))
        
        return scheduler

    def generate_qmap(self, x):
        '''
            Generate the qmap for the input image.
            x: the input image. BxCxHxW, range in [0, 1], the channel order is RGB.
        '''
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        b,_,h,w = x.shape
        pixel_mean = einops.repeat(mean, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        pixel_std = einops.repeat(std, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        x = (x - pixel_mean) / pixel_std

        qmap = self.qmap_generator(x)
        return qmap
 
    def train(self):
        self.model.eval()
        self.task_model.model.eval()
        self.qmap_generator.train()
        self.best_rdo = torch.inf

        for epoch in range(args.PTDBA_epoch):
            for iteration, batch in enumerate(self.train_loader):
                x, scale_factor = batch
                x = x.cuda()
                qmap = self.generate_qmap(x)
                
                res = self.model(x, qmap, noisy=False)
                x_hat = res['x_hat'].clamp(0, 1)    # 0~1

                x, x_hat, qmap = self.rescale_to_detectron_distribution(x, x_hat, qmap, scale_factor)

                # calculate feature level loss
                feats = self.get_feat(self.task_model, x*255, 'p')
                for k,v in feats.items():
                    feats[k] = v.detach()
                feat_hats = self.get_feat(self.task_model, x_hat*255, 'p')

                dist_loss = []
                for k, ratio in zip(feats.keys(), args.td_feat_ratio):
                    feat = feats[k]
                    feat_hat = feat_hats[k]
                    dist_loss.append(nn.MSELoss()(feat, feat_hat) * ratio)

                loss = {}
                lmbda = args.PTDBA_lmbda    # default: 8
                loss['dist_loss'] = sum(dist_loss)
                loss = self.calculate_bpp_loss(x, res, loss)
                loss['loss'] = lmbda * loss['dist_loss'] + loss['bpp_loss']

                if iteration % 1000 == 0:
                    torchvision.utils.save_image(x, 'logs/visualization/PTDBA/train/x.png')
                    torchvision.utils.save_image(x_hat, 'logs/visualization/PTDBA/train/x_hat.png')
                    torchvision.utils.save_image(qmap, 'logs/visualization/PTDBA/train/qmap.png')
                    print('lr: {}'.format(self.optimizer.param_groups[0]['lr']))
                    rdo = self.training_eval(epoch, iteration)
                    if rdo <= self.best_rdo:
                        self.best_rdo = rdo
                        self.save_ckpt(epoch*len(self.train_loader)+iteration)

                    self.qmap_generator.train()

                # optimize
                self.optimizer.zero_grad()
                loss['loss'].backward()
                self.optimizer.step()
                self.scheduler.step()

        return 

    def rescale_to_detectron_distribution(self, x, x_hat, qmap, scale_factor):
        x_tmp = []
        x_hat_tmp = []
        qmap_tmp = []
        for idx, sca in enumerate(scale_factor):
            tmpx = F.interpolate(x[idx].unsqueeze(0), scale_factor=float(sca), mode='bilinear')
            tmpxh = F.interpolate(x_hat[idx].unsqueeze(0), scale_factor=float(sca), mode='bilinear')
            tmpq = F.interpolate(qmap[idx].unsqueeze(0), scale_factor=float(sca), mode='bilinear')
            h_start = np.random.randint(0, tmpx.shape[2]-self.crop_size)
            w_start = np.random.randint(0, tmpx.shape[3]-self.crop_size)
            x_tmp.append(tmpx[:, :, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size])
            x_hat_tmp.append(tmpxh[:, :, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size])
            qmap_tmp.append(tmpq[:, :, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size])
        x = torch.cat(x_tmp, dim=0)
        x_hat = torch.cat(x_hat_tmp, dim=0)
        qmap = torch.cat(qmap_tmp, dim=0)
        return x, x_hat, qmap
    
    def training_eval(self, epoch, train_iteration):
        self.qmap_generator.eval()
        dist_loss_sum, bpp_sum, loss_sum = 0, 0, 0

        with torch.no_grad():
            x_viss, x_hat_viss, qmap_viss = [], [], []
            for iteration, batch in enumerate(tqdm.tqdm(self.eval_loader)):
                x, scale_factor = batch
                x = x.cuda()
                # pad the image to the multiple of 64
                _,_,h_ori,w_ori = x.shape
                factor = 64
                dh = factor * math.ceil(h_ori / factor) - h_ori
                dw = factor * math.ceil(w_ori / factor) - w_ori
                x = F.pad(x, (0, dw, 0, dh))
                qmap = self.generate_qmap(x)

                res = self.model(x, qmap, noisy=False)
                x_hat = res['x_hat'].clamp(0, 1)    # 0~1
                x = x[:, :, :h_ori, :w_ori]
                x_hat = x_hat[:, :, :h_ori, :w_ori]

                x_vis, x_hat_vis = x, x_hat

                x = F.interpolate(x, scale_factor=float(scale_factor), mode='bilinear')
                x_hat = F.interpolate(x_hat, scale_factor=float(scale_factor), mode='bilinear')
                _,_,h,w = x_hat.shape
                factor = 32
                dh = factor * math.ceil(h / factor) - h
                dw = factor * math.ceil(w / factor) - w
                x = F.pad(x, (0, dw, 0, dh))
                x_hat = F.pad(x_hat, (0, dw, 0, dh))

                # calculate feature level loss
                feats = self.get_feat(self.task_model, x*255, 'p')
                for k,v in feats.items():
                    feats[k] = v.detach()
                feat_hats = self.get_feat(self.task_model, x_hat*255, 'p')

                dist_loss = []
                for k, ratio in zip(feats.keys(), args.td_feat_ratio):
                    feat = feats[k]
                    feat_hat = feat_hats[k]
                    dist_loss.append(nn.MSELoss()(feat, feat_hat) * ratio)

                loss = {}
                lmbda = args.PTDBA_lmbda    # default: 8
                loss['dist_loss'] = sum(dist_loss)
                loss = self.calculate_bpp_loss(x, res, loss)
                loss['loss'] = lmbda * loss['dist_loss'] + loss['bpp_loss']

                dist_loss_sum += loss['dist_loss'].item()
                bpp_sum += loss['bpp_loss'].item()
                loss_sum += loss['loss'].item()

                if iteration < 16:
                    pad_factor = 600
                    x_vis = F.pad(x_vis, (0, pad_factor, 0, pad_factor))[:,:, :pad_factor, :pad_factor]
                    x_hat_vis = F.pad(x_hat_vis, (0, pad_factor, 0, pad_factor))[:,:, :pad_factor, :pad_factor]
                    qmap_vis = F.pad(qmap, (0, pad_factor, 0, pad_factor))[:,:, :pad_factor, :pad_factor]

                    x_viss.append(x_vis)
                    x_hat_viss.append(x_hat_vis)
                    qmap_viss.append(qmap_vis)

            x_viss = torch.cat(x_viss, dim=0)
            x_hat_viss = torch.cat(x_hat_viss, dim=0)
            qmap_viss = torch.cat(qmap_viss, dim=0)

            torchvision.utils.save_image(x_viss, 'logs/visualization/PTDBA/eval/x_{}_{}.png'.format(epoch, train_iteration), nrow=4)
            torchvision.utils.save_image(x_hat_viss, 'logs/visualization/PTDBA/eval/x_hat_{}_{}.png'.format(epoch, train_iteration), nrow=4)
            torchvision.utils.save_image(qmap_viss, 'logs/visualization/PTDBA/eval/qmap_{}_{}.png'.format(epoch, train_iteration), nrow=4)
                
            self.log('Evaluation: [Epoch-Iteration/All]:[{}-{}/{}]\tdist_loss: {:.4f}\tbpp_loss: {:.4f}\tloss: {:.4f}\n'.format(
                    epoch, train_iteration, len(self.train_loader),
                    dist_loss_sum / len(self.eval_loader), 
                    bpp_sum / len(self.eval_loader), 
                    loss_sum / len(self.eval_loader)))

        return loss_sum / len(self.eval_loader)
    
    def freeze(self):
        # freeze codec and task model 
        for param in self.task_model.model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.qmap_generator.parameters():
            param.requires_grad = True

    def save_ckpt(self, iteration, name=None):
        if name:
            filename = name
        else:
            filename = 'qmap_generator_best.pth'

        state_dict = self.qmap_generator.state_dict()
        torch.save({
            'best_rdo': self.best_rdo,
            'iteration': iteration,
            'parameters': state_dict
        }, os.path.join(args.save, filename))


class VBRTrainer(DefaultTrainer):
    def build_logger(self):
        args.save = os.path.join(
            args.save, 
            args.model + '_vbr_{}'.format('_'.join(args.metric)))

        os.makedirs(args.save, exist_ok=True)
        self.p_log = os.path.join(
            args.save,
            '{}.txt'.format(str(datetime.datetime.now()).replace(':', '-')[:-7]))
        write_log(self.p_log, str(args).replace(', ', ',\n\t') + '\n')

    def build_model(self):
        if args.model == 'ours_spatialvbr_imgcomnet':
            args.N, args.M = 192, 320
            model = vbr_models['ours_spatialvbr_imgcomnet'](args)
        elif 'swin' in args.model:
            if args.swin_disable_norm:
                norm_layer = nn.Identity
                self.log('disable the layer normalization in transformation.')
            else:
                norm_layer = nn.LayerNorm
            if 'TfChARM' in args.model:
                model = vbr_models[args.model](norm_layer=norm_layer,
                                       char_layernum=args.char_layernum,
                                       depths_char=args.depths_char,
                                       num_heads_char=args.num_heads_char,
                                       char_embed=args.char_embed)
        else:
            model = vbr_models[args.model](args.transform_channels, args.hyper_channels)
        model.train()
        model.cuda()
        self.log('\n'+str(model)+'\n\n')
        return model

    def build_loss_fn(self):
        # assert args.metric in ['mse', 'msssim']
        # if args.metric == 'mse':
        #     loss_fn = nn.MSELoss(reduce=False)
        # elif args.metric == 'msssim':
        #     loss_fn = MSSSIMLoss(1.0, normalize=True)
        # return loss_fn.cuda()
        loss_fn = {}
        if 'mse' in args.metric:
            loss_fn['mse'] = nn.MSELoss(reduce=False)
        if 'msssim' in args.metric:
            loss_fn['msssim'] = MSSSIMLoss(1.0, normalize=True, size_average=False)
        if 'lpips' in args.metric:
            loss_fn['lpips'] = LPIPS(net='alex', spatial=True).cuda()
            for p in loss_fn['lpips'].parameters():
                p.requires_grad = False

        return loss_fn

    def train(self):
        self.log('pre evaluation...\n')
        self.eval()  
        if 'group' in args.model:
            self.eval(eval_fg=True)
            self.log('\n')

        prefetcher = data_prefetcher(self.train_loader)

        self.model.train()
        for iteration in range(self.start_iteration, args.total_iteration + 1):
            #fetch data
            frames = prefetcher.next()
            if frames is None:
                prefetcher = data_prefetcher(self.train_loader)
                frames = prefetcher.next()

            # train one step
            with torch.autograd.set_detect_anomaly(True):
                b, c, h, w = frames[0].shape
                # bs, ps = args.batch_size, args.patch_size
                # qmap = torch.ones(b, 1, h, w).cuda() * torch.rand(b, 1, 1, 1).cuda()
                qmap = []
                for k in range(b):
                    qmap.append(
                        generate_random_qmap(args.patch_size).unsqueeze(0))
                qmap = torch.cat(qmap, dim=0).cuda()
                if 'group' in args.model:
                    msk = generate_random_group_msk(b, h, w, 16)
                    for bi in range(b):
                        if random.random() > 0.5:
                            msk[bi, ...] = 0
                    res = self.model(frames[0], qmap, noisy=True, msk=msk)
                else:
                    res = self.model(frames[0], qmap, noisy=True)

            ## calculate loss
            loss = self.calculate_loss(frames[0], res, qmap)

            # optimize
            self.optimizer.zero_grad()
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if iteration < (args.total_iteration * 0.9):
                eval_interval = args.eval_interval * 10
            else:
                eval_interval = args.eval_interval

            # eval_interval = args.eval_interval * 10 if iteration < args.saving_iteration else args.eval_interval
            # eval_interval = args.eval_interval  # ! debug

            if iteration % eval_interval == 0:
                rdo = self.eval(iteration)
                if 'group' in args.model:
                    rdo += self.eval(iteration, eval_fg=True)
                    self.log('\n')
                ## save best model
                if rdo < self.best_rdo:
                    self.best_rdo = rdo
                    if iteration >= args.saving_iteration:
                        self.log('Best model. Rdo is {:.4f} and save model to {}\n\n'.format(
                            rdo, args.save))
                        self.save_ckpt(iteration)

            if args.saving_interval:
                if (iteration+1) % args.saving_interval == 0:
                    self.log('Save model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    self.save_ckpt(iteration, '{}.pth'.format(iteration+1))

    def eval(self, iteration=None, valid_points=4, eval_fg=False):
        self.model.eval()
        rdo = 0
        for i in range(valid_points):
            q_factor = (i+1) / valid_points
            torch.cuda.empty_cache()
            log = {
                'bpp':0,
                'bpp_y':0,
                'bpp_side':0,
                'psnr': 0,
                'ms_ssim':0,
                'gan_loss':0,
                'lpips_loss':0,
            }
            
            with torch.no_grad():
                for input_p in self.eval_ps:
                    torch.cuda.empty_cache()
                    ## forward
                    x, hx, wx = load_img(input_p, padding=True, factor=64)
                    qmap = torch.tensor(q_factor).repeat(1, 1, x.size(2), x.size(3)).float()
                    x, qmap = x.cuda(), qmap.cuda()
                    if eval_fg:
                        if args.groupvit_load_group_msk:
                            msk = Image.open(
                                os.path.join(args.groupvit_load_group_msk,
                                             input_p.split('/')[-1])).convert(
                                'L')
                            msk = np.asarray(msk).astype(np.uint8)
                            msk[msk != 0] = 1
                            _, msk, _, _ = cv2.connectedComponentsWithStats(
                                msk, connectivity=4)
                            msk = torch.from_numpy(msk).unsqueeze(0).unsqueeze(
                                0)
                        else:
                            b, c, h, w = x.shape
                            msk = generate_random_group_msk(b, h, w, 16)
                            if args.groupvit_save_group_msk:
                                os.makedirs(args.groupvit_save_group_msk,
                                            exist_ok=True)
                                torchvision.utils.save_image(
                                    msk.float() / msk.max(), os.path.join(
                                        args.groupvit_save_group_msk,
                                        input_p.split('/')[-1]))
                        res = self.model(x, qmap, noisy=False, msk=msk,
                                         only_rec_fg=eval_fg)
                    else:
                        if 'group' in args.model:
                            b, c, h, w = x.shape
                            msk = generate_random_group_msk(b, h, w, 16)
                            # add by yixin
                            # print('only for debug!')
                            # msk = torch.ones([b, 1, h // 16, w // 16])
                            res = self.model(x, qmap, noisy=False, msk=msk)
                        else:
                            res = self.model(x, qmap, noisy=False)

                    loss = self.calculate_loss(x, res, qmap)
                    x = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                    x_hat = res['x_hat'][:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                    if eval_fg:
                        fg_msk = msk.float()
                        fg_msk[fg_msk != 0] = 1
                        fg_msk = F.interpolate(fg_msk, size=(hx, wx),
                                               mode='nearest').cuda()  # [1,1,hx,wx]
                        psnr = 20 * np.log10(255.) - 10 * torch.log10(
                            (((x - x_hat) ** 2) * fg_msk).sum() / (
                                        fg_msk.sum() * 3))
                        # debug = True
                        # if debug:
                        #     torchvision.utils.save_image(
                        #         x_hat[:, :, :hx, :wx] / 255, f"logs/debug/{input_p.split('/')[-1].split('.')[0]}_{str(i)}.png")
                    else:
                        psnr = 20 * np.log10(255.) - 10 * torch.log10(
                            ((x - x_hat) ** 2).mean())
                    msssim = ms_ssim(x, x_hat, data_range=255).item()

                    # update log
                    log['bpp'] += loss['bpp_loss'].item()
                    log['bpp_y'] += loss['bpp_y'].item()
                    if 'bpp_side' in loss.keys():
                        log['bpp_side'] += loss['bpp_side'].item()
                    log['psnr'] += psnr.item()
                    log['ms_ssim'] += msssim
                    if 'gan_loss' in list(loss.keys()):
                        log['gan_loss'] += loss['gan_loss'].item()
                    if 'lpips_loss' in list(loss.keys()):
                        log['lpips_loss'] += loss['lpips_loss'].item()
                
                for key in log.keys():
                    log[key] /= len(self.eval_ps)
                log['q_factor'] = q_factor

                self.display_log(log, iteration, 0)

            rdo_tmp = self.calculate_rdo(log, qmap)
            rdo += rdo_tmp

        self.model.train()
        self.log('\n')
        return rdo

    def calculate_rdo(self, log, qmap):
        rdo = log['bpp']
        if 'mse' in args.metric:
            rdo += 1 / (10 ** (log['psnr'] / 10.)) * quality2lambda(
                qmap, args.low_lmbda, args.high_lmbda).mean() * args.MSE_weight
        if 'msssim' in args.metric:
            rdo += (1 - log['ms_ssim']) * quality2lambda(
                qmap, args.low_lmbda, args.high_lmbda).mean() * args.MSSSIM_weight
        if 'lpips' in args.metric:
            raise NotImplementedError
        return rdo

    def calculate_loss(self, x, res, qmap):
        loss = {}
        loss = self.calculate_dist_loss(x, res, loss, qmap)
        loss = self.calculate_bpp_loss(x, res, loss)
        loss['loss'] = loss['dist_loss'] + loss['bpp_loss']
        return loss

    def calculate_dist_loss(self, x, res, loss, qmap):
        x_hat = res['x_hat']
        lmbda_map = quality2lambda(qmap, args.low_lmbda, args.high_lmbda)  

        loss['dist_loss'] = torch.tensor(0.).cuda()        
        if 'mse' in args.metric:
            loss['mse_loss'] = ((self.loss_fn['mse'](x, x_hat)) * lmbda_map).mean()
            loss['dist_loss'] += args.MSE_weight * loss['mse_loss']
        if 'msssim' in args.metric:
            msssim_dist = self.loss_fn['msssim'](x, x_hat)
            lambda_msssim_map = torch.mean(lmbda_map, dim=(1,2,3))
            distortion_msssim = (msssim_dist * lambda_msssim_map).mean()
            loss['msssim_loss']  = distortion_msssim 
            loss['dist_loss'] += args.MSSSIM_weight * loss['msssim_loss']
        if 'lpips' in args.metric:
            perceptual_loss = self.loss_fn['lpips'](x, x_hat)   # BCHW
            
            # original
            # lambda_lpips_map = lmbda_map.mean(dim=(1,2,3), keepdim=True)
            # perceptual_loss = perceptual_loss.mean(dim=(1,2,3), keepdim=True)
            # loss['lpips_loss'] = (lambda_lpips_map * perceptual_loss).mean()

            # spatial bit allocation
            loss['lpips_loss'] = (lmbda_map * perceptual_loss).mean()

            loss['dist_loss'] += args.LPIPS_weight * loss['lpips_loss']

        return loss


class FeatureCompressionTrainer(TaskDrivenTrainer):
    def __init__(self):
        args.save = os.path.join(args.save, 'feature_compression')
        self.build_logger()
        self.model = self.build_model()
        self.task_model = self.build_task_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_loader = self.build_train_loader()
        self.eval_ps = self.load_eval_ps()
        self.loss_fn = self.build_loss_fn()
        
        if args.resume:
            self.resume(args.resume)
        else:
            self.best_rdo = float('+inf')
            self.start_iteration = 1

        args.td_feat_ratio = [each/sum(args.td_feat_ratio) for each in args.td_feat_ratio]
        self.pixel_mean = torch.tensor([103.530, 116.280, 123.675])
        self.pixel_std = torch.tensor([57.375, 57.120, 58.395])

        # freeze task model
        for param in self.task_model.model.parameters():
            param.requires_grad = False

        self.show_learnable_params()

    def build_model(self):
        if 'ours' in args.model:
            if 'swin' in args.model:
                raise NotImplementedError
            elif 'group' in args.model:
                raise NotImplementedError
            else:
                channels = 256      # transform_channels=[128, 128, 128, 192], hyper_channels=None
                model = models[args.model](
                    args.transform_channels, args.hyper_channels,
                    in_channel=channels, out_channel=channels)
        else:
            raise NotImplementedError
        model.train()
        model.cuda()
        self.log('\n'+str(model)+'\n\n')
        return model

    def train(self):
        self.log('pre evaluation...\n')
        self.eval()

        prefetcher = data_prefetcher(self.train_loader)

        self.model.train()
        for iteration in range(self.start_iteration, args.total_iteration + 1):
            #fetch data
            frames = prefetcher.next()
            if frames is None:
                prefetcher = data_prefetcher(self.train_loader)
                frames = prefetcher.next()

            # train one step
            with torch.autograd.set_detect_anomaly(True):
                # compress stem feature
                # feat = self.get_feat(self.task_model, frames[0]*255, 'stem').detach()   # stem feature.

                # compress res2 feature
                feat = self.imgtores2(frames[0]*255).detach()

                res = self.model(feat, noisy=True)      # input是feature？

            ## calculate loss
            loss = self.calculate_loss(feat, res)

            # optimize
            self.optimizer.zero_grad()
            loss['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()

            if iteration < (args.total_iteration * 0.9):
                eval_interval = args.eval_interval * 10
            else:
                eval_interval = args.eval_interval
            # eval_interval = args.eval_interval // 10    # ! debug

            if iteration % eval_interval == 0:
                rdo = self.eval(iteration)

                ## save best model
                if rdo < self.best_rdo:
                    self.best_rdo = rdo
                    self.log('Best model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    if iteration >= args.saving_iteration:
                        self.save_ckpt(iteration)

            if args.saving_interval:
                if (iteration+1) % args.saving_interval == 0:
                    self.log('Save model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    self.save_ckpt(iteration, '{}.pth'.format(iteration+1))

    def eval(self, iteration=None):
        self.model.eval()
        torch.cuda.empty_cache()
        log = {
            'bpp':0,
            'bpp_y':0,
            'bpp_side':0,
        }
        
        with torch.no_grad():
            for input_p in self.eval_ps:
                torch.cuda.empty_cache()
                ## forward
                x, hx, wx = load_img(input_p, padding=True, factor=64)
                x = x.cuda()

                # compress stem feature
                # feat = self.get_feat(self.task_model, x*255, 'stem').detach()   # stem feature.

                # compress res2 feature
                feat = self.imgtores2(x*255).detach()

                res = self.model(feat, noisy=False)

                loss = self.calculate_loss(feat, res)
                log = self.update_log(log, loss)

            for key in log.keys():
                log[key] /= len(self.eval_ps)

        self.display_log(log, iteration)
        self.model.train()

        rdo = self.calculate_rdo(log)

        return rdo

    def stemtop(self, x):   # need to understands
        '''
        x: 
        Pass the stem feature to task model and output feature of p-layers.
        '''
        # get bottom_up_features
        outputs = {}
        bottom_up = self.task_model.model.backbone.bottom_up
        for name, stage in zip(bottom_up.stage_names, bottom_up.stages):
            x = stage(x)
            if name in bottom_up._out_features:
                outputs[name] = x
        bottom_up_features = outputs    # ['res2', 'res3', 'res4', 'res5']


        # get p-layered features
        backbone = self.task_model.model.backbone
        # backbone._out_features : ['p2', 'p3', 'p4', 'p5', 'p6']
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        assert len(backbone._out_features) == len(results)
        return {f: res for f, res in zip(backbone._out_features, results)}

    def imgtores2(self, x):
        # preprocess including normalize 
        x = x.flip(dims=[1])

        b,_,h,w = x.shape
        pixel_mean = einops.repeat(self.pixel_mean, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        pixel_std = einops.repeat(self.pixel_std, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        x = (x - pixel_mean) / pixel_std

        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        # x = self.stem(x)
        bottom_up = self.task_model.model.backbone.bottom_up
        x = bottom_up.stem(x)
        x = bottom_up.stages[0](x)
        return x

    def res2toreses(self, x):   # 从模型的不同阶段提取特征
        outputs = {}
        bottom_up = self.task_model.model.backbone.bottom_up
        count = 0
        for name, stage in zip(bottom_up.stage_names, bottom_up.stages):
            if count != 0:
                x = stage(x)
            if name in bottom_up._out_features:
                outputs[name] = x
            count += 1        
        
        return outputs

    def calculate_feat_loss(self, x, res, loss):
        x_hat = res['x_hat']

        # feats = self.stemtop(x)
        feats = self.res2toreses(x)
        for k,v in feats.items():
            feats[k] = v.detach()
        # feat_hats = self.stemtop(x_hat)
        feat_hats = self.res2toreses(x_hat)

        loss['feat_loss'] = torch.tensor(0.).to(x_hat.device)

        if len(feats.keys()) != len(args.td_feat_ratio):
            raise ValueError('Invalid params. feats n: {}, td_feat_ratio: {}'.format(
                len(feats.keys()), len(args.td_feat_ratio)
            ))

        for k, ratio in zip(feats.keys(), args.td_feat_ratio):
            feat = feats[k]
            feat_hat = feat_hats[k]
            feat = torch.clamp(feat, min=-255.5, max=256.49)
            feat_hat = torch.clamp(feat_hat, min=-255.5, max=256.49)
            loss['feat_loss'] += nn.MSELoss()(feat, feat_hat) * ratio

        return loss

    def calculate_bpp_loss(self, x, res, loss):
        b, _, h, w = x.shape
        n_pixels = b*h*w * (4*4)    # 4 times down-sampling
        loss['bpp_y'] = res['bits']['y'] / n_pixels
        if ('z' in res['bits'].keys()):
            loss['bpp_side'] = res['bits']['z'] / n_pixels
            loss['bpp_loss'] = loss['bpp_y'] + loss['bpp_side']
        else:
            loss['bpp_loss'] = loss['bpp_y']
        return loss

    def update_log(self, log, loss):
        if 'dist_loss' not in log.keys():
            log['dist_loss'] = 0
        if 'feat_loss' not in log.keys():
            log['feat_loss'] = 0
        if 'lr' not in log.keys():
            log['lr'] = 0

        log['bpp'] += loss['bpp_loss'].item()
        log['bpp_y'] += loss['bpp_y'].item()
        if 'bpp_side' in loss.keys():
            log['bpp_side'] += loss['bpp_side'].item()
        log['dist_loss'] += loss['dist_loss'].item()
        log['feat_loss'] += loss['feat_loss'].item()
        log['lr'] += self.optimizer.state_dict()['param_groups'][0]['lr']

        return log

    def inference(self):
        import json

        from detectron2.data.detection_utils import read_image
        coco_map_dict = load_coco_labels()
        self.inference_bits = 0
        self.inference_pixels = 0

        # print(self.task_model.input_format) # BGR

        self.model.eval()

        with torch.no_grad():
            torch.cuda.empty_cache()
            files = os.listdir(args.fc_inference_dir)
            final_instances = []

            for file in tqdm.tqdm(files):
                assert file[-4:] in ['.jpg', '.png']
                input_p = os.path.join(args.fc_inference_dir, file)
                # use PIL, to be consistent with evaluation
                original_image = read_image(input_p, format="BGR")
                self.inference_pixels += original_image.shape[0] * original_image.shape[1]

                ## inference
                # . in DefaultPredictor
                height, width = original_image.shape[:2]
                image = self.task_model.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": height, "width": width}

                # . in GeneralizedRCNN
                batched_inputs = [inputs]
                model_ = self.task_model.model
                images = model_.preprocess_image(batched_inputs)

                # features = model_.backbone(images.tensor)
                features = self.inference_get_features(images.tensor, input_p)

                if model_.proposal_generator is not None:
                    proposals, _ = model_.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(model_.device) for x in batched_inputs]

                results, _ = model_.roi_heads(images, features, proposals, None)

                # do_postprocess
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                predictions =  self.inference_postprocess(results, batched_inputs, images.image_sizes)
                instances = predictions[0]['instances'].to("cpu")

                instances_list = []

                img_id = file[:-4]
                if 'coco' in args.save:
                    img_id = int(img_id)

                for i in range(len(instances)):
                    instance = parse_instance(instances[i])
                    # if args.task == 'det':
                    instances_list.append({
                        'image_path': input_p,
                        'image_id': img_id,
                        'category_id': coco_map_dict[instance['pred_class']], 
                        'bbox': [
                            instance['pred_boxes'][0],
                            instance['pred_boxes'][1],
                            instance['pred_boxes'][2],
                            instance['pred_boxes'][3]
                        ],
                        'score': instance['scores'],
                    })
                final_instances += instances_list

            self.log('\n\ntotal instances: {}\n\n'.format(len(final_instances)))
            self.log('bpp: {}\n\n'.format(self.inference_bits / self.inference_pixels))

        for i in range(len(final_instances)):
            final_instances[i]['bbox'][2] = final_instances[i]['bbox'][2] - final_instances[i]['bbox'][0]
            final_instances[i]['bbox'][3] = final_instances[i]['bbox'][3] - final_instances[i]['bbox'][1]

        # save result
        p_det_result = os.path.join(args.save, 'result.json')
        json.dump(final_instances, open(p_det_result, 'w'), indent=4)

    def inference_postprocess(self, instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        from detectron2.modeling.postprocessing import detector_postprocess

        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def inference_get_features(self, x, input_p):
        # print(x.shape)  # [1, 3, 800, 1216]
        B,C,H,W = x.shape
        # torchvision.utils.save_image(x.flip(dims=[1]), 'x.png', normalize=True)
        import json

        import cv2
        bottom_up = self.task_model.model.backbone.bottom_up
        x = bottom_up.stem(x)
        x = bottom_up.stages[0](x)  # 1x256xHxW

        if args.fc_ssic:
            # load the detected results
            if not args.fc_det_load_result:
                raise NotImplementedError
            p_load = os.path.join(
                args.fc_det_load_result, input_p.split('/')[-1][:-4]+'.json')
            instances_list = json.load(open(p_load, 'r'))

            # generate the mask
            img = cv2.imread(input_p)
            # h,w,c = img.shape
            def get_scale(x, img):  # x: feature map, img: source image
                h,w,_ = img.shape
                _,_,h_new,w_new = x.shape
                h_new *= 4
                w_new *= 4
                if h_new == 1344:
                    return 1344 / h
                if w_new == 1344:
                    return 1344 / h
                if h_new == 800:
                    return 800 / h
                if w_new == 800:
                    return 800 / w
                return min(h_new/h, w_new/w)

            scale = get_scale(x, img)
            # print(scale)
            for instance in instances_list:
                # print(instance['bbox'])
                instance['bbox'] = [each * scale for each in instance['bbox']]
                # print(instance['bbox'])

            _, group_msk, _ = self.get_group_msk(H, W, instances_list)

            # process the mask to the same shape as input x 
            # print(img.shape)        #         (427, 640, 3)
            # print(x.shape)          # [1, 256, 200, 304]
            # print(group_msk.shape)  #         (28, 40)

            # torchvision.utils.save_image(x.squeeze(0).unsqueeze(1), './fea.png')
            # cv2.imwrite('group_msk.png', group_msk*(255/(group_msk.max()+1e-6)))

            group_msk = torch.from_numpy(group_msk).float().unsqueeze(0).unsqueeze(0).cuda()
            # group_msk = F.interpolate(group_msk, size=(x.shape[2]//4, x.shape[3]//4), mode='nearest')
            # print(group_msk.shape)
            # torchvision.utils.save_image(group_msk, './group_msk_bef.png')
            if group_msk.sum() == 0:
                group_msk = torch.ones_like(group_msk)
            else:
                group_msk[group_msk!=0] = 1
            # torchvision.utils.save_image(group_msk, './group_msk_aft.png')
            
            # group_msk[group_msk!=0] = 1

            # input the mask to the codec for ssic bitstream generation
            res = self.model.forward_partial(x, noisy=False, msk=group_msk)
            # res = self.model(x, noisy=False)
            # torchvision.utils.save_image(res['x_hat'].squeeze(0).unsqueeze(1), './fea_hat.png')

            # raise NotImplementedError

        else:
            res = self.model(x, noisy=False)
        x_hat = res['x_hat']
        self.inference_bits += res['bits']['y'] + res['bits']['z']
        
        bottom_up_features = self.res2toreses(x_hat)

        # get p-layered features
        backbone = self.task_model.model.backbone
        # backbone._out_features : ['p2', 'p3', 'p4', 'p5', 'p6']
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        assert len(backbone._out_features) == len(results)
        return {f: res for f, res in zip(backbone._out_features, results)}

    def get_group_msk(self, h, w, instances_list, mode='bbox'):
        '''
            copied from ssic/ssic.py
        '''
        import cv2
        ss_expand_bbox_factor = 16
        det_threshold = 0.05
        ss_det_human = False
        ss_block_size = 32
        padding_factor = 32

        msk_objects = np.zeros((h, w), dtype=np.uint8)
        msks = []

        if mode == 'bbox':
            for instance in instances_list:
                x1, y1, x2, y2 = instance['bbox']
                if ss_expand_bbox_factor:
                    factor = ss_expand_bbox_factor
                    x1 = np.clip(x1-factor, 0, w)   # 确保bbox的坐标不会超出边界，同时将它们向外扩展factor像素
                    x2 = np.clip(x2+factor, 0, w)
                    y1 = np.clip(y1-factor, 0, h)
                    y2 = np.clip(y2+factor, 0, h)
                score = instance['score'] 
                if score < det_threshold:
                    continue
                if ss_det_human: 
                    if instance['category_id'] != 1:
                        continue

                msk_objects[int(y1):int(y2), int(x1):int(x2)] = 1
                x1 = (np.floor(x1 / ss_block_size) *ss_block_size).astype(np.int32)
                x2 = (np.ceil(x2 / ss_block_size) *ss_block_size).astype(np.int32)
                y1 = (np.floor(y1 / ss_block_size) *ss_block_size).astype(np.int32)
                y2 = (np.ceil(y2 / ss_block_size) *ss_block_size).astype(np.int32)
                instance_msk = torch.zeros((h, w))
                instance_msk[y1:y2, x1:x2] = 1
                msks.append(instance_msk.unsqueeze(0))
        else:
            raise NotImplementedError

        # generate the msk of groups.
        dh = padding_factor * math.ceil(h/padding_factor) - h
        dw = padding_factor * math.ceil(w/padding_factor) - w
        
        if len(msks) > 0:
            msks = torch.cat(msks, dim=0).float()       # 93x427x640

            # pad the msk
            msks = F.pad(msks, (0, dw, 0, dh))  # 93x448x640
            
            # Process the msk: If the block contains more than 1 pixels that belong to a object.
            # Set it as 1.
            msks = msks.sum(dim=0)
            # This is faster, which aviod loops.
            msks = einops.rearrange(msks, 
                                    '(h h_grid) (w w_grid) -> h w (h_grid w_grid)', 
                                    h_grid=ss_block_size, w_grid=ss_block_size)
            msks = msks.sum(dim=-1)
            new_msk = msks
            new_msk[new_msk!=0] = 1

            new_msk = new_msk.numpy().astype(np.uint8)*255

            # generate the connected components
            # 识别和分离不同的物体实例
            num_labels, group_msk, stats, centroids = cv2.connectedComponentsWithStats(
                new_msk, connectivity=4)    # group_msk is the original labels
            # use a larger bounding box to deal with the overlapped situation
            if 'group' not in args.model:
                new_msk = np.zeros_like(group_msk)
                for i in range(1, len(stats)):
                    bbox = stats[i]
                    xb, yb, wb, hb, _ = bbox
                    new_msk[yb:yb+hb, xb:xb+wb] = 1

            # 将 new_msk 调整回原始图像的大小，使用最近邻插值法，然后将像素值缩放到0-1范围，生成新的分割掩码
            new_msk = cv2.resize(new_msk, (w+dw, h+dh), interpolation=cv2.INTER_NEAREST)[:h, :w] / 255
            new_msk[new_msk!=0] = 1
            w_group_msk = int((w+dw)/16)
            h_group_msk = int((h+dh)/16)
            group_msk = cv2.resize(group_msk, (w_group_msk, h_group_msk), interpolation=cv2.INTER_NEAREST)
            if group_msk.sum() == int(group_msk.shape[0]*group_msk.shape[1]):
                group_msk *= 0
            return new_msk.astype(np.uint8), group_msk.astype(np.uint8), msk_objects
        else:   # 在之前的循环中没有检测到物体实例
            h_group_msk = int((h+dh)/16)
            w_group_msk = int((w+dw)/16)
            if ss_det_human:
                # TODO: optimize this temperary solution, which is only compress the left-top block.
                new_msk = np.zeros((h, w), dtype=np.uint8)
                new_msk[:ss_block_size, :ss_block_size] = 1
                group_msk = np.zeros((h_group_msk, w_group_msk), dtype=np.uint8)
                group_msk[:ss_block_size // 16, :ss_block_size // 16] = 1
                msk_objects = np.zeros((h, w), dtype=np.uint8)
                msk_objects[:ss_block_size, :ss_block_size] = 1
                return new_msk, group_msk, msk_objects
            else:
                group_msk = np.zeros((h_group_msk, w_group_msk), dtype=np.uint8)
                return np.ones((h, w), dtype=np.uint8), group_msk, np.ones((h, w), dtype=np.uint8)


def main():
    if args.task_driven:    # no
        trainer = TaskDrivenTrainer()
    elif args.task_driven_bit_allocation:    # no
        trainer = TaskDrivenBitAllocation()
    elif args.predicted_task_driven_bit_allocation:    # no
        trainer = PredictedTaskDrivenBitAllocation()
    elif args.feature_compression:    # no
        trainer = FeatureCompressionTrainer()
        if args.fc_inference_dir:   # feature compression inference dir
            trainer.inference()
            return 
    elif args.vbr_training:     # no
        print('VBR training.')
        trainer = VBRTrainer()
    else:
        trainer = DefaultTrainer()
        if args.freeze_transform:
            trainer.log('-------------------------------\n')
            trainer.log('Freeze parameters for transform! \n')
            trainer.log('-------------------------------\n')
            trainer.model.freeze_transform()
            trainer.show_learnable_params()  

    ## ddp training.
    if args.multiprocessing_distributed and torch.cuda.device_count() > 1:
        if args.task_driven or args.task_driven_bit_allocation or args.feature_compression:
            raise NotImplementedError
        trainer.ddp_training()
        trainer.log('Use DDP training.\n\n')
    else:
        trainer.log('Training with a single process on 1 GPUs.\n\n')

    if args.eval_only: 
        if args.task_driven or args.task_driven_bit_allocation or args.predicted_task_driven_bit_allocation:
            log = trainer.eval()
        else:
            log = trainer.eval(eval_fg=args.only_rec_fg)
    else:
        trainer.train()
    return 


if __name__ == '__main__':
    main()
