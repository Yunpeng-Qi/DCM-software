import sys
from tokenize import group
import matplotlib.pyplot as plt
sys.path.append('.')
from PIL import Image
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
from models.MIK_codecs import RDT_CheckerCube_base, E0ToE3CompressModel, Adapter
torch.backends.cudnn.benchmark=True
torch.set_num_threads(1)
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.MIK_codecs import build_feature_coding

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # deug时设置
# Common Setting
parser = argparse.ArgumentParser()
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--model', type=str, default='ours_meanscalehyper')
parser.add_argument('--total-iter', type=int, default=2000000)
parser.add_argument('--saving-iteration', type=int, default=0)
parser.add_argument('--eval-interval', type=int, default=10000)
parser.add_argument('--saving-interval', type=int, default=200000)
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
parser.add_argument('--save-dir', '-s', default='./logs/cube_train', type=str, help='directory for saving')
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
                    default='./packages/Model/Detector_model/model_final_68b088.pkl')
parser.add_argument('--td-det-cfg-file', type=str, 
                    default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
parser.add_argument('--td-det-threshold', type=float, default=0.05)
parser.add_argument('--td-lmbda-feat', type=int, default=32)
parser.add_argument('--td-feat-ratio', type=float, nargs='+', 
                    default=[1,2,3,4], help='the ratio of different layers.')

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
parser.add_argument('--MIK_path', type=str, default='./packages/Model/MIKcodec_model/ckp_step48000_ete_model_bpp4.pth', help='')

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

def prepadding(x, factor=64):
    _, _, h_ori, w_ori = x.shape
    dh = factor * math.ceil(h_ori / factor) - h_ori
    dw = factor * math.ceil(w_ori / factor) - w_ori
    x = F.pad(x, (0, dw, 0, dh))
    return x

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

        return image, scale_factor, file_name

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

    def get_detectron2_scale_factor(self, x): # detectron2 大小范围[800, 1333]
        _, _, h_ori, w_ori = x.shape
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

class TaskDrivenTrainer():
    def __init__(self):
        args.td_feat_ratio = [each/sum(args.td_feat_ratio) for each in args.td_feat_ratio]  # 1/5
        self.build_logger()
        self.feature_extractor, self.adapter = self.build_extractor_adapter(args.MIK_path)
        self.model = self.build_model(args.MIK_path)
        self.task_model = self.build_task_model()   # detectron2 PanopticFPN backbone 
        # freeze task model
        for params in self.task_model.parameters():
            params.requires_grad = False
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.eval_ps = self.load_eval_ps()

        self.pixel_mean = torch.tensor([103.530, 116.280, 123.675])
        self.pixel_std = torch.tensor([57.375, 57.120, 58.395])

        # resume
        self.best_rdo = float('+inf')
        if args.resume:  # default = ''
            self.resume(args.resume)
        else:
            self.start_iteration = 1
            self.best_rdo = float('+inf')
        
        # build qmap generator and loader    
        self.crop_size = 256
        train_dataset = CustomMSCOCO('/data/qiyp/detectron2/datasets/coco/train2017',
                                    crop_size=self.crop_size)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                       num_workers=args.num_workers, pin_memory=True)
        eval_dataset = CustomMSCOCO('/data/qiyp/CompressTask/detectron2/datasets/coco/val2017', 
                                    crop_size=self.crop_size, subset_ratio=0.01, split='val')
        self.eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        save_dataset = CustomMSCOCO('/data/qiyp/CompressTask/detectron2/datasets/coco/val2017', 
                                    crop_size=self.crop_size, subset_ratio=1, split='val')
        self.save_loader = DataLoader(save_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    def build_logger(self):
        args.save_dir = os.path.join(
            args.save_dir, 
            args.model + '_checker_cube_{}'.format('_'.join(args.metric)) + str(args.lmbda))
        
        os.makedirs(args.save_dir, exist_ok=True)
        self.p_log = os.path.join(
            args.save_dir,
            '{}.txt'.format(str(datetime.datetime.now()).replace(':', '-')[:-7]))
        write_log(self.p_log, str(args).replace(', ', ',\n\t') + '\n')

    def log(self, content):
        return write_log(self.p_log, content)
    
    def build_extractor_adapter(self, p_ckpt):
        self.log('building extrctor & adapter ... ')
        ckpt = torch.load(p_ckpt, map_location='cpu')
        extractor = E0ToE3CompressModel()
        adapter = Adapter(128, 256)
        extractor.load_state_dict(ckpt['feature_extractor'], strict=False)
        adapter.load_state_dict(ckpt['feature_adapter'], strict=False)
        extractor.cuda()
        adapter.cuda()
        return extractor, adapter

    def build_model(self, p_ckpt):
        # args.N, args.M = 192, 320
        self.log('building feature compression model ... ')
        args.N = 128
        # model = RDT_CheckerCube_base(args.N)
        model = build_feature_coding(quality=3)
        ckpt = torch.load(p_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['feature_coding'], strict=False)
        model.cuda()
        self.log('\n'+str(model)+'\n\n')
        return model
    
    def build_task_model(self):
        self.log('building task model ... ')

        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg_file = args.td_det_cfg_file # COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.td_det_threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = args.td_det_weights
        detector = DefaultPredictor(cfg)
        model = detector.model
        model.cuda()

        self.log('building task done.\n')

        return model
    
    def build_optimizer(self):
        optimizer = optim.Adam(
            [
                {'params': self.feature_extractor.parameters()},
                {'params': self.model.parameters()},
                {'params': self.adapter.parameters()},
                {'params': self.task_model.parameters()}
            ], 
            lr=args.lr
        )
        return optimizer
    
    def build_scheduler(self):
        assert args.scheduler in ['multistep', 'cos']   # args.scheduler: default='multistep'
        if args.scheduler == 'multistep':
            scheduler = MultiStepLR(
                self.optimizer, 
                milestones=args.multistep_milestones, 
                gamma=args.multistep_gamma)
        elif args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(self.optimizer, args.total_iter)
        else:
            raise NotImplementedError    
        return scheduler

    def load_eval_ps(self):
        eval_ps = sorted(glob.glob(os.path.join('/data/qiyp/CompressTask/test_img', '*.jpg')))
        # eval_ps = sorted(glob.glob(os.path.join('/data/datasets/kodak/images/', '*.png')))
        return eval_ps

    def train(self):
        self.eval(train_iteration=self.start_iteration)
        print("## ============ Statr Training =========== ##")
        self.model.train()
        self.task_model.train()
        self.feature_extractor.train()
        self.adapter.train()
        current_iter = self.start_iteration
        print("current_iter=", current_iter)
        n_epochs = 1 + (args.total_iter - current_iter) // len(self.train_loader)
        for epoch in range(n_epochs):
            for batch_idx, data in enumerate(tqdm(self.train_loader, desc=f"epoch: {epoch+1}", unit="epoch")):
                if current_iter >= args.total_iter:
                    break
                else:
                    current_iter += 1
                
                x, scale_factor, _ = data
                x = x.cuda()
                x0 = x
                x = prepadding(x)
                self.save_tensor(x[0:1, :, :, :], './vis', 'x_vis.jpg', "image")
                # rescale images to the format of detectron2
                x = self.rescale_to_detectron_distribution(x, scale_factor)
                x0 = x
                self.save_tensor(x[0:1, :, :, :], './vis', 'x_rescale.jpg', "image")
                feats = self.get_multiscale_feat(self.task_model, x*255)
                for k, v in feats.items():
                    feats[k] = v.detach()
                self.vis_feature_map(feats['res2'][0:1, :, :, :], './vis', 'res2_ori.jpg')    # [1, 256, 272, 208]
                self.vis_feature_map(feats['res3'][0:1, :, :, :], './vis', 'res3_ori.jpg')
                self.vis_feature_map(feats['res4'][0:1, :, :, :], './vis', 'res4_ori.jpg')
                self.vis_feature_map(feats['res5'][0:1, :, :, :], './vis', 'res5_ori.jpg')
                res2 = self.feature_extractor(x)
                self.vis_feature_map(res2[0:1, :, :, :], './vis', 'res2_f_extractor.jpg')
                # res2_hat, bits, side_bits = self.model(res2, noisy=True)
                res_hat = self.model(res2)
                res2_hat_f = res_hat['x_hat']
                self.vis_feature_map(res2_hat_f[0:1, :, :, :], './vis', 'res2_f_coding.jpg')
                bits = - torch.sum(res_hat["likelihoods"]["y"]) / (-math.log(2))
                side_bits = - torch.sum(res_hat["likelihoods"]["z"]) / (-math.log(2))
                res2_hat = self.adapter(res2_hat_f) # channel: 128 ——>256
                self.vis_feature_map(res2_hat[0:1, :, :, :], './vis', 'res2_f_adapter.jpg')
                rec_feats = {}
                rec_feats["res2"] = res2_hat
                rec_feats["res3"] = self.task_model.backbone.bottom_up.res3(rec_feats["res2"])
                rec_feats["res4"] = self.task_model.backbone.bottom_up.res4(rec_feats["res3"])
                rec_feats["res5"] = self.task_model.backbone.bottom_up.res5(rec_feats["res4"])
                for k, v in rec_feats.items():
                    rec_feats[k] = v.detach()
                self.vis_feature_map(rec_feats['res3'][0:1, :, :, :], './vis', 'rec_res3.jpg')
                self.vis_feature_map(rec_feats['res4'][0:1, :, :, :], './vis', 'rec_res4.jpg')
                self.vis_feature_map(rec_feats['res5'][0:1, :, :, :], './vis', 'rec_res5.jpg')

                # calculate feature level loss
                dist_loss = []
                for k, ratio in zip(feats.keys(), args.td_feat_ratio):  # what is td_feat_ratio
                    feat = feats[k]
                    feat_hat = rec_feats[k]
                    dist_loss.append(nn.MSELoss()(feat, feat_hat) * ratio)

                loss = {}
                lmbda = args.lmbda    # default: 8
                loss['dist_loss'] = sum(dist_loss)
                loss = self.calculate_bpp_loss(x0, bits, side_bits, loss)
                loss['loss'] = lmbda * loss['dist_loss'] + loss['bpp_loss']

                # if current_iter < (args.total_iter * 0.9):
                    # eval_interval = args.eval_interval
                # else:
                    # eval_interval = args.eval_interval // 5
                eval_interval = args.eval_interval
                
                if current_iter % eval_interval == 0:
                    print('lr: {}'.format(self.optimizer.param_groups[0]['lr']))
                    rdo = self.eval(epoch, current_iter)
                    print("Current Rdo is ", rdo, "Best Rdo is ", self.best_rdo)
                    ## save best model
                    if rdo <= self.best_rdo:
                        self.best_rdo = rdo
                        self.log('Best model. Rdo is {:.4f} and save model to {}\n\n'.format(rdo, args.save_dir))
                        ckpt_name = 'best_' + str(args.lmbda) + '_.pth'
                        self.save_ckpt(current_iter, ckpt_name)
            
                # compulsively save the checkpoint
                if args.saving_interval:
                    if current_iter % args.saving_interval == 0:
                        self.log('Save model. Rdo is {:.4f} and save model to {}\n\n'.format(
                            rdo, args.save_dir))
                        ckpt_name = str(current_iter) + '_' + str(args.lmbda) + '_.pth'
                        self.save_ckpt(current_iter, ckpt_name)

                # if current_iter >= args.total_iter:
                #     log = self.eval_save(iteration=current_iter)
                # optimize
                self.optimizer.zero_grad()
                loss['loss'].backward()
                self.optimizer.step()
                self.scheduler.step()        

    def eval(self, epoch=None, train_iteration=None):
        self.model.eval()
        self.task_model.eval()
        self.feature_extractor.eval()
        self.adapter.eval()
        dist_loss_sum, bpp_sum, loss_sum = 0, 0, 0
        with torch.no_grad():
            for bat_idx, data in enumerate(tqdm(self.eval_loader)):
                x, scale_factor, _ = data
                x = x.cuda()
                self.save_tensor(x, './vis', 'x_vis.jpg', "image")
                x0 = x
                # x = prepadding(x)

                # rescale images to the format of detectron2
                # x = self.rescale_to_detectron_distribution(x, scale_factor)
                x = F.interpolate(x, scale_factor=float(scale_factor), mode='bilinear')
                x = prepadding(x, factor=64) # [1088, 832]
                self.save_tensor(x, './vis', 'x1_vis.jpg', "image")
                feats = self.get_multiscale_feat(self.task_model, x)
                for k, v in feats.items():
                    feats[k] = v.detach()
                self.vis_feature_map(feats['res2'], './vis', 'res2_ori.jpg')    # [1, 256, 272, 208]
                self.vis_feature_map(feats['res3'], './vis', 'res3_ori.jpg')
                self.vis_feature_map(feats['res4'], './vis', 'res4_ori.jpg')
                self.vis_feature_map(feats['res5'], './vis', 'res5_ori.jpg')
                res2 = self.feature_extractor(x) # [1, 128, 272, 208]
                self.vis_feature_map(res2, './vis', 'res2_f_extractor.jpg')
                # res2_hat, bits, side_bits = self.model(res2, noisy=False)
                res_hat = self.model(res2)
                res2_hat_f = res_hat['x_hat']
                self.vis_feature_map(res2_hat_f, './vis', 'res2_f_coding_rec.jpg')
                bits = - torch.sum(torch.log2(res_hat["likelihoods"]["y"]), dim=(1,2,3))
                side_bits = - torch.sum(torch.log2(res_hat["likelihoods"]["z"]), dim=(1,2,3))
                # bits = - torch.sum(res_hat["likelihoods"]["y"]) / (-math.log(2))
                # side_bits = - torch.sum(res_hat["likelihoods"]["z"]) / (-math.log(2))
                res2_hat = self.adapter(res2_hat_f)
                # res2_hat = self.adapter(res2_hat)
                self.vis_feature_map(res2_hat, './vis', 'res2_f_adapter.jpg')
                rec_feats = {}
                rec_feats['res2'] = res2_hat
                rec_feats['res3'] = self.task_model.backbone.bottom_up.res3(rec_feats['res2'])
                rec_feats['res4'] = self.task_model.backbone.bottom_up.res4(rec_feats['res3'])
                rec_feats['res5'] = self.task_model.backbone.bottom_up.res5(rec_feats['res4'])
                for k, v in rec_feats.items():
                    rec_feats[k] = v.detach()
                self.vis_feature_map(rec_feats['res3'], './vis', 'rec_res3.jpg')
                self.vis_feature_map(rec_feats['res4'], './vis', 'rec_res4.jpg')
                self.vis_feature_map(rec_feats['res5'], './vis', 'rec_res5.jpg')
                # calculate feature level loss
                dist_loss = []
                for k, ratio in zip(feats.keys(), args.td_feat_ratio):  # what is td_feat_ratio
                    feat = feats[k]
                    feat_hat = rec_feats[k]
                    dist_loss.append(nn.MSELoss()(feat, feat_hat) * ratio)

                loss = {}
                lmbda = args.lmbda    # default: 8
                loss['dist_loss'] = sum(dist_loss)
                # loss = self.calculate_bpp_loss(res_hat['x_hat'], res_hat, loss)
                loss = self.calculate_bpp_loss(x0, bits, side_bits, loss)
                loss['loss'] = lmbda * loss['dist_loss'] + loss['bpp_loss']
                dist_loss_sum += loss['dist_loss'].item()
                bpp_sum += loss['bpp_loss'].item()
                loss_sum += loss['loss'].item()
                
            self.log('Evaluation: [Epoch-Iteration/All]:[{}-{}/{}]\tdist_loss: {:.4f}\tbpp_loss: {:.4f}\tloss: {:.4f}\n'.format(
                    epoch, train_iteration, len(self.train_loader),
                    dist_loss_sum / len(self.eval_loader), 
                    bpp_sum / len(self.eval_loader), 
                    loss_sum / len(self.eval_loader)))

        self.model.train()
        self.task_model.train()
        self.adapter.train()
        self.feature_extractor.train()
        return loss_sum / len(self.eval_loader)

    def rescale_to_detectron_distribution(self, x, scale_factor):
        x_tmp = []
        for idx, sca in enumerate(scale_factor):
            # rescale
            tmpx = F.interpolate(x[idx].unsqueeze(0), scale_factor=float(sca), mode='bilinear')

            h_start = np.random.randint(0, tmpx.shape[2]-self.crop_size)    # retun a random int
            w_start = np.random.randint(0, tmpx.shape[3]-self.crop_size)
            x_tmp.append(tmpx[:, :, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size])
        x = torch.cat(x_tmp, dim=0)
        return x

    def save_ckpt(self, iteration, name=None):
        if name:
            filename = name
        else:
            filename = 'best.pth'
        try:
            self.model.fix_tables() ## fix cdf tables
        except:
            self.log('error occured when self.model.fix_tables()')

        torch.save({
            'best_rdo': self.best_rdo,
            'iteration': iteration,
            'feature_extractor': self.feature_extractor.state_dict(),
            'feature_coding': self.model.state_dict(),
            'feature_adapter': self.adapter.state_dict()
        }, os.path.join(args.save_dir, filename))

    def resume(self, p_ckpt):
        ckpt = torch.load(p_ckpt, map_location='cpu')
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

        self.feature_extractor.load_state_dict(ckpt['feature_extractor'], strict=False)
        msg = self.model.load_state_dict(ckpt['feature_coding'], strict=False)
        self.adapter.load_state_dict(ckpt['feature_adapter'], strict=False)
        # msg_task = self.task_model.load_state_dict(ckpt['parameters'], strict=False)
        self.log('resume the ckpt from : {} and the message is {}\n'.format(
            p_ckpt, msg
        ))
        
    def show_learnable_params(self):
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.log("Parameters to be updated: ")
        for each in enabled:
            self.log('\t{}\n'.format(str(each)))
        self.log('\n')

    def get_feat(self, task_model, x, feat_type='stem'):   
        '''
            task model: a task model in detectron2 —— GeneralizedRCNN
            x: B x C x H x W, range in [0, 255]
            preprocess including normalize, 
            remember that the channel order of detectron2 is 'BGR'
            'Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.'
        '''
        x = x.flip(dims=[1])    # why?

        b, _, h, w = x.shape
        pixel_mean = einops.repeat(self.pixel_mean, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        pixel_std = einops.repeat(self.pixel_std, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        x = (x - pixel_mean) / pixel_std    # normalize

        if feat_type == 'stem':
            feat = task_model.backbone.bottom_up.stem(x)
        elif feat_type == 'p':
            feat = task_model.backbone(x)
        else:
            raise NotImplementedError

        return feat
    
    def get_multiscale_feat(self, task_model, x):
        """
        task_model: GeneralizedRCNN, task_model.backbone.bottom_up: ResNet
        task_model.backbone: FPN, in_feature:['res2', 'res3', 'res4', 'res5'](the out put of task_model.backbone.bottom_up)
        task_model.proposal_generator: RPN, in_feature:['p2', 'p3', 'p4', 'p5', 'p6'](the output of task_model.backbone)
        """

        feats = {}
        x = x.flip(dims=[1])
        b, _, h, w = x.shape
        pixel_mean = einops.repeat(self.pixel_mean, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        pixel_std = einops.repeat(self.pixel_std, 'c -> b c h w', 
                                    b=b, h=h, w=w).to(x.device)
        x = (x - pixel_mean) / pixel_std
        # stem = task_model.backbone.bottom_up.stem(x)
        # feats["res2"] = task_model.backbone.bottom_up.res2(stem)
        # feats["res3"] = task_model.backbone.bottom_up.res3(feats["res2"])
        # feats["res4"] = task_model.backbone.bottom_up.res4(feats["res3"])
        # feats["res5"] = task_model.backbone.bottom_up.res5(feats["res4"])
        feats = task_model.backbone.bottom_up(x)

        return feats
    
    def calculate_bpp_loss(self, x, bits, side_bits, loss):
    # def calculate_bpp_loss(self, x, res_hat, loss):
        b, _, h, w = x.shape
        n_pixels = b * h * w
        # loss['bpp_y'] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * n_pixels))
        #     for likelihoods in res_hat["likelihoods"]["y"]
        # )
        # loss['bpp_side'] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * n_pixels))
        #     for likelihoods in res_hat["likelihoods"]["z"]
        # )
        loss['bpp_y'] = bits / n_pixels
        loss['bpp_side'] = side_bits / n_pixels
        loss['bpp_loss'] = loss['bpp_y'] + loss['bpp_side']
        return loss

    def update_log(self, log, loss, psnr, msssim):
        log['bpp'] += loss['bpp_loss'].item()
        log['bpp_y'] += loss['bpp_y'].item()
        if 'bpp_side' in loss.keys():
            log['bpp_side'] += loss['bpp_side'].item()
        log['psnr'] += psnr.item()
        log['ms_ssim'] += msssim
        # log['acc'] += acc
        return log

    def display_log(self, log, iter=None, n_blankline=1):
        if iter:
            self.log('iteration: {}\t'.format(iter))
        for k,v in log.items():
            self.log('{}: {:>6.5f}  '.format(k, v))
        for i in range(n_blankline+1):
            self.log('\n')
    
    def save_tensor(self, img: torch.Tensor, path, filename, type):
        assert (len(img.shape) == 4 and img.shape[0] == 1)
        img = img.clone().detach()
        img = img.to(torch.device('cpu'))
        # img_dir = path + str(args.lmbda)
        img_dir = os.path.join(path, str(args.lmbda))
        if os.path.isdir(img_dir) is not True:
            os.makedirs(img_dir)
        img_name = img_dir +'/'+ filename
        if type == "image":
            # torchvision.utils.save_image(img/255.0, img_name)
            torchvision.utils.save_image(img, img_name)
        if type == "qmap":
            torchvision.utils.save_image(img, img_name)
    
    def save_img(self, img: torch.Tensor, path, input_p, q_factor):
        assert (len(img.shape) == 4 and img.shape[0] == 1)
        img = img.clone().detach()
        img = img.to(torch.device('cpu'))
        dir = path + str(q_factor)
        if os.path.isdir(dir) is not True:
            os.makedirs(dir)
        end = '/'
        # path = path[0]
        img_name = dir + str(input_p[input_p.rfind(end):])
        torchvision.utils.save_image(img, img_name)
    
    #  可视化特征图
    def vis_feature_map(self, f_map, path, filename):
        '''
        :param f_map: [1, dims, H, W]
        :return: None
        '''
        import imageio
        assert (len(f_map.shape) == 4 and f_map.shape[0] == 1)
        
        # 在通道维度上求平均
        average_image = torch.mean(f_map, dim=1, keepdim=True)
        average_image_np = average_image[0].detach().cpu().numpy()
        
        # 创建图像目录
        img_dir = os.path.join(path, str(args.lmbda))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        # 保存平均后的灰度图像
        img_name = os.path.join(img_dir, filename)
        imageio.imwrite(img_name, (average_image_np[0] * 255).astype(np.uint8))


def main():
    trainer = TaskDrivenTrainer()
    # trainer.show_learnable_params()  
    if args.eval_only: 
        # log = trainer.inference()
        log = trainer.test()
    else:
        trainer.train()

    return 


if __name__ == "__main__":
    # with torch.cuda.device(1):
    main()
