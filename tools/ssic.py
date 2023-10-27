import argparse
import datetime
import glob
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

sys.path.append('.')

import cv2
import einops
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchac
import torchvision
import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.utils.mylib import (generate_local_region_msk, load_coco_labels,
                                load_img, pack_bool, pack_string, pack_strings,
                                pack_uints, pack_ushorts, parse_instance,
                                quality2lambda, unpack_bool, unpack_string,
                                unpack_strings, unpack_uints, unpack_ushorts,
                                write_log)
from models.utils.pytorch_msssim import MSSSIMLoss, ms_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

models = {
    "ours_spatialvbr_imgcomnet": models.CLIC_based_codecs.Spatially_adaptive_ImgComNet,
}

coco_map_dict = load_coco_labels()

parser = argparse.ArgumentParser(description="Compress image to bit-stream")
# basic setting.
parser.add_argument('--mode', type=str, choices=['compress', 'decompress'])
parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
parser.add_argument('--save', '-s', default='./logs/default/ssic/debug', type=str, help='directory for saving')
parser.add_argument('--texture-codec-idx', type=int, default=0, choices=[0, 1],
                    help='0 (default): learned based codec' \
                         '1: vvc codec')

# semantic strucutred image coding setting.
parser.add_argument('--ss-enabled-flag', action='store_true')
parser.add_argument('--ss-block-size', type=int, default=64,
                    help='the size of minimal basic units that constructing the image.')
parser.add_argument('--ss-fg-only', action='store_true', 
                    help='only compress groups with foreground.')
parser.add_argument('--ss-det-human', action='store_true', 
                    help='only detect human.')
parser.add_argument('--ss-w-bboxes-flag', action='store_true',
                    help='compress bounding boxes or not.')
parser.add_argument('--ss-expand-bbox-factor', type=int, default=0,
                    help='expand the bounding box to generate the group mask.')

# codec setting
parser.add_argument("--model", type=str, default="ours_meanscalehyper", help="Model")
parser.add_argument("--transform-channels", type=int, nargs='+',
                    default=[128, 128, 128, 192],help="Transform channels.")
parser.add_argument("--hyper-channels", type=int, nargs='+',
                    default=None,help="Transform channels.")
parser.add_argument('--resume', type=str, default='logs/default/ours_meanscalehyper/2048/best.pth',
                    help='Checkpoint path')

# vbr setting 
parser.add_argument('--qmap-factor', type=float, default=0.5)
parser.add_argument('--qmap-dir', type=str, default='')

# vvc setting
parser.add_argument('--vvc-qp', type=int, default=42)

# inputs, bins, and outputs
parser.add_argument("--input-files-dir", type=str, default="",
                    help="Input image path")
parser.add_argument('--byte-stream-dir', type=str, default='workspace/bitstream')
parser.add_argument("--output-files-dir", type=str, default="workspace/rec",
                    help="Output binary file")

# detection setting
parser.add_argument('--det-weights', type=str, 
                    default='./../detectron2/ckpts/model_final_68b088.pkl')
parser.add_argument('--det-cfg-file', type=str, 
                    default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
parser.add_argument('--det-threshold', type=float, default=0.05)
parser.add_argument('--det-save-result', type=str, default='')
parser.add_argument('--det-load-result', type=str, default='')
args = parser.parse_args()


class SemanticStructuredImageCoding():
    def __init__(self):
        self.build_logger()
        # build codec
        self.codec = self.build_codec()
        self.max_group_numbers = 64
        self.padding_factor = 64

        # build detector
        if args.ss_enabled_flag:
            if args.mode == 'compress':
                if not args.det_load_result:
                    self.detector = self.build_detector()
                else:
                    self.detector = None

    def build_logger(self):
        os.makedirs(args.save, exist_ok=True)
        self.p_log = os.path.join(args.save,
            '{}.txt'.format(str(datetime.datetime.now()).replace(':', '-')[:-7]))
        write_log(self.p_log, str(args).replace(', ', ',\n\t') + '\n')
        
    def log(self, content):
        return write_log(self.p_log, content)

    def display_log(self, log, n_blankline=1):
        for k,v in log.items():
            self.log('{}: {:.4f}, '.format(k, v))
        for i in range(n_blankline+1):
            self.log('\n')

    def build_codec(self):
        if args.device == 'cpu':
            args.device = torch.device("cpu")
        elif args.device == 'gpu':
            assert args.texture_codec_idx != 1  # 0: NIC, 1: VVC
            args.device = torch.device("cuda")
        else:
            raise NotImplementedError

        if args.texture_codec_idx == 0:
            codec = self.build_learned_based_codec()
        if args.texture_codec_idx == 1:
            raise NotImplementedError
        return codec

    def build_learned_based_codec(self):
        if 'ours' in args.model:
            if args.model == 'ours_spatialvbr_imgcomnet':
                args.N, args.M = 192, 320
                model = models['ours_spatialvbr_imgcomnet'](args)
            else:
                model = models[args.model](args.transform_channels, args.hyper_channels)
        elif 'groupvit' in args.model:
            model = models[args.model](
                args.hyper_channels, window_size=args.groupvit_window_size)
        else:
            model = models[args.model](args.N, args.M)

        ckpt = torch.load(args.resume, map_location='cpu')

        # load the cdf_shape first, then initialize tables with cdf_shape,
        # finally load the _cdf, _cdf_offset, _cdf_length, e.t.c.
        model.load_state_dict(ckpt['parameters'], strict=False)
        model.init_tables()
        msg = model.load_state_dict(ckpt['parameters'], strict=True)

        self.log('resume the ckpt from : {} and the message is {}\n'.format(
            args.resume, msg))
        model.to(args.device).eval()

        return model

    def build_detector(self):
        cfg = get_cfg()
        cfg_file = args.det_cfg_file
        cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = args.det_weights
        detector = DefaultPredictor(cfg)
        return detector

    def compress(self, input_p, bin_p, group_msk=None, instances_list=None):
        # * compress header.
        x, hx, wx = load_img(input_p, padding=True, factor=self.padding_factor)
        _,c,h,w = x.shape
        if c != 3:
            self.log('{} with invalid shape of: {}, ' \
                    'take the first 3 channels to be an RGB image.\n'.format(
                input_p, x.shape))
            x = x[:,:3,...]
                
        x = x.to(args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()

        # * compress header
        bs_header = self.enc_header(
            hx, wx, h, w, args.ss_enabled_flag, group_msk, 
            instances_list, args.ss_w_bboxes_flag, args.texture_codec_idx)
        byte_stream = bs_header
        # self.log('header size: {} bits.\n'.format(len(bs_header) * 8))

        # * compress texture.
        if args.qmap_dir:
            assert args.texture_codec_idx == 0
            p_qmap = os.path.join(args.qmap_dir, input_p.split('/')[-1][:-4]+'.png')

        if args.ss_enabled_flag:
            byte_stream, bs_texture = self.enc_ssic_texture(
                byte_stream, x, group_msk, args.texture_codec_idx, p_qmap)
        else:
            byte_stream, bs_texture = self.enc_full_texture(
                byte_stream, x, args.texture_codec_idx, p_qmap)

        with open(str(bin_p), "wb") as f:
            f.write(byte_stream)

        with open(str(bin_p), "rb") as f:
            string = f.read()
            f.close()
        # self.log('file size: {} bits.\n'.format(len(string) * 8))
        # calcualte bit per pixel.
        num_pixels = hx * wx
        bpp_header = len(bs_header) * 8. / num_pixels
        bpp_texture = len(bs_texture) * 8. / num_pixels
        bpp = len(string) * 8. / num_pixels
        self.log('Encoded: {} -> {}. ' \
            'Bpp(header): {:.4f}, Bpp(texture): {:.4f}, Bpp(full): {:.4f}\n'.format(
                input_p, bin_p, bpp_header, bpp_texture, bpp))

    def decompress(self, bin_p, rec_p, groups_tobe_decode=None):
        with open(str(bin_p), "rb") as f:
            byte_stream = f.read()
        # print(bin_p, len(byte_stream) * 8)
        # raise
        # * decompress header.
        header, byte_stream = self.dec_header(byte_stream)
        # print(len(header), len(byte_stream))
        # print(len(byte_stream))
        # raise
        # print(header['group_msk'].shape)
        # torchvision.utils.save_image(header['group_msk'] / header['group_msk'].max(), 'groupmsk.png')
        # raise

        # * decompress texture.
        shape = (header['hx'], header['wx'])
        if args.ss_enabled_flag:
            x_hat = self.dec_ssic_texture(
                shape, byte_stream, 
                header['group_msk'], groups_tobe_decode, header['texture_codec_idx'])
        else:
            x_hat = self.dec_full_texture(
                shape, byte_stream,
                header['texture_codec_idx'])

        ## save image
        # torchvision.utils.save_image(x_hat, 'x_hat.png')
        # raise
        x_hat = x_hat[:, :, :shape[0], :shape[1]].clamp(0, 1).mul(255).round()
        x_hat = x_hat.squeeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        x_hat = Image.fromarray(x_hat)
        x_hat.save(rec_p)

    def enc_header(
        self, hx, wx, h, w, ss_enabled_flag, group_msk, 
        instances_list, ss_w_bboxes_flag, texture_codec_idx):
        # h, w
        bs_hx_wx = pack_uints((hx, wx))
        byte_stream = bs_hx_wx

        # - texture codec
        bs_texture_codec_idx = pack_uints((texture_codec_idx, ))
        byte_stream += bs_texture_codec_idx

        # - semantic_structured_enabled_flag
        byte_stream += pack_bool(ss_enabled_flag)  

        if ss_enabled_flag:
            # - group mask's block-size
            ss_block_size = args.ss_block_size
            assert ss_block_size in [16, 32, 64, 128]
            bs_ss_block_size = pack_uints((ss_block_size, ))
            byte_stream += bs_ss_block_size

            # - group msk
            # torchvision.utils.save_image(group_msk/group_msk.max(), 'workspace/debug/group_msk.png')
            group_msk_scale = ss_block_size // 16
            group_msk = torch.nn.functional.interpolate(
                group_msk, scale_factor=1/group_msk_scale, mode='nearest')
            assert len(group_msk.unique()) < self.max_group_numbers
            cdf_group_msk = self.get_cdf_group_msk(h, w, ss_block_size)
            sym = group_msk.short() + 1         # -1 means not compress. 
                                                # todo : change not compress to not transmit. 
            bs_group_msk = torchac.encode_float_cdf(cdf_group_msk, sym, check_input_bounds=True)
            # self.log('bs_group_msk: {}\n'.format(len(bs_group_msk)))
            group_msk_length = len(bs_group_msk)
            bs_group_msk_length = pack_uints((group_msk_length, ))
            byte_stream += bs_group_msk_length
            byte_stream += bs_group_msk

            # - bounding boxes
            byte_stream += pack_bool(ss_w_bboxes_flag)

            if ss_w_bboxes_flag:
                n_bboxes = len(instances_list)
                bs_n_bboxes = pack_uints((n_bboxes, ))
                byte_stream += bs_n_bboxes
                # self.log('bs_bboxes: {}\n'.format(n_bboxes*20))
                for instance in instances_list:
                    bbox = [round(each) for each in instance['bbox']]
                    x, y, w, h = bbox
                    category_id = instance['category_id']

                    bs_bbox = pack_uints((x, y, w, h, category_id))
                    byte_stream += bs_bbox
                
        return byte_stream

    def dec_header(self, byte_stream):
        header = {}
        # - decode h, w first.
        shape, byte_stream = unpack_uints(byte_stream, 2)
        hx, wx = shape
        header['hx'], header['wx'] = hx, wx

        # - texture codec
        texture_codec_idx, byte_stream = unpack_uints(byte_stream, 1)
        texture_codec_idx = texture_codec_idx[0]
        header['texture_codec_idx'] = texture_codec_idx

        # - decode semantic_structured_enabled_flag
        ss_enabled_flag, byte_stream = unpack_bool(byte_stream)
        ss_enabled_flag = ss_enabled_flag[0]
        header['ss_enabled_flag'] = ss_enabled_flag

        if ss_enabled_flag:
            # - decode group mask's block-size
            ss_block_size, byte_stream = unpack_uints(byte_stream, 1)
            ss_block_size = ss_block_size[0]

            # - decode group msk
            group_msk_length, byte_stream = unpack_uints(byte_stream, 1)
            group_msk_length = group_msk_length[0]
            bs_group_msk = byte_stream[:group_msk_length]
            h = int(np.ceil(hx/ self.padding_factor) * self.padding_factor)
            w = int(np.ceil(wx/ self.padding_factor) * self.padding_factor)
            cdf_group_msk = self.get_cdf_group_msk(h, w, ss_block_size)
            byte_stream = byte_stream[group_msk_length:]
            group_msk = torchac.decode_float_cdf(cdf_group_msk, bs_group_msk) - 1
            group_msk_scale = ss_block_size // 16
            group_msk = torch.nn.functional.interpolate(
                group_msk.float(), scale_factor=group_msk_scale, mode='nearest').short()
            # torchvision.utils.save_image(group_msk/group_msk.max(), 'workspace/debug/group_msk_rec.png')
            header['group_msk'] = group_msk

            # - decode bounding boxes
            ss_w_bboxes_flag, byte_stream = unpack_bool(byte_stream)
            ss_w_bboxes_flag = ss_w_bboxes_flag[0]
            header['ss_w_bboxes_flag'] = ss_w_bboxes_flag
            if ss_w_bboxes_flag:
                header['bboxes'] = []
                n_bboxes, byte_stream = unpack_uints(byte_stream, 1)
                n_bboxes = n_bboxes[0]
                for i in range(n_bboxes):
                    bbox, byte_stream = unpack_uints(byte_stream, 5)
                    header['bboxes'].append(bbox)
        return header, byte_stream

    def enc_full_texture(self, byte_stream, x, texture_codec_idx, p_qmap=None):
        if texture_codec_idx == 0:  # learned based codec
            if args.model == 'ours_spatialvbr_imgcomnet':   # vbr model
                if p_qmap:
                    strings = self.codec.compress(x, p_qmap)
                else:
                    strings = self.codec.compress(x, args.qmap_factor)
            else:
                strings = self.codec.compress(x)
        elif texture_codec_idx == 1:    # traditional codec.
            strings = self.codec.compress(x, args.vvc_qp)
        else:
            raise NotImplementedError
        bs_texture = pack_strings(strings)
        byte_stream += bs_texture

        return byte_stream, bs_texture

    def dec_full_texture(self, shape, byte_stream, texture_codec_idx):
        if texture_codec_idx == 0:
            num_strings = 2
            if args.model == 'ours_spatialvbr_imgcomnet':
                num_strings += 1
            strings, byte_stream = unpack_strings(byte_stream, num_strings)
            x_hat = self.codec.decompress(strings, shape)  # de-transform
        elif texture_codec_idx == 1:
            num_strings = 1
            strings, byte_stream = unpack_strings(byte_stream, num_strings)

            factor = 64
            shape = [int(math.ceil(s / factor)) * factor for s in shape]
            x_hat = self.codec.decompress(strings, shape)  # de-transform
        else:
            raise NotImplementedError
        return x_hat

    def enc_ssic_texture(self, byte_stream, x, group_msk, texture_codec_idx, p_qmap=None):
        if texture_codec_idx == 0:  # learned based codec
            if args.model == 'ours_spatialvbr_imgcomnet':
                if p_qmap:
                    strings = self.codec.group_compress(x, group_msk, p_qmap)
                else:
                    strings = self.codec.group_compress(x, group_msk, args.qmap_factor)
            else:
                strings = self.codec.group_compress(x, group_msk)
        elif texture_codec_idx == 1:    # traditional codec.
            strings = self.codec.group_compress(x, group_msk, args.vvc_qp)
        else:
            raise NotImplementedError
        bs_texture = pack_strings(strings)
        byte_stream += bs_texture
        return byte_stream, bs_texture

    def dec_ssic_texture(self, shape, byte_stream, group_msk, groups_tobe_decode, texture_codec_idx):
        num_groups = len(group_msk.unique())
        if -1 in group_msk.unique():
            num_groups -= 1

        if groups_tobe_decode:
            for group in groups_tobe_decode:
                assert group in group_msk.unique()

        if texture_codec_idx == 0:
            if args.model == 'ours_spatialvbr_imgcomnet':
                num_groups *= 2
            num_strings = num_groups + 1    # side_string
        elif texture_codec_idx == 1:
            num_strings = num_groups
        else: 
            raise NotImplementedError
        # print(len(byte_stream))
        strings, byte_stream = unpack_strings(byte_stream, num_strings)
        # print(len(byte_stream))
        # print(len(strings))
        # print(len(strings[0]), len(strings[1]), len(strings[2]))    # √
        # raise
        x_hat = self.codec.group_decompress(strings, shape, group_msk, 
                                            groups_tobe_decode)
        # print(x_hat.shape)
        # torchvision.utils.save_image(x_hat, 'x_hat.png')
        # raise

        return x_hat

    def get_cdf_group_msk(self, h, w, ss_block_size):
        # --- Use Uniform distribution.
        # . My implementation
        # cdf_group_msk = torch.ones_like(group_msk) * 2 ** 10
        # cdf_group_msk = einops.repeat(
        #     cdf_group_msk, 'b c h w -> b c h w Lp', Lp=self.max_group_numbers)
        # cdf_group_msk = torch.cumsum(cdf_group_msk, dim=-1).float()
        # cdf_group_msk = cdf_group_msk / cdf_group_msk.max(dim=-1, keepdim=True)[0]
        # . torch implementation
        # dist_uniform = torch.distributions.Uniform(0, self.max_group_numbers)
        # cdf_group_msk = dist_uniform.cdf(torch.arange(0, self.max_group_numbers))
        # cdf_group_msk = einops.repeat(cdf_group_msk, 'Lp -> b c h w Lp', 
        #                             b=1,c=1,h=int(h/16),w=int(w/16))
        # --- use Normal distribution.
        dist_normal = torch.distributions.Normal(0, 16)
        cdf_group_msk = dist_normal.cdf(torch.arange(1, self.max_group_numbers+1))
        cdf_group_msk = (cdf_group_msk - .5) * 2 
        # cdf_group_msk = einops.repeat(cdf_group_msk, 'Lp -> b c h w Lp', 
        #                             b=1,c=1,h=int(h/16),w=int(w/16))
        cdf_group_msk = einops.repeat(cdf_group_msk, 'Lp -> b c h w Lp', 
                                    b=1,c=1,h=int(h/ss_block_size),w=int(w/ss_block_size))
                                    
        cdf_group_msk = F.pad(cdf_group_msk, (1, 0))
        return cdf_group_msk

    def detect_bboxes(self, detector, img_p, img_id=0):
        img = cv2.imread(img_p) # todo : RGB -> BGR ?
        if args.det_load_result:
            p_load = os.path.join(
                args.det_load_result, img_p.split('/')[-1][:-4]+'.json')
            instances_list = json.load(open(p_load, 'r'))
        else:
            predictions = detector(img)
            instances = predictions['instances'].to("cpu")
            instances_list = []

            for i in range(len(instances)):
                instance = parse_instance(instances[i])
                instances_list.append({
                    'image_path': img_p,
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
            
            if args.det_save_result:
                p_save = os.path.join(
                    args.det_save_result, img_p.split('/')[-1][:-4]+'.json')
                json.dump(instances_list, open(p_save, 'w'), indent=4)

        h,w,c = img.shape
        return instances_list, h, w

    def get_group_msk(self, h, w, instances_list, mode='bbox'):
        msk_objects = np.zeros((h, w), dtype=np.uint8)
        msks = []

        if mode == 'bbox':
            for instance in instances_list:
                x1, y1, x2, y2 = instance['bbox']
                if args.ss_expand_bbox_factor:
                    factor = args.ss_expand_bbox_factor
                    x1 = np.clip(x1-factor, 0, w)
                    x2 = np.clip(x2+factor, 0, w)
                    y1 = np.clip(y1-factor, 0, h)
                    y2 = np.clip(y2+factor, 0, h)
                score = instance['score'] 
                if score < args.det_threshold:
                    continue
                if args.ss_det_human: 
                    if instance['category_id'] != 1:
                        continue

                msk_objects[int(y1):int(y2), int(x1):int(x2)] = 1
                x1 = (np.floor(x1 / args.ss_block_size) *args.ss_block_size).astype(np.int32)
                x2 = (np.ceil(x2 / args.ss_block_size) *args.ss_block_size).astype(np.int32)
                y1 = (np.floor(y1 / args.ss_block_size) *args.ss_block_size).astype(np.int32)
                y2 = (np.ceil(y2 / args.ss_block_size) *args.ss_block_size).astype(np.int32)
                instance_msk = torch.zeros((h, w))
                instance_msk[y1:y2, x1:x2] = 1
                msks.append(instance_msk.unsqueeze(0))
        else:
            raise NotImplementedError

        # generate the msk of groups.
        dh = self.padding_factor * math.ceil(h/self.padding_factor) - h
        dw = self.padding_factor * math.ceil(w/self.padding_factor) - w
        
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
                                    h_grid=args.ss_block_size, w_grid=args.ss_block_size)
            msks = msks.sum(dim=-1)
            new_msk = msks
            new_msk[new_msk!=0] = 1

            new_msk = new_msk.numpy().astype(np.uint8)*255

            # generate the connected components
            num_labels, group_msk, stats, centroids = cv2.connectedComponentsWithStats(
                new_msk, connectivity=4)    # group_msk is the original labels
            # use a larger bounding box to deal with the overlapped situation
            if 'groupvit' not in args.model:
                new_msk = np.zeros_like(group_msk)
                for i in range(1, len(stats)):
                    bbox = stats[i]
                    xb, yb, wb, hb, _ = bbox
                    new_msk[yb:yb+hb, xb:xb+wb] = 1

            new_msk = cv2.resize(new_msk, (w+dw, h+dh), interpolation=cv2.INTER_NEAREST)[:h, :w] / 255
            new_msk[new_msk!=0] = 1
            w_group_msk = int((w+dw)/16)
            h_group_msk = int((h+dh)/16)
            group_msk = cv2.resize(group_msk, (w_group_msk, h_group_msk), interpolation=cv2.INTER_NEAREST)
            if group_msk.sum() == int(group_msk.shape[0]*group_msk.shape[1]):
                group_msk *= 0
            return new_msk.astype(np.uint8), group_msk.astype(np.uint8), msk_objects
        else:
            h_group_msk = int((h+dh)/16)
            w_group_msk = int((w+dw)/16)
            if args.ss_det_human:
                # TODO: optimize this temperary solution, which is only compress the left-top block.
                new_msk = np.zeros((h, w), dtype=np.uint8)
                new_msk[:args.ss_block_size, :args.ss_block_size] = 1
                group_msk = np.zeros((h_group_msk, w_group_msk), dtype=np.uint8)
                group_msk[:args.ss_block_size // 16, :args.ss_block_size // 16] = 1
                msk_objects = np.zeros((h, w), dtype=np.uint8)
                msk_objects[:args.ss_block_size, :args.ss_block_size] = 1
                return new_msk, group_msk, msk_objects
            else:
                group_msk = np.zeros((h_group_msk, w_group_msk), dtype=np.uint8)
                return np.ones((h, w), dtype=np.uint8), group_msk, np.ones((h, w), dtype=np.uint8)
                

if __name__ == '__main__':

    with torch.no_grad():
        ssic = SemanticStructuredImageCoding()
        os.makedirs(args.output_files_dir, exist_ok=True)
        files = sorted(os.listdir(args.byte_stream_dir))
        total_pixels, total_size = 0, 0

        for file in tqdm.tqdm(files):
            p_file_bin = os.path.join(args.byte_stream_dir, file)
            p_file_rec = os.path.join(args.output_files_dir, file[:-4]+'.png')

            ssic.decompress(p_file_bin, p_file_rec)

            # record the pixel numbers
            img = cv2.imread(p_file_rec)    # bgr
            total_pixels += img.shape[0] * img.shape[1]
            with open(str(p_file_bin), "rb") as f:
                string = f.read()
                f.close()
            total_size += len(string) * 8

        ssic.log('bpp: {:.4f}\n'.format(total_size / total_pixels))


        