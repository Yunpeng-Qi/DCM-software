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

# ******************MIKcodec*************************************************
# ******************Start****************************************************
import shutil
import compressai
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from models.MIK_codecs import build_feature_extractor, build_feature_coding, build_feature_adapter

# quality-models dictionary
model_dict = {
    1: "ckp_step48000_ete_model_bpp0.5.pth",
    2: "ckp_step48000_ete_model_bpp2.pth",
    3: "ckp_step48000_ete_model_bpp4.pth",
    4: "ckp_step48000_ete_model_bpp10.pth",
}
torch.set_num_threads(4)
# ******************End******************************************************
# 保证不同运行环境下的结果一致性
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

models = {
    "ours_spatialvbr_imgcomnet": models.CLIC_based_codecs.Spatially_adaptive_ImgComNet, # song2021vbr modified version
    'ours_checkcube': models.MIK_codecs.RDT_CheckerCube,
    'ours_groupswin_channelar': models.ours_vit.GroupChARTTC,
}

coco_map_dict = load_coco_labels()

parser = argparse.ArgumentParser(description="Compress image to bit-stream")
# basic setting.
parser.add_argument('--mode', type=str, choices=['compress', 'decompress'])
parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
parser.add_argument('--save', '-s', default='./logs/default/ssic/debug', type=str, help='log file save dir')
parser.add_argument('--texture-codec-idx', type=int, default=0, choices=[0, 1],
                    help='0 (default): learned based codec' \
                         '1: vvc codec')

# semantic strucutred image coding setting.
parser.add_argument('--ss-enabled-flag', action='store_true')   # semantic strucutred enable
parser.add_argument('--ss-block-size', type=int, default=64,
                    help='the size of minimal basic units that constructing the image.')
parser.add_argument('--ss-fg-only', action='store_true', 
                    help='only compress groups with foreground.')
parser.add_argument('--ss-det-human', action='store_true', 
                    help='only detect human.')
parser.add_argument('--ss-w-bboxes-flag', action='store_true',  # semantic strucutred with bbox
                    help='compress bounding boxes or not.')
parser.add_argument('--ss-expand-bbox-factor', type=int, default=0,
                    help='expand the bounding box to generate the group mask.')

# codec setting
parser.add_argument("--model", type=str, default="ours_meanscalehyper", help="Model")
parser.add_argument("--transform-channels", type=int, nargs='+',
                    default=[128, 128, 128, 192],help="Transform channels.")
parser.add_argument("--hyper-channels", type=int, nargs='+',
                    default=None,help="Transform channels.")
parser.add_argument('--resume', type=str, default='ckpts/ours_spatialvbr_imgcomnet_vbr_mse_msssim/2000000.pth',
                    help='Checkpoint path')

# vbr setting 
parser.add_argument('--qmap-factor', type=float, default=0.5)
parser.add_argument('--qmap-dir', type=str, default='')

# vvc setting
parser.add_argument('--vvc-qp', type=int, default=42)

# inputs, bins, and outputs
parser.add_argument("--input-files-path", type=str, default="../logs/samples/000000384616.jpg",
                    help="Input image path")
parser.add_argument('--byte-stream-path', type=str, default='../workspace/bitstream')
parser.add_argument("--output-files-path", type=str, default="../workspace/rec",    # reconstructed feature path
                    help="Output binary file")

# detection setting
parser.add_argument('--det-cfg-file', type=str, 
                    default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
parser.add_argument('--det-weights', type=str,
                    default='./packages/Model/Detector_model/model_final_68b088.pkl')

parser.add_argument('--det-threshold', type=float, default=0.05)
parser.add_argument('--quality',
                    type=int,
                    default=1,
                    help='quality of model weight') # different quality

# image_resize(only encoder)
parser.add_argument('--image_resize_xmin',
                    type=int,
                    default=800,
                    help='image resize xmin')
parser.add_argument('--image_resize_xmax',
                    type=int,
                    default=1333,
                    help='image resize xmax')

# use other feature adapter(only decoder)
parser.add_argument('--other_feature_adapter_flag', action='store_true')
parser.add_argument('--other_feature_adapter_path', type=str,
                    default=None,
                    help='other feature adapter model path')

parser.add_argument(
    "--mik_model_path",
    default="./packages/Model/MIKcodec_model",
    help="path to MIKcodec",
)
parser.add_argument(
    "--config_file",
    default="./packages/Environment/detectron2-0.5/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="Minimum score for instance predictions to be shown (PANOPTIC_FPN)",
)
parser.add_argument(
    "--score_threshold",
    type=float,
    default=0.05,
    help="Minimum score for instance predictions to be shown (ROI_HEADS, RETINANET)",
)
parser.add_argument(
    "-c",
    "--coder",
    choices=compressai.available_entropy_coders(),
    default=compressai.available_entropy_coders()[0],
    help="Entropy coder (default: %(default)s)",
)
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
    default=['MODEL.WEIGHTS', './ckpts/detector/model_final_68b088.pkl'],
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

class dtmVBase():
    def __init__(self):
        self.build_logger()

    def build_logger(self):
        os.makedirs(args.save, exist_ok=True)
        # self.p_log = os.path.join(args.save,
        #     '{}.txt'.format(str(datetime.datetime.now()).replace(':', '-')[:-7]))
        #write_log(self.p_log, str(args).replace(', ', ',\n\t') + '\n')
        
    # def log(self, content):
    #     return write_log(self.p_log, content)

class SemanticStructuredBase(dtmVBase):
    def __init__(self):
        super().__init__()

        # build detector to detect bounding boxes (coordinates and category)
        if args.ss_enabled_flag and args.mode == 'compress':
            if args.mode == 'compress':
                self.detector = self.build_detector()

    def build_detector(self):   # build detector from detectron2
        cfg = get_cfg()
        cfg_file = args.det_cfg_file    # COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = args.det_weights
        detector = DefaultPredictor(cfg)
        return detector

    def get_cdf_group_msk(self, h, w, ss_block_size):
        """ get the cdf to compress group mask

        Args:
            h (int): the height of the image. 
            w (int): the width of the image.
            ss_block_size (_type_): block size for grouping, default is 16.

        Returns:
            cdf_group_msk (Torch.tensor): the cdf to compress group mask
        """        
        # --- use Normal distribution.
        dist_normal = torch.distributions.Normal(0, 16) # mean=0, var=16
        cdf_group_msk = dist_normal.cdf(torch.arange(1, self.max_group_numbers+1))  # calculate cdf value of each number
        cdf_group_msk = (cdf_group_msk - .5) * 2    # 将 cdf_group_msk 中的值映射到区间 [-1, 1]
        cdf_group_msk = einops.repeat(cdf_group_msk, 'Lp -> b c h w Lp', 
                                    b=1, c=1, h=int(h/ss_block_size), w=int(w/ss_block_size))
                                    
        cdf_group_msk = F.pad(cdf_group_msk, (1, 0))
        return cdf_group_msk

    def detect_bboxes(self, detector, img_p, img_id=0):
        """detect bounding boxes of the corresponding image.

        Args:
            detector (_type_): 
            img_p (_type_): path of the image to be detected. 
            img_id (int, optional): _description_. Defaults to 0.

        Returns:
            list: the detected results.
        """        
        img = cv2.imread(img_p) 
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
        
        h,w,c = img.shape
        return instances_list, h, w

    def get_group_msk(self, h, w, instances_list, mode='bbox'):
        """
        Generate the group mask based on detected results.
        Here only includes the pipeline based on bounding boxes.

        Args:
            h (int): the height of the image
            w (int): the width of the image
            instances_list (_type_): the detected instances information.
            mode (str, optional): the mode based on to generate group mask. Defaults to 'bbox'.

        Raises:
            NotImplementedError: If the mode is not implemented. 

        Returns:
            _type_: _description_
        """        
        msk_objects = np.zeros((h, w), dtype=np.uint8)
        msks = []

        if mode == 'bbox':
            for instance in instances_list:
                x1, y1, x2, y2 = instance['bbox']
                if args.ss_expand_bbox_factor:  # expand the bounding box, default=0
                    factor = args.ss_expand_bbox_factor
                    x1 = np.clip(x1-factor, 0, w)   # np.clip(a, a_min, a_max)
                    x2 = np.clip(x2+factor, 0, w)
                    y1 = np.clip(y1-factor, 0, h)
                    y2 = np.clip(y2+factor, 0, h)
                score = instance['score'] 
                if score < args.det_threshold:
                    continue
                if args.ss_det_human: 
                    if instance['category_id'] != 1:
                        continue

                msk_objects[int(y1):int(y2), int(x1):int(x2)] = 1   # get mask for instances
                x1 = (np.floor(x1 / args.ss_block_size) *args.ss_block_size).astype(np.int32) # refine x,y to grid edge
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
            msks = torch.cat(msks, dim=0).float()   

            # pad the msk
            msks = F.pad(msks, (0, dw, 0, dh)) 
            
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
                for i in range(1, len(stats)):  # stats对应每个连通区域外接矩形的起始坐标x,y和wide,height
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
                # TODO: optimize this temperary solution, which only compresses the left-top block for simplicity.
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

class SemanticStructuredImageCoding(SemanticStructuredBase):
    def __init__(self):
        super().__init__()
        # build codec
        self.codec = self.build_codec()
        self.max_group_numbers = 64
        self.padding_factor = 64

    def build_codec(self):
        if args.device == 'cpu':
            args.device = torch.device("cpu")
        elif args.device == 'gpu':
            assert args.texture_codec_idx != 1  # 0: NTC, 1: VVC
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

        # self.log('resume the ckpt from : {} and the message is {}\n'.format(
        #     args.resume, msg))
        model.to(args.device).eval()

        return model

    def compress(self, input_p, bin_p, group_msk=None, instances_list=None):
        # * compress header.
        x, hx, wx = load_img(input_p, padding=True, factor=self.padding_factor)
        _, c, h, w = x.shape
        if c != 3:
            # self.log('{} with invalid shape of: {}, ' \
            #         'take the first 3 channels to be an RGB image.\n'.format(
            #     input_p, x.shape))
            # x = x[:,:3,...]
            raise ValueError
                
        x = x.to(args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()

        # * compress header
        bs_header = self.enc_header(
            hx, wx, h, w, args.ss_enabled_flag, group_msk, 
            instances_list, args.ss_w_bboxes_flag, args.texture_codec_idx)
        byte_stream = bs_header

        # * compress texture.
        if args.qmap_dir:
            assert args.texture_codec_idx == 0
            p_qmap = os.path.join(args.qmap_dir, input_p.split('/')[-1][:-4]+'.png')
        else:
            p_qmap = None

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
        # self.log('Encoded: {} -> {}. ' \
        #     'Bpp(header): {:.4f}, Bpp(texture): {:.4f}, Bpp(full): {:.4f}\n'.format(
        #           input_p, bin_p, bpp_header, bpp_texture, bpp))

    def decompress(self, bin_p, rec_p, groups_tobe_decode=None):
        with open(str(bin_p), "rb") as f:
            byte_stream = f.read()
        # print(len(byte_stream) * 8)
        # * decompress header.
        header, byte_stream = self.dec_header(byte_stream)
        # print(header, byte_stream)

        # * decompress texture.
        shape = (header['hx'], header['wx'])
        if header['ss_enabled_flag']:
            x_hat = self.dec_ssic_texture(
                shape, byte_stream, 
                header['group_msk'], groups_tobe_decode, header['texture_codec_idx'])
        else:
            x_hat = self.dec_full_texture(
                shape, byte_stream,
                header['texture_codec_idx'])

        ## save image
        x_hat = x_hat[:, :, :shape[0], :shape[1]].clamp(0, 1).mul(255).round()
        x_hat = x_hat.squeeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        x_hat = Image.fromarray(x_hat)
        x_hat.save(rec_p)

    def enc_header(
        self, hx, wx, h, w, ss_enabled_flag, group_msk, 
        instances_list, ss_w_bboxes_flag, texture_codec_idx):
        # h, w
        bs_hx_wx = pack_uints((hx, wx)) # pack uints to byte stream
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
            group_msk_scale = ss_block_size // 16
            group_msk = torch.nn.functional.interpolate(
                group_msk, scale_factor=1/group_msk_scale, mode='nearest')
            assert len(group_msk.unique()) < self.max_group_numbers
            cdf_group_msk = self.get_cdf_group_msk(h, w, ss_block_size)
            sym = group_msk.short() + 1         # -1 means not compress. 
                                                # todo : change not compress to not transmit. 
            bs_group_msk = torchac.encode_float_cdf(cdf_group_msk, sym, check_input_bounds=True)
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
        strings, byte_stream = unpack_strings(byte_stream, num_strings)
        x_hat = self.codec.group_decompress(strings, shape, group_msk, 
                                            groups_tobe_decode)

        return x_hat

class MIKImageEncoder(nn.Module):
    def __init__(self, cfg, feature_extractor, feature_coding):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_coding = feature_coding
        self.device = torch.device("cpu")
        num_channels = len(cfg.MODEL.PIXEL_MEAN)    # detectron stuffs
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)

    def preprocess_image(self, images, pixel_mean, pixel_std, device):
        # 1、resize
        # images = images.to(device)
        h, w, _ = images.shape
        images = Image.fromarray(images)
        size = args.image_resize_xmin
        max_size = args.image_resize_xmax
        scale = size * 1.0 / min(h, w)

        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        images = images.resize((neww, newh))
        images = np.asarray(images)
        images = torch.as_tensor(images.astype("float32").transpose(2, 0, 1)).unsqueeze(dim=0)

        # 2、normalization
        images = (images - pixel_mean) / pixel_std

        # 3、padding
        [batches, C, H, W] = images.shape

        padding_bottom = (H // 64 + 1) * 64 - H if not H % 64 == 0 else 0
        padding_right = (W // 64 + 1) * 64 - W if not W % 64 == 0 else 0
        after_padding_W = W + padding_right
        after_padding_H = H + padding_bottom
        new_image = torch.FloatTensor(batches, C, after_padding_H, after_padding_W)
        for i in range(0, batches):
            image = images[i]
            image = F.pad(
                image,
                (0, padding_right, 0, padding_bottom),
                mode="constant",
                value=0,
            )
            new_image[i] = image

        return new_image

    def forward(self, batched_inputs):
        # resize + normalization + padding
        new_image = self.preprocess_image(batched_inputs, self.pixel_mean, self.pixel_std, self.device)
        features = self.feature_extractor(new_image)
        out_enc = self.feature_coding.compress(features)
        return out_enc

class MIKImageDecoder(nn.Module):
    def __init__(self, feature_coding, feature_adapter):
        super().__init__()
        self.feature_coding = feature_coding
        self.feature_adapter = feature_adapter
        self.device = torch.device("cpu")

    def forward(self, strings, shape):
        out_dec = self.feature_coding.decompress(strings, shape)
        feature_out = self.feature_adapter(out_dec["x_hat"])
        return feature_out

class SemanticStructuredFeatureCoding(SemanticStructuredBase):
    def __init__(self):
        super().__init__()
        self.max_group_numbers = 64
        self.padding_factor = 64
        if args.device == 'gpu':
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.score_threshold  # default: 0.05
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold  # default: 0.05
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold  # default: 0.5
        cfg.freeze()
        return cfg

    def write_uints(self, fd, values, fmt=">{:d}I"):
        fd.write(struct.pack(fmt.format(len(values)), *values))

    def write_bytes(self, fd, values, fmt=">{:d}s"):
        if len(values) == 0:
            return
        fd.write(struct.pack(fmt.format(len(values)), values))

    def write_uchars(self, fd, values, fmt=">{:d}B"):
        fd.write(struct.pack(fmt.format(len(values)), *values))

    def read_uints(self, fd, n, fmt=">{:d}I"):
        sz = struct.calcsize("I")
        return struct.unpack(fmt.format(n), fd.read(n * sz))

    def read_bytes(self, fd, n, fmt=">{:d}s"):
        sz = struct.calcsize("s")
        return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

    def read_uchars(self, fd, n, fmt=">{:d}B"):
        sz = struct.calcsize("B")
        return struct.unpack(fmt.format(n), fd.read(n * sz))

    def enc_Semantic_information(self, mikEncoder_input_image_path):
        # 1、get Semantic information
        if args.device == 'cuda':
            torch.cuda.synchronize()    # 在GPU上同步操作
        if args.ss_enabled_flag:
            # get bbox
            instances_list, h1, w1 = self.detect_bboxes(self.detector, mikEncoder_input_image_path)
            msk, group_msk, msk_objects = self.get_group_msk(h1, w1, instances_list)
            group_msk = torch.from_numpy(group_msk).unsqueeze(0).unsqueeze(0).float()
            if args.ss_fg_only:
                if group_msk.sum() != 0:
                    group_msk[group_msk == 0] = -1
        else:
            group_msk, instances_list = None, None

        # 2、compress Semantic information
        x, hx, wx = load_img(mikEncoder_input_image_path, padding=True, factor=self.padding_factor)
        _, c, h, w = x.shape
        bs_header = self.enc_header(
            hx, wx, h, w, args.ss_enabled_flag, group_msk,
            instances_list, args.ss_w_bboxes_flag, args.texture_codec_idx)

        return bs_header

    def enc_header(
            self, hx, wx, h, w, ss_enabled_flag, group_msk,
            instances_list, ss_w_bboxes_flag, texture_codec_idx):
        # # h, w
        # bs_hx_wx = pack_uints((hx, wx))
        # byte_stream = bs_hx_wx

        # # - texture codec
        # bs_texture_codec_idx = pack_uints((texture_codec_idx,))
        # byte_stream = bs_texture_codec_idx

        # - semantic_structured_enabled_flag
        byte_stream = pack_bool(ss_enabled_flag)

        if ss_enabled_flag:
            # - group mask's block-size
            ss_block_size = args.ss_block_size
            assert ss_block_size in [16, 32, 64, 128]
            bs_ss_block_size = pack_uints((ss_block_size,))
            byte_stream += bs_ss_block_size

            # - group msk
            group_msk_scale = ss_block_size // 16
            group_msk = torch.nn.functional.interpolate(
                group_msk, scale_factor=1 / group_msk_scale, mode='nearest')
            assert len(group_msk.unique()) < self.max_group_numbers
            cdf_group_msk = self.get_cdf_group_msk(h, w, ss_block_size)
            sym = group_msk.short() + 1  # -1 means not compress.
            # todo : change not compress to not transmit.
            bs_group_msk = torchac.encode_float_cdf(cdf_group_msk, sym, check_input_bounds=True)
            group_msk_length = len(bs_group_msk)
            bs_group_msk_length = pack_uints((group_msk_length,))
            byte_stream += bs_group_msk_length
            byte_stream += bs_group_msk

            # - bounding boxes
            byte_stream += pack_bool(ss_w_bboxes_flag)

            if ss_w_bboxes_flag:
                n_bboxes = len(instances_list)
                bs_n_bboxes = pack_uints((n_bboxes,))
                byte_stream += bs_n_bboxes
                for instance in instances_list:
                    bbox = [round(each) for each in instance['bbox']]
                    x, y, w, h = bbox
                    category_id = instance['category_id']

                    bs_bbox = pack_uints((x, y, w, h, category_id))
                    byte_stream += bs_bbox
        return byte_stream

    def dec_header(self, byte_stream,wx, hx):
        header = {}
        # - decode h, w first.
        # shape, byte_stream = unpack_uints(byte_stream, 2)
        # hx, wx = shape
        header['hx'], header['wx'] = hx, wx

        # # - texture codec
        # texture_codec_idx, byte_stream = unpack_uints(byte_stream, 1)
        # texture_codec_idx = texture_codec_idx[0]
        # header['texture_codec_idx'] = texture_codec_idx

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
            h = int(np.ceil(hx / self.padding_factor) * self.padding_factor)
            w = int(np.ceil(wx / self.padding_factor) * self.padding_factor)
            cdf_group_msk = self.get_cdf_group_msk(h, w, ss_block_size)
            byte_stream = byte_stream[group_msk_length:]
            group_msk = torchac.decode_float_cdf(cdf_group_msk, bs_group_msk) - 1
            group_msk_scale = ss_block_size // 16
            group_msk = torch.nn.functional.interpolate(
                group_msk.float(), scale_factor=group_msk_scale, mode='nearest').short()
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

    def compress(self, mikEncoder_input_image_path, mikEncoder_output_bitstream_path, mikEncoder_encode_quality,
                 mikEncoder_logdir):
        dtm_v_encoder_start_time = time.time()
        # 1、get and compress Semantic information
        bs_header = self.enc_Semantic_information(mikEncoder_input_image_path)

        # 2、MIKEncoder
        cfg = mikCodec.setup_cfg(args)  # where mikCodec
        compressai.set_entropy_coder(args.coder)    # compressai.available_entropy_coders()[0]

        # log
        logger = setup_logger(name="Encoder")

        # load model
        feature_extractor = build_feature_extractor()
        feature_coding = build_feature_coding(quality=3)
        model_path = os.path.join(mikCodec_model_path, model_dict[mikEncoder_encode_quality])
        checkpoint = torch.load(model_path, map_location=args.device)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        feature_coding.load_state_dict(checkpoint['feature_coding'])
        encode_model = MIKImageEncoder(cfg, feature_extractor, feature_coding)
        encode_model.eval()

        # encode
        print("***************************************************************************************************************")
        logger.info("Start encode...")
        images = read_image(mikEncoder_input_image_path, format="BGR")
        H, W, _ = images.shape

        with torch.no_grad():
            out_enc = encode_model(images)
        # gen bitstream
        quality_code = mikEncoder_encode_quality - 1 & 0x0F
        shape = out_enc["shape"]

        with Path(mikEncoder_output_bitstream_path).open("wb") as f:
            # MIK information
            self.write_uchars(f, (quality_code,))
            self.write_uints(f, (W, H))
            self.write_uints(f, (shape[0], shape[1], len(out_enc["strings"])))
            for s in out_enc["strings"]:
                self.write_uints(f, (len(s[0]),))
                self.write_bytes(f, s[0])

            # Semantic information
            bs_header_len = len(bs_header)
            self.write_uints(f, (bs_header_len,))
            f.write(bs_header)
        
        # 记录编码结束时间
        dtm_v_encoder_end_time = time.time()

        dtm_v_encoder_cost_time = dtm_v_encoder_end_time - dtm_v_encoder_start_time
        # cal bpp and time
        pixel = W * H
        size = os.path.getsize(mikEncoder_output_bitstream_path)
        bpp_rate = float(size) * 8 / pixel

        # save log
        try:
            input_image_name = mikEncoder_input_image_path.split("/")[-1].split(".")[0]
        except Exception as e:
            input_image_name = time.strftime('%Y%m%d_%H%M%S')

        info_path = os.path.join(mikEncoder_logdir, "{}_enc_info_{}.txt".format(input_image_name, mikEncoder_encode_quality))
        info_txt = open(info_path, 'a')
        info_txt.write(
            "Encode on:{},bpp:{},enc_time:{} [s]".format(mikEncoder_input_image_path,bpp_rate, dtm_v_encoder_cost_time))
        info_txt.close()

        logger.info(
            "Inference on image {}, enc time: {} [s], bpp: {}".format(mikEncoder_input_image_path, dtm_v_encoder_cost_time, bpp_rate))
        print("Done!")
        print("***************************************************************************************************************")
        return out_enc

    def decompress(self, mikDecoder_input_bitstream_path, mikDecoder_output_reconfeature_path, mikCodec_model_path, mikDecoder_logdir):
        dtm_v_decoder_start_time = time.time()
        compressai.set_entropy_coder(args.coder)
        # log
        logger = setup_logger(name="Decoder")
        # decode
        print(
            "***************************************************************************************************************")
        logger.info("Start decode...")
        with Path(mikDecoder_input_bitstream_path).open("rb") as f:
            # decode MIK information
            code = self.read_uchars(f, 1)
            original_size = self.read_uints(f, 2)
            W, H = original_size
            shape = self.read_uints(f, 2)
            strings = []
            n_strings = self.read_uints(f, 1)[0]
            for _ in range(n_strings):
                s = self.read_bytes(f, self.read_uints(f, 1)[0])
                strings.append([s])

            # decode Semantic information
            header_len = self.read_uints(f, 1)[0]
            byte_stream = f.read(header_len)
            header, byte_stream = self.dec_header(byte_stream, W, H)

        # all model path
        decode_quality = (code[0] & 0x0F) + 1
        model_path = os.path.join(mikCodec_model_path, model_dict[decode_quality])
        checkpoint = torch.load(model_path, map_location=args.device)

        # feature adapter path
        if args.other_feature_adapter_flag:
            feature_adapter = build_feature_adapter(128, 256)
            other_feature_adapter_model_path = args.other_feature_adapter_path
            feature_adapter_checkpoint = torch.load(other_feature_adapter_model_path, map_location=args.device)
            feature_adapter.load_state_dict(feature_adapter_checkpoint['feature_adapter'])
        else:
            feature_adapter = build_feature_adapter(128, 256)
            feature_adapter.load_state_dict(checkpoint['feature_adapter'])

        feature_coding = build_feature_coding(quality=3)
        feature_coding.load_state_dict(checkpoint['feature_coding'])
        decode_model = MIKImageDecoder(feature_coding, feature_adapter)
        decode_model.eval()

        with torch.no_grad():
            feature_out = decode_model(strings, shape)

        # save reco_feature
        res2_ete_codec = feature_out
        torch.save(res2_ete_codec, mikDecoder_output_reconfeature_path)
        dtm_v_decoder_end_time = time.time()
        dtm_v_decoder_cost_time = dtm_v_decoder_end_time - dtm_v_decoder_start_time
        print("Semantic Information:")
        print(header)
        logger.info(
            "Reconstructing from bitstream {}, dec_time: {} [s]".format(mikDecoder_input_bitstream_path, dtm_v_decoder_cost_time))

        # save log
        try:
            out_feature_name = mikDecoder_input_bitstream_path.split("/")[-1].split(".")[0]
        except Exception as e:
            out_feature_name = time.strftime('%Y%m%d_%H%M%S')

        info_path = os.path.join(mikDecoder_logdir, "{}_dec_info_{}.txt".format(out_feature_name,decode_quality))

        info_txt = open(info_path, 'a')
        info_txt.write(
            "Reconstruct features from bitstream {},dec_time:{} [s]".format(mikDecoder_input_bitstream_path, dtm_v_decoder_cost_time))
        info_txt.close()
        print("Done!")
        print("***************************************************************************************************************")
        return header, byte_stream

if __name__ == '__main__':
    # here is an example of compressing and decompressing.
    with torch.no_grad():
        mikCodec = SemanticStructuredFeatureCoding()  # DTM-V-FeatureCoding
        mikCodec_model_path = args.mik_model_path     # DTM-V model path ./packages/Model/MIKcodec_model
        if args.mode == 'compress':
            # DTM-V-Encoder parameters
            dtmV_input_image_path = args.input_files_path        # DTM-V-Encoder input image path
            dtmV_output_bitstream_path = args.byte_stream_path   # DTM-V-Encoder bitstream path
            dtmV_encode_quality = args.quality                   # DTM-V-Encoder quality: 0~4
            dtmV_encode_logdir = args.save                       # DTM-V-Encoder log save dir
            if not os.path.exists(dtmV_encode_logdir):
                os.makedirs(dtmV_encode_logdir)

            # encode
            out_enc = mikCodec.compress(dtmV_input_image_path, dtmV_output_bitstream_path,
                                        dtmV_encode_quality, dtmV_encode_logdir)
            # [out_enc] for Semantic Structure

        elif args.mode == 'decompress':
            # DTM-V-Decoder parameters
            dtmV_input_bitstream_path = args.byte_stream_path        # DTM-V-Decoder input bitstream path
            dtmV_output_reconfeature_path = args.output_files_path   # DTM-V-Decoder output recon feature path
            dtmV_decode_logdir = args.save                           # DTM-V-Decoder log save dir
            if not os.path.exists(dtmV_decode_logdir):
                os.makedirs(dtmV_decode_logdir)
            # decode
            header, byte_stream = mikCodec.decompress(dtmV_input_bitstream_path, dtmV_output_reconfeature_path,
                                mikCodec_model_path, dtmV_decode_logdir)
            # [header, byte_stream] for Semantic
        else:
            raise NotImplementedError