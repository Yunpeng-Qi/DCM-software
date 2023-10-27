# Copyright (c) 2021 Zhejiang University-Bingjie Zhu. All rights reserved.
import time
import os
import math
import numpy as np
import subprocess
import multiprocessing
import pandas as pd
import argparse
from PIL import Image


def parser():
    parser = argparse.ArgumentParser(description='DCM anchor pre-setting')
    parser.add_argument('--QP', type=int, default=37, metavar='N',
                        help='quantization parameter')
    parser.add_argument('--input_path', default='', metavar="DIR",
                        help='Input yuv path')
    parser.add_argument('--ori_path', default='', metavar="DIR",
                        help='Original image path')
    parser.add_argument('--cfg', default='encoder_intra_vtm.cfg', metavar="FILE",
                        help='cfg_file for encoding')

    return parser


def get_yuv_list(args):
    input_yuv_path = args.input_path
    ori_path = args.ori_path
    cfg_file = args.cfg
    yuv_list = os.listdir(input_yuv_path)
    QP = args.QP
    for img_file in yuv_list:
        img_id_str = img_file.split('.')[0]
        ori_img_path = os.path.join(ori_path, img_id_str + '.jpg')
        ori_img = Image.open(ori_img_path)
        W, H = ori_img.size
        NEW_WDT = math.ceil(W / 2) * 2
        NEW_HGT = math.ceil(H / 2) * 2

        # ---uncompressed yuv----
        out_yuv_name = img_id_str + '.yuv'
        out_yuv = os.path.join(input_yuv_path, out_yuv_name)
        # ---compressed vvc-----
        out_vvc_name = img_id_str + '.vvc'
        vvc_out_dir = "./bin/qp{}/".format(QP)
        out_vvc = os.path.join(vvc_out_dir, out_vvc_name)
        if not os.path.exists(vvc_out_dir):
            os.makedirs(vvc_out_dir)
        # ---encoding log-----
        out_log_name = img_id_str + '.log'
        log_out_dir = "./log/qp{}/".format(QP)
        out_log = os.path.join(log_out_dir, out_log_name)
        if not os.path.exists(log_out_dir):
            os.makedirs(log_out_dir)
        # ---dec yuv  ----
        out_rec_name = img_id_str + '.yuv'
        rec_out_dir = "./decyuv/qp{}/".format(QP)
        out_rec_yuv = os.path.join(rec_out_dir, out_rec_name)
        if not os.path.exists(rec_out_dir):
            os.makedirs(rec_out_dir)
        item_list.append((QP, img_id_str, cfg_file, out_yuv, out_vvc, NEW_WDT, NEW_HGT, out_rec_yuv, out_log))


def process_item(item):
    QP, img_id_str, cfg_file, out_yuv, out_vvc, NEW_WDT, NEW_HGT, out_rec_yuv, out_log = item

    # encoder
    # cfg_file = "/home/jiake/VTM-12.0/encoder_intra_vtm.cfg"
    cur_dec_yuv = os.listdir("./decyuv/qp{}/".format(QP))
    if not '{}.yuv'.format(img_id_str) in cur_dec_yuv:

        encoder_cmd = "EncoderApp -c {} -i {} -b {} -q {} --ConformanceWindowMode=1" \
                      " -wdt {} -hgt {} -f 1 -fr 1 --InternalBitDepth=10 >{}" \
            .format(cfg_file, out_yuv, out_vvc, QP, NEW_WDT, NEW_HGT, out_log)
        print(encoder_cmd)
        p = subprocess.Popen(encoder_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

        for line in p.stdout.readlines():
            print(line)
        print('\n\n')

        decoder_cmd = "DecoderApp -b {} -o {}".format(out_vvc, out_rec_yuv)
        print(decoder_cmd)

        p = subprocess.Popen(decoder_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

        for line in p.stdout.readlines():
            print(line)
        print('\n\n')
    return


if __name__ == "__main__":
    args = parser().parse_args()
    print(args)
    item_list = []
    get_yuv_list(args)
    pool = multiprocessing.Pool(1)
    complexity = pool.map(process_item, item_list)
    pool.close()
    pool.join()

'''
    complexity = pd.DataFrame(complexity)
    complexity.to_csv(complexity_file, index=False)
'''

'''
python encoder_yuv_detection_multiprocessing.py --QP=42
'''
