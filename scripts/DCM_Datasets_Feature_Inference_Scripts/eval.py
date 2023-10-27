from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pylab
import argparse
import os
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script
import torch

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def get_parser():
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument('--infer_json_name', default='infer_json.json', metavar="FILE NAME",
                        help='name of output complete json file (all images)')
    parser.add_argument('--final_json_dir', default='../image_output/json', metavar="DIR",
                        help='output of json directory (inference per image)')
    parser.add_argument('--anno_json_file', default='../../annotations/dcm_chosen_anno.csv', metavar="FILE",
                        help='name of annotation json file (all images)')
    parser.add_argument('--task',
                        default=0,
                        type=int,
                        help='different tasks')
    return parser


def eval_chosen_imgs():
    # ~ eval for final chosen images
    args = get_parser().parse_args()
    annType = ['bbox', 'segm', 'keypoints']
    annType = annType[args.task]  # specify type here
    print('Running demo for *%s* results.' % (annType))
    # initialize COCO ground truth api
    annFile = args.anno_json_file
    cocoGt = COCO(annFile)  # 使用COCO API初始化一个COCO ground truth实例
    # initialize COCO detections api
    final_json_dir = args.final_json_dir
    infer_json_file = os.path.join(final_json_dir, 'infer_json.json')

    # Check if the file exists before loading

    # if not os.path.exists(infer_json_file):
    #     # 创建一个空的 JSON 数据结构
    #     data = {}
    #     import json   
    #     # 将数据保存到 JSON 文件中
    #     with open(infer_json_file, 'w') as json_file:
    #         json.dump(data, json_file)


    # if os.path.exists(infer_json_file):
    #     resFile = infer_json_file
    #     cocoDt = cocoGt.loadRes(resFile)
    #     # running evaluation
    #     cocoEval = COCOeval(cocoGt, cocoDt, annType)
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()
    #     mAP_result = cocoEval.stats
    #     print("************************************************************************************")
    #     print("Result mAP@0.5:0.05:0.95 = {}".format(format(mAP_result[0], '.14f')))    # IoU threshold=0.5, calculate precession from 0.05 to 0.95
    #     print("************************************************************************************")
    # else:
    #     print(f"Error: {infer_json_file} does not exist.")

    resFile = infer_json_file
    cocoDt = cocoGt.loadRes(resFile)
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP_result = cocoEval.stats
    print("************************************************************************************")
    print("Result mAP@0.5:0.05:0.95 = {}".format(format(mAP_result[0], '.14f')))
    print("************************************************************************************")

if __name__ == '__main__':
    eval_chosen_imgs()



