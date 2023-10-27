from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pylab
import argparse
import os

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def get_parser():
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument('--infer_json_name', default='infer_json.json', metavar="FILE NAME",
                        help='name of output complete json file (all images)')
    parser.add_argument('--anno_json_file', default='../../annotations/dcm_chosen_anno.csv', metavar="FILE",
                        help='name of annotation json file (all images)')
    parser.add_argument('--task',
                        default=0,
                        type=int,
                        help='different tasks')
    return parser


def eval_chosen_imgs():
    #~ eval for final chosen images
    args = get_parser().parse_args()
    annType = ['bbox', 'segm', 'keypoints']
    annType = annType[args.task]      #specify type here
    print('Running demo for *%s* results.'%(annType))
    #initialize COCO ground truth api
    annFile = args.anno_json_file
    cocoGt=COCO(annFile)
    #initialize COCO detections api
    final_json_dir='./output/json'
    infer_json_file = os.path.join(final_json_dir, args.infer_json_name)
    resFile = infer_json_file
    cocoDt=cocoGt.loadRes(resFile)
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    eval_chosen_imgs()



