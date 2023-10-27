# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import time
import json
import torch
import struct
import argparse
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F
from predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper
from detectron2.structures import ImageList
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer



def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit    # 导入PyTorch的JIT模块，该模块用于将PyTorch模型转换为脚本，以实现更高的性能和部署能力
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


from pycocotools import mask as mask_util
os.environ["PYTORCH_JIT"] = "0"
WINDOW_NAME = "COCO detections"

coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)   # cfg
    cfg.merge_from_list(args.opts)  # weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.score_threshold  # default: 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold  # default: 0.05
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold  # default: 0.5
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config_file",
        default="../../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
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
    parser.add_argument('--dataset',    # dataset name -- 0: coco, 1:openimages
                        default=0,
                        type=int,
                        help='coco or openimages')
    parser.add_argument('--task',   # # task name -- 0: object detction, 1: instance segemtation, 2: pose estimation
                        default=0,
                        type=int,
                        help='different tasks')
    parser.add_argument('--val_dir', default='../chosen_pictures', metavar="DIR",
                        help='openimage validation image directory')
    parser.add_argument('--feature_dir', default='../chosen_pictures', metavar="DIR",
                        help='openimage validation image directory')
    parser.add_argument('--infer_json_dir', default='../image_output/json', metavar="DIR",
                        help='output of json directory (inference per image)')
    parser.add_argument('--final_json_dir', default='../image_output/json', metavar="DIR",
                        help='output of json directory (inference per image)')
    parser.add_argument('--infer_json_name', default='infer_json.json', metavar="FILE NAME",
                        help='name of output complete json file (all images)')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', '../../pre_train_model/model_final_68b088.pkl'],
        nargs=argparse.REMAINDER,

    )
    return parser


def parse_instance(predictions):
    """
    extract instances' predicted information then store them into info_dict
    """
    info_dict = {}
    info_dict['pred_boxes'] = predictions['instances']._fields['pred_boxes'][i].tensor.detach().cpu().numpy()[0].tolist()
    info_dict['scores'] = predictions['instances']._fields['scores'][i].detach().cpu().item()
    info_dict['pred_class'] = predictions['instances']._fields['pred_classes'][i].detach().cpu().item()

    has_mask = predictions["instances"].has("pred_masks")
    has_keypoints = predictions["instances"].has("pred_keypoints")

    if has_mask:
        masks = predictions['instances'][i]._fields['pred_masks'].cpu()
        masks = masks.squeeze()
        mask = np.asfortranarray(masks).astype(np.uint8)
        segmentation = mask_util.encode(mask)
        mask_rle = {            # why RLE format?
            'counts': segmentation["counts"].decode('utf-8'),
            'size': segmentation["size"]
        }
        info_dict['segmentation'] = mask_rle

    if has_keypoints:
        keypoints = predictions["instances"].to(torch.device("cpu")).pred_keypoints
        keypoints[i][:, :2] -= 0.5
        info_dict['keypoints'] = keypoints[i].flatten().tolist()

    return info_dict


def load_openimage_names(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
        del lines[0], lines[-1]
        for i in range(len(lines)):
            lines[i] = lines[i].split(',')
        img_names = list(zip(*lines))[0]
        img_names = list(set(img_names))    # set 去除重复的图像文件名

    return img_names


def load_coco_labels(): # map model output with real coco labels
    category_ids = [i for i in range(0, 80)]
    real_category_ids = list(coco_id_name_map.keys())
    tmp_dict = {}
    for x, y in zip(category_ids, real_category_ids):
        tmp_dict[x] = y

    return tmp_dict


def load_instances(file_names, json_dir, coco_map_dict, task_class, dataset_class):
    final_instances = []
    for file_name in file_names:
        name = file_name[:-4]
        file_path = os.path.join(json_dir, name + '.json')
        if dataset_class == 0:
            name = int(name)
        with open(file_path, 'r') as f:
            tmp = json.loads(f.read())
            instances = tmp['instances']
            for instance in instances:
                try:
                    if task_class == 0:
                        final_instances.append({
                            'image_id': name,
                            'category_id': coco_map_dict[instance['pred_class']],
                            'bbox': [
                                instance['pred_boxes'][0],
                                instance['pred_boxes'][1],
                                instance['pred_boxes'][2] - instance['pred_boxes'][0],
                                instance['pred_boxes'][3] - instance['pred_boxes'][1],
                            ],
                            'score': instance['scores'],
                        })
                    elif task_class == 1:
                        final_instances.append({
                            'segmentation': instance['segmentation'],
                            'image_id': name,
                            'category_id': coco_map_dict[instance['pred_class']],
                            'bbox': [
                                instance['pred_boxes'][0],
                                instance['pred_boxes'][1],
                                instance['pred_boxes'][2] - instance['pred_boxes'][0],
                                instance['pred_boxes'][3] - instance['pred_boxes'][1],
                            ],
                            'score': instance['scores'],
                        })
                    elif task_class == 2:
                        final_instances.append({
                            'keypoints': instance['keypoints'],
                            'image_id': name,
                            'category_id': coco_map_dict[instance['pred_class']],
                            'score': instance['scores'],
                        })
                except:
                    print('Warning: {} not be calculate in this inference'.format(file_name))
                    continue
    return final_instances

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def my_fpn_forward(bottom_up_features, model):
    """
    Args:
        input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
            feature map tensor for each feature level in high to low resolution order.

    Returns:
        dict[str->Tensor]:
            mapping from feature map name to FPN feature map tensor
            in high to low resolution order. Returned feature names follow the FPN
            paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
            ["p2", "p3", ..., "p6"].
    """

    x = [bottom_up_features[f] for f in model.backbone.in_features[::-1]]   # [::-1] 将列表中的元素反向排序
    # x form high resolution to low resolution
    results = []
    prev_features = model.backbone.lateral_convs[0](x[0])   # lateral_convs[0]第一个横向卷积层
    results.append(model.backbone.output_convs[0](prev_features))
    for features, lateral_conv, output_conv in zip(
            x[1:], model.backbone.lateral_convs[1:], model.backbone.output_convs[1:]
    ):  # zip 函数将三个可迭代对象打包在一起，并在每次迭代中分别提取它们的元素
        top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
        lateral_features = lateral_conv(features)
        prev_features = lateral_features + top_down_features    # FPN add operation
        if model.backbone._fuse_type == "avg":
            prev_features /= 2  # 一种平滑融合的方式
        results.insert(0, output_conv(prev_features))

    if model.backbone.top_block is not None:
        top_block_in_feature = bottom_up_features.get(model.backbone.top_block.in_feature, None)
        if top_block_in_feature is None:
            top_block_in_feature = results[model.backbone._out_features.index(model.backbone.top_block.in_feature)]
        results.extend(model.backbone.top_block(top_block_in_feature))
    assert len(model.backbone._out_features) == len(results)
    return dict(zip(model.backbone._out_features, results))


if __name__ == "__main__":
    # torch.set_printoptions(threshold=np.inf)
    dtm_v_task_start_time = time.time()
    torch.backends.cudnn.deterministic = True   # 设置CuDNN以确定性模式运行
    args = get_parser().parse_args()
    val_dir = args.val_dir
    feature_dir = args.feature_dir
    chosen_images = os.listdir(val_dir)
    final_json_dir = args.final_json_dir
    os.makedirs(final_json_dir, exist_ok=True)
    infer_json_dir = args.infer_json_dir
    os.makedirs(infer_json_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    setup_logger(name="Inference")  # logger form detectron2
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)   
    demo = VisualizationDemo(cfg)

    task_class = args.task
    dataset_class = args.dataset
    assert task_class in [0, 1, 2]
    assert dataset_class in [0, 1]

    device = torch.device("cuda")
    task_model = build_model(cfg)
    DetectionCheckpointer(task_model).load(cfg.MODEL.WEIGHTS)
    dataset = DatasetMapper(cfg, is_train=None)
    task_model.eval()
    task_model = task_model.to(device)


    for img_file in chosen_images:
        img_id_str = img_file.split(".")[0] # split filename
        save_path = os.path.join(infer_json_dir, '{}.json'.format(img_id_str))
        img_path = os.path.join(val_dir, img_file)
        # use PIL, to be consistent with evaluation
        img = read_image(img_path, format="BGR")
        inputs = [dataset(dict(file_name=img_path))]

        start_time = time.time()

        images = [x["image"].to(task_model.device) for x in inputs]
        images = [(x - task_model.pixel_mean) / task_model.pixel_std for x in images]   # normalization
        images = ImageList.from_tensors(images, task_model.backbone.size_divisibility)

        res2_ete_codec = torch.load('{}/res2_{}.pt'.format(feature_dir, img_id_str), map_location=device)
        # print(res2_ete_codec)
        res2_ete = res2_ete_codec
        res3_ete = task_model.backbone.bottom_up.res3(res2_ete)
        res4_ete = task_model.backbone.bottom_up.res4(res3_ete)
        res5_ete = task_model.backbone.bottom_up.res5(res4_ete)

        fpn_output = {'res2': res2_ete, 'res3': res3_ete, 'res4': res4_ete, 'res5': res5_ete}
        backbone_output = my_fpn_forward(fpn_output, task_model)
        proposals, _ = task_model.proposal_generator(images, backbone_output)
        results, _ = task_model.roi_heads(images, backbone_output, proposals, None)
        outputs = task_model._postprocess(results, inputs, images.image_sizes)

        # task_time = time.time() - start_time - decode_time
        # task_total_time += task_time

        predictions = outputs[0]

        # predictions = demo.run_on_image(img)
        # assert False
        instances_dict = {}
        instances_dict['path'] = img_path
        instances_dict['instances'] = []
        instances_dict['image_size'] = img.shape[:2]  # HxW
        for i in range(len(predictions['instances'])):
            instance_dict = parse_instance(predictions)
            instances_dict['instances'].append(instance_dict)

        tmp = json.dumps(instances_dict)


        with open(save_path, 'w') as f:
            f.write(tmp)
        logger.info(
            "{}: {} in {:.2f}s".format(
                img_path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

    coco_map_dict = load_coco_labels()
    # load predictions
    all_instances = load_instances(chosen_images, infer_json_dir, coco_map_dict, task_class, dataset_class)
    # convert predictions
    final_json_dir = args.final_json_dir
    # os.makedirs(final_json_dir, exist_ok=True)
    final_json_file = os.path.join(final_json_dir, args.infer_json_name)
    json.dump(all_instances, open(final_json_file, 'w'), indent=4)
    print('task done')
    dtm_v_task_end_time = time.time()
    dtm_v_task_all_time = dtm_v_task_end_time - dtm_v_task_start_time
    print("************************************************************************************")
    print("Result dtm_v_task_all_time:{} [s]".format(format(dtm_v_task_all_time, '.10f')))
    print("************************************************************************************")
