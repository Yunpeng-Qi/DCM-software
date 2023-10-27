# This .sh is a template to generate dcm anchor of coco validation and openimages validation
# You can modify these pathes and you should choose dataset and task if you need

#########################You can change below########################
export CUDA_VISIBLE_DEVICES=7

# dataset name -- 0: coco, 1:openimages
dataset=0

# task name -- 0: object detction, 1: instance segemtation, 2: pose estimation
# coco can inference 0, 1, 2
# openimages can inference 0, 1
task=2

# inference json save path and name
infer_json_dir=./infer
infer_json_name='infer_json.json'

# validation images path
val_dir=/work/Users/jiake/dcm_anchor/image_output/recover/coco/raw/qp37
#val_dir=/work/Users/jiake/dcm_anchor/image_output/recover/openimages/raw/qp37
#####################################################################

if (($dataset==0)); then
  echo 'Start inference on coco validation dataset!'
  ann_file_name=instances_val2017.json
  if (($task==2)); then
    ann_file_name=person_keypoints_val2017.json
  fi
elif (($dataset==1)); then
  echo 'Start inference on openimages validation dataset!'
  ann_file_name=dcm_openimages_annotation.csv
else
  ann_file_name=none
fi

if (($task==0)); then
  echo 'Inference on Object Detection task!'
  config_name=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
  model_name=model_final_68b088.pkl
elif (($task==1)); then
  echo 'Inference on Instance Segmentation task!'
  config_name=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
  model_name=model_final_2d9806.pkl
elif (($task==2)); then
  echo 'Inference on Pose Estimation task!'
  config_name=COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml
  model_name=model_final_5ad38f.pkl
else
  config_name=none
  model_name=none
fi

######################You can change below path######################
config_path=/work/Users/jiake/detectron2-0.2.1/configs/${config_name}
model_file=/work/Users/jiake/detectron2-0.2.1/trained_model/${model_name}
ann_path=/work/Users/jiake/dcm_anchor/annotations/${ann_file_name}
#####################################################################

python inference.py --config_file $config_path --confidence_threshold 0.5 --score_threshold 0.05 \
--val_dir $val_dir --infer_json_dir $infer_json_dir --task $task --dataset $dataset \
--opts MODEL.WEIGHTS $model_file

echo $ann_path
python eval.py --anno_json_file $ann_path --task $task