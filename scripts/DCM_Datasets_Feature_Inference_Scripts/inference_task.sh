# This is the inference script for the dtm-v reconstruction feature
#####################Parameters you need to modify########################
export CUDA_VISIBLE_DEVICES=3

# dataset name -- 0: coco, 1:openimages
dataset=0

# task name -- 0: object detction, 1: instance segemtation, 2: pose estimation
# coco can inference 0, 1, 2
# openimages can inference 0, 1
task=1

# quality can be 1, 2, 3, 4
quality=2

# dtm-v reconstruct features path
feature_dir='../../dtmV_output/rec/openimages/2'

# validation images path
# val_dir='/Your/Path/openimages/openimages-val-dataset'
val_dir = '/data/qiyp/DTM-V/test_img'

# inference json save path and name
infer_json_dir=./infer/${dataset}/${task}/${quality}/infer
final_json_dir=./infer/${dataset}/${task}/${quality}/output/json

# # 检查 infer_json_dir 是否存在，如果不存在则创建
# if [ ! -d "$infer_json_dir" ]; then
#     mkdir -p "$infer_json_dir"
#     echo "Created directory: $infer_json_dir"
# fi

# # 检查 final_json_dir 是否存在，如果不存在则创建
# if [ ! -d "$final_json_dir" ]; then
#     mkdir -p "$final_json_dir"
#     echo "Created directory: $final_json_dir"
# fi

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

#################Parameters you need to modify########################
config_path=/data/qiyp/DTM-V/packages/Environment/detectron2-0.5/configs/${config_name}
model_file=/data/qiyp/DTM-V/packages/Environment/detectron2-0.5/trained_model/${model_name}
ann_path=/data/qiyp/detectron2/datasets/coco/annotations/${ann_file_name}


#####################################################################

python inference_task.py --config_file $config_path --confidence_threshold 0.5 --score_threshold 0.05 \
--val_dir $val_dir --infer_json_dir $infer_json_dir --final_json_dir $final_json_dir --task $task --dataset $dataset \
--feature_dir $feature_dir \
--opts MODEL.WEIGHTS $model_file

echo $ann_path
python eval.py --anno_json_file $ann_path --final_json_dir $final_json_dir --task $task