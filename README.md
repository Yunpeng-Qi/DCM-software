# Introduction
This is the official implementation of DTM-V.

# Installation

Download detectron2 and compressAI from the following link,then put them under `./packages/Environment/`.  
- detectron2  
https://codeload.github.com/facebookresearch/detectron2/zip/refs/tags/v0.5   
- compressAI  
https://codeload.github.com/InterDigitalInc/CompressAI/zip/refs/heads/master  

Download detectron2 Model from the following link,then put it under `./packages/Model/Detector_model/`.  
- model_final_68b088.pkl  
https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl

To install the DTM-V package, run

```
bash install.sh
```

# Usage
- To encode an image
```
CUDA_VISIBLE_DEVICES=1 python ./tools/main.py --ss-enabled-flag --ss-w-bboxes-flag --mode compress --input-files-path <image_file_path> --byte-stream-path <bitstream_file_path> --quality <1~4> --save <log_file_save_dir> 
```
- To decode a bitstream
```
CUDA_VISIBLE_DEVICES=1 python ./tools/main.py --mode decompress --byte-stream-path <bitstream_file_path> --output-files-path <rec_feature_path> --save <log_file_save_dir>
```  
- To inference by reconstructing features
```
cd ./scripts/Feature_inference_scripts
bash inference_task.sh
```  
There is a codec test script in `./scripts/demo.sh`.  

# Development  
- To use other feature adapters

```
--other_feature_adapter_flag
--other_feature_adapter_path <feature adapter model path> 
```
There is a codec test script in `./scripts/demo_use_other_feature_adapter.sh`.  
- To use scaling with a given size during image preprocessing  

```
--image_resize_xmin <image short edge scaling length> 
--image_resize_xmax <image scaling edge length threshold>
```
There is a codec test script in `./scripts/demo_image_resize.sh`.  
# Encoding/decoding DCM test dataset  
The scripts to encode/decode the DCM test dataset are in folder `./scripts/DCM_Datasets_Scripts`.    
# Release information
- Version 0.2.1  
  - Adjust the syntax structure of the bitstream.   
- Version 0.2 
  - Add image pre-processing scaling interface and feature adapter interface.  
- Version 0.1 
  - The initial version of the DTM-V common framework (DCM reference software).  

