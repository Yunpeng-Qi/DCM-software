# QP parameters
QP=22
# input yuv file
input_path=/work/Users/jiake/dcm_anchor/image_output/img2yuv
# orirginal image path, to calculate weight and height
ori_path=/work/Users/jiake/coco/val2017
# VTM-13.0 config file path
cfg=/work/Users/jiake/VTM-13.0/cfg/encoder_intra_vtm.cfg

python codec_multiprocess.py --QP ${QP} --input_path ${input_path} --ori_path ${ori_path} --cfg ${cfg}
