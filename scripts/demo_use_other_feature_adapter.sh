# This is a codec test script
cd ..
# compress
CUDA_VISIBLE_DEVICES=1 python ./tools/main.py \
    --ss-enabled-flag --ss-w-bboxes-flag \
    --mode compress \
    --input-files-path "./logs/samples/000000384616.jpg" \
    --byte-stream-path "./dtmV_output/bitstream/test/000000384616.bin" \
    --save "./logs/default/Encoder" \
    --quality 4  # codec quality (1~4)(high~low)

# decompress
CUDA_VISIBLE_DEVICES=1 python ./tools/main.py \
    --mode decompress \
    --byte-stream-path "./dtmV_output/bitstream/test/000000384616.bin" \
    --output-files-path "./dtmV_output/rec/test/000000384616.pt" \
    --save "./logs/default/Decoder" \
    --other_feature_adapter_flag --other_feature_adapter_path "./packages/Model/MIKcodec_model/ckp_step48000_ete_model_bpp10.pth"  # use other feature adapter




