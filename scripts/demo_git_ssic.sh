# This is a codec test script
cd ..
# compress
CUDA_VISIBLE_DEVICES=3 python ./tools/git_ssic.py \
    --ss-enabled-flag --ss-w-bboxes-flag \
    --mode compress \
    --input-files-path "./logs/samples/000000384616.jpg" \
    --byte-stream-path "./dtmV_output/bitstream/test/000000384616.bin" \
    --save "./logs/git_ssic/Encoder" \   # save log dir
    --quality 4  # codec quality (1~4)(high~low)

# decompress
CUDA_VISIBLE_DEVICES=3 python ./tools/git_ssic.py \
    --mode decompress \
    --byte-stream-path "./dtmV_output/bitstream/test/000000384616.bin" \
    --output-files-path "./dtmV_output/rec/test/000000384616.pt" \
    --save "./logs/git_ssic/Decoder"

# pip --default-timeout=1688 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

CUDA_VISIBLE_DEVICES=6 python tools/train.py \
    --model ours_groupswin_channelar --hyper-channels 192 192 192 \
    --lmbda 512 \
    --lr 5e-5 \
    --groupvit-load-group-msk logs/visualization/group_msk \
    --resume logs/default/ours_groupswin_channelar/512/best.pth --reset-rdo \
    --total-iteration 800000 --multistep-milestones 600000 
