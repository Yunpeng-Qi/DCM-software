# training
CUDA_VISIBLE_DEVICES=1 python train_cube.py \
    --model RDT_CheckerCube --hyper-channels 192 192 192 \
    --lmbda 16384 \
    --lr 5e-5 \
    --total-iter 1000000 --multistep-milestones 1000000 \
    --eval-interval 100 \
    # --resume ./logs/cube_train/RDT_CheckerCube_checker_cube_mse16384/best_16384_.pth