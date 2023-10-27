
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install opencv-python einops torchac tqdm scipy pycocotools
pip install ninja

# install detectron2
cd ./packages/Environment
unzip detectron2-0.5
cd detectron2-0.5
pip install -e .
cd ../../../

# install CompressAI
cd ./packages/Environment
unzip CompressAI-master
cd CompressAI-master
pip install -e .
cd ../../../