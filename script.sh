#!/usr/bin/env bash

nvidia-smi
sudo chmod -R 777 *.py

export volna="."
export NGPUS=4
export OUTPUT_PATH="."
export snapshot_dir=$OUTPUT_PATH/output

export batch_size=16
export snapshot_iter=1

PATH+=:/home/user/miniconda/bin

export learning_rate=0.001 
export nepochs=200

export network='unet'
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$NGPUS --master_port=$MASTER_PORT train_aac.py

