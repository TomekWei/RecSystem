#!/bin/bash
#SBATCH --job-name=dlrm_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4 
#SBATCH --mem=64G #
#SBATCH --cpus-per-task=4 
#SBATCH --time=02:00:00 
#SBATCH --partition=gpuA100x4 

conda activate DLRCs
export PREPROCESSED_DATASET=/scratch/bcjw/twei1/RecSystem/data/output
export GLOBAL_BATCH_SIZE=16384
export WORLD_SIZE=8

torchx run -s slurm dist.ddp -j 1x4 --script /u/twei1/Projects/DLRCs/RecSystem/torchrec_dlrm/dlrm_main.py -- \
    --in_memory_binary_criteo_path $PREPROCESSED_DATASET \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --learning_rate 1.0 \
    --dataset_name criteo_kaggle
