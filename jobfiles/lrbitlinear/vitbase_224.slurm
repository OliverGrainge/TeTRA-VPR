#!/bin/bash 

#SBATCH --job-name=vitbase_224
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/lrbitlinear/logs/vitbase_224.log
#SBATCH --error=jobfiles/lrbitlinear/logs/vitbase_224.err
#SBATCH --partition=swarm_a100

source tetraenv/bin/activate
python pretrain.py --backbone_arch vit_base_LRBitLinear --agg_arch Salad --teacher_preset DinoSalad --image_size 224 224 --batch_size 64 --accumulate_grad_batches 4 --num_workers 16 --max_epochs 3 --use_attention True --augmentation_level Severe --val_dataset_dir "/scratch/oeg1n18/datasets/vpr/" --train_dataset_dir "/scratch/oeg1n18/datasets/vpr/sf_xl/raw/train/panoramas"