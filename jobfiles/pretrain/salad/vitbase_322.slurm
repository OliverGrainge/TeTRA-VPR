#!/bin/bash 

#SBATCH --job-name=vitbase_322

#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/pretrain/salad/logs/vitbase_322.log
#SBATCH --error=jobfiles/pretrain/salad/logs/vitbase_322.err
#SBATCH --partition=swarm_h100

source tetraenv/bin/activate
python pretrain.py --backbone_arch vit_base_PLRBitLinear --agg_arch Salad --teacher_preset DinoSalad --weight_decay_schedule constant --image_size 322 322 --batch_size 32 --accumulate_grad_batches 8 --num_workers 16 --max_epochs 4 --use_attention False --augmentation_level Severe --val_dataset_dir "/scratch/oeg1n18/datasets/vpr/" --train_dataset_dir "/scratch/oeg1n18/datasets/vpr/sf_xl/raw/train/panoramas"