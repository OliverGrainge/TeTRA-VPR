#!/bin/bash 

#SBATCH --job-name=vitbaseT_224
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=jobfiles/pretrain/logs/vitbaseT_224.log
#SBATCH --error=jobfiles/pretrain/logs/vitbaseT_224.err
#SBATCH --partition=a100

source tetraenv/bin/activate
python pretrain.py --backbone_arch VitbaseT --agg_arch Salad --weight_decay 0.00 --teacher_preset DinoSalad --image_size 224 224 --batch_size 64 --accumulate_grad_batches 4 --num_workers 16 --max_epochs 3 --augmentation_level Severe --val_dataset_dir "/scratch/oeg1n18/datasets/vpr/" --train_dataset_dir "/scratch/oeg1n18/datasets/vpr/sf_xl/raw/train/panoramas" --use_attention