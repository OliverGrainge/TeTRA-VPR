#!/bin/bash 

#SBATCH --job-name=vitbasePT_224
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=jobfiles/pretrain/logs/vitbasePT_224.log
#SBATCH --error=jobfiles/pretrain/logs/vitbasePT_224.err
#SBATCH --partition=swarm_a100

source tetraenv/bin/activate
export WANDB_MODE=offline
python pretrain.py --backbone_arch Vitbaset --weight_decay 0.05 --agg_arch Salad --teacher_preset DinoSalad --image_size 224 224 --batch_size 64 --accumulate_grad_batches 4 --num_workers 16 --max_epochs 5 --augmentation_level Severe --val_dataset_dir "/scratch/oeg1n18/datasets/vpr/" --train_dataset_dir "/scratch/oeg1n18/datasets/vpr/sf_xl/raw/train/panoramas" --use_progressive_quant