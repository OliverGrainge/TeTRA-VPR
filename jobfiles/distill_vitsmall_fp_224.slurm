#!/bin/bash 

#SBATCH --job-name=vitsmall_fp_224
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/vitsmall_fp_224.log
#SBATCH --error=jobfiles/logs/vitsmall_fp_224.err
#SBATCH --partition=a100

source activate venv
python pretrain.py --backbone_arch vit_small --agg_arch Salad --teacher_preset DinoSalad --image_size 224 224 --batch_size 32 --num_workers 17 --max_epochs 3 --use_attention True --augmentation_level Moderate --val_dataset_dir "" --train_dataset_dir ""