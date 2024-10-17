#!/bin/bash 

#SBATCH --job-name=resnet18_gem
#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=jobfiles/logs/resnet18_gem.log
#SBATCH --error=jobfiles/logs/resnet18_gem.err
#SBATCH --partition=a100


source activate venv
python train.py --training_method Distill --backbone_arch resnet18 --agg_arch gem --batch_size 128 --image_size 224 224 --num_workers 8 --teacher_preset DinoSalad --max_epochs 500