#!/bin/bash 


#SBATCH --job-name=tvitsmall
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=2
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/tvitsmall.log
#SBATCH --error=jobfiles/logs/tvitsmall.err
#SBATCH --partition=a100

srun python pretrain.py --training_method imagenet --backbone_arch ternary_vit_small --agg_arch cls 
