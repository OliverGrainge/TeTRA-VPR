#!/bin/bash 


#SBATCH --job-name=imagenet_ternary_vit_small
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ouput=jobfiles/logs/imagenet_ternary_vit_small.log
#SBATCH --error=jobfiles/logs/imagenet_ternary_vit_small.err
#SBATCH --partition=a100

python pretrain.py --training_method imagenet --backbone_arch ternary_vit_small --agg_arch cls 
