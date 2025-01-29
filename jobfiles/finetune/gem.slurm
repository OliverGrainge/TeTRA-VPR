#!/bin/bash 

#SBATCH --job-name=gem
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/finetune/logs/gem.log
#SBATCH --error=jobfiles/finetune/logs/gem.err
#SBATCH --partition=swarm_a100

source tetraenv/bin/activate
export WANDB_MODE=offline
python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 322 322 --batch_size 128 --num_workers 12 --desc_divider_factor 1

