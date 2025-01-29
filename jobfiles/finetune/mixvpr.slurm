#!/bin/bash 

#SBATCH --job-name=mixvpr
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/finetune/logs/mixvpr.log
#SBATCH --error=jobfiles/finetune/logs/mixvpr.err
#SBATCH --partition=swarm_a100

source tetraenv/bin/activate
export WANDB_MODE=offline
python finetune.py --agg_arch mixvpr --backbone_arch vitbaset --image_size 322 322 --batch_size 128 --num_workers 12 --desc_divider_factor 1

