#!/bin/bash 

#SBATCH --job-name=salad
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/finetune/logs/salad.log
#SBATCH --error=jobfiles/finetune/logs/salad.err
#SBATCH --partition=swarm_a100

source tetraenv/bin/activate
export WANDB_MODE=offline
python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 100 --num_workers 12 --desc_divider_factor 1

python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 224 224 --batch_size 100 --num_workers 12 --desc_divider_factor 1

python finetune.py --agg_arch salad --backbone_arch vitsmallt --image_size 322 322 --batch_size 100 --num_workers 12 --desc_divider_factor 1

python finetune.py --agg_arch salad --backbone_arch vitsmallt --image_size 224 224 --batch_size 100 --num_workers 12 --desc_divider_factor 1



python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 100 --num_workers 12 --desc_divider_factor 2

python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 224 224 --batch_size 100 --num_workers 12 --desc_divider_factor 2

python finetune.py --agg_arch salad --backbone_arch vitsmallt --image_size 322 322 --batch_size 100 --num_workers 12 --desc_divider_factor 2

python finetune.py --agg_arch salad --backbone_arch vitsmallt --image_size 224 224 --batch_size 100 --num_workers 12 --desc_divider_factor 2



python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 100 --num_workers 12 --desc_divider_factor 4

python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 224 224 --batch_size 100 --num_workers 12 --desc_divider_factor 4

python finetune.py --agg_arch salad --backbone_arch vitsmallt --image_size 322 322 --batch_size 100 --num_workers 12 --desc_divider_factor 4

python finetune.py --agg_arch salad --backbone_arch vitsmallt --image_size 224 224 --batch_size 100 --num_workers 12 --desc_divider_factor 4

