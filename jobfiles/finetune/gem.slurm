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
#SBATCH --partition=a100

source tetraenv/bin/activate

python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 322 322 --batch_size 80 --num_workers 12 --desc_divider_factor 1

python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 224 224 --batch_size 80 --num_workers 12 --desc_divider_factor 1

python finetune.py --agg_arch gem --backbone_arch vitsmallt --image_size 322 322 --batch_size 80 --num_workers 12 --desc_divider_factor 1

python finetune.py --agg_arch gem --backbone_arch vitsmallt --image_size 224 224 --batch_size 80 --num_workers 12 --desc_divider_factor 1



python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 322 322 --batch_size 80 --num_workers 12 --desc_divider_factor 2

python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 224 224 --batch_size 80 --num_workers 12 --desc_divider_factor 2

python finetune.py --agg_arch gem --backbone_arch vitsmallt --image_size 322 322 --batch_size 80 --num_workers 12 --desc_divider_factor 2

python finetune.py --agg_arch gem --backbone_arch vitsmallt --image_size 224 224 --batch_size 80 --num_workers 12 --desc_divider_factor 2


python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 322 322 --batch_size 80 --num_workers 12 --desc_divider_factor 4

python finetune.py --agg_arch gem --backbone_arch vitbaset --image_size 224 224 --batch_size 80 --num_workers 12 --desc_divider_factor 4

python finetune.py --agg_arch gem --backbone_arch vitsmallt --image_size 322 322 --batch_size 80 --num_workers 12 --desc_divider_factor 4

python finetune.py --agg_arch gem --backbone_arch vitsmallt --image_size 224 224 --batch_size 80 --num_workers 12 --desc_divider_factor 4