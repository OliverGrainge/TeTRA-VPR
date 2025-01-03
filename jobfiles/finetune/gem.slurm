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

python finetune.py --agg_arch gem --backbone_arch vit_base_PLRBitLinear --image_size 322 322 --weights_path checkpoints/Distill/backbone[vit_base_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[322x322]_aug[severe]_decay[constant]/epoch=3-msls_R1=85.60.ckpt --batch_size 100 --num_workers 12

python finetune.py --agg_arch gem --backbone_arch vit_base_PLRBitLinear --image_size 224 224 --weights_path checkpoints/Distill/backbone[vit_base_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[224x224]_aug[severe]_decay[constant]/epoch=3-msls_R1=81.04.ckpt --batch_size 100 --num_workers 12

python finetune.py --agg_arch gem --backbone_arch vit_small_PLRBitLinear --image_size 322 322 --weights_path checkpoints/Distill/backbone[vit_small_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[322x322]_aug[severe]_decay[constant]/epoch=2-msls_R1=82.14.ckpt --batch_size 100 --num_workers 12

python finetune.py --agg_arch gem --backbone_arch vit_small_PLRBitLinear --image_size 224 224 --weights_path checkpoints/Distill/backbone[vit_small_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[224x224]_aug[severe]_decay[constant]/epoch=1-msls_R1=77.87.ckpt --batch_size 100 --num_workers 12