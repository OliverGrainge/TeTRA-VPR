#!/bin/bash 

#SBATCH --job-name=salad
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/tetra/logs/salad.log
#SBATCH --error=jobfiles/tetra/logs/salad.err
#SBATCH --partition=a100

source tetraenv/bin/activate

python finetune.py --agg_arch salad --backbone_arch vit_base_PLRBitLinear --image_size 322 322 --weights_path checkpoints/Distill/backbone[vit_base_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[322x322]_aug[severe]_decay[constant]/epoch=2-Pitts30k_val_R1=93.49.ckpt --batch_size 80 --num_workers 12

python finetune.py --agg_arch salad --backbone_arch vit_base_PLRBitLinear --image_size 224 224 --weights_path checkpoints/Distill/backbone[vit_base_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[224x224]_aug[severe]_decay[constant]/epoch=2-Pitts30k_val_R1=91.85.ckpt --batch_size 80 --num_workers 12

python finetune.py --agg_arch salad --backbone_arch vit_small_PLRBitLinear --image_size 322 322 --weights_path checkpoints/Distill/backbone[vit_small_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[322x322]_aug[severe]_decay[constant]/epoch=2-Pitts30k_val_R1=91.40.ckpt --batch_size 80 --num_workers 12

python finetune.py --agg_arch salad --backbone_arch vit_small_PLRBitLinear --image_size 224 224 --weights_path checkpoints/Distill/backbone[vit_small_plrbitlinear]_agg[salad]_teacher[dinosalad]_res[224x224]_aug[severe]_decay[constant]/epoch=1-Pitts30k_val_R1=88.58.ckpt --batch_size 80 --num_workers 12