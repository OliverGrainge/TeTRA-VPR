#!/bin/bash 

#SBATCH --job-name=vit_base_salad_attn
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/vit_base_salad_attn.log
#SBATCH --error=jobfiles/logs/vit_base_salad_attn.err
#SBATCH --partition=swarm_a100

source activate venv
python train.py --training_method distill --backbone_arch vit_base --agg_arch Salad --teacher_preset DinoSalad --batch_size 128 --num_workers 16 --max_epochs 3 --use_attention True