#!/bin/bash 

#SBATCH --job-name=vitdist
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/vitdist.log
#SBATCH --error=jobfiles/logs/vitdist.err
#SBATCH --partition=swarm_a100

source activate venv
python train.py --training_method distill --backbone_arch vit_base --agg_arch Salad --teacher_preset DinoSalad --batch_size 256 --num_workers 16 --max_epochs 3