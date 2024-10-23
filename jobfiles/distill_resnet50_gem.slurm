#!/bin/bash 

#SBATCH --job-name=resdist
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/resdist.log
#SBATCH --error=jobfiles/logs/resdist.err
#SBATCH --partition=swarm_a100

source activate venv
python train.py --training_method distill --backbone_arch ResNet50 --agg_arch gem --teacher_preset EigenPlaces --batch_size 256 --num_workers 12 --max_epochs 2