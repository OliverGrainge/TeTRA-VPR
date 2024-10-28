#!/bin/bash 

#SBATCH --job-name=BitLinear_vitbase1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/BitLinear_vitbase1.log
#SBATCH --error=jobfiles/logs/BitLinear_vitbase1.err
#SBATCH --partition=swarm_a100

source activate venv
python train.py --training_method distill --backbone_arch vit_BitLinear_base --agg_arch Salad --teacher_preset DinoSalad --batch_size 128 --num_workers 16 --max_epochs 3 --use_attention True --weight_decay_schedule staged_linear --weight_decay_scale 0.05 --mse_loss_scale 100