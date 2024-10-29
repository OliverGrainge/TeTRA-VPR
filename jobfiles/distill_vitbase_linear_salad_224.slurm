#!/bin/bash 

#SBATCH --job-name=Linear224_vitbase
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/Linear224_vitbase.log
#SBATCH --error=jobfiles/logs/Linear224_vitbase.err
#SBATCH --partition=swarm_a100

source activate venv
python train.py --training_method distill --image_size 224 224 --backbone_arch vit_base --agg_arch Salad --teacher_preset DinoSalad --batch_size 128 --num_workers 16 --max_epochs 3 --use_attention True --weight_decay_schedule constant --weight_decay_scale 0.05 --mse_loss_scale 1000