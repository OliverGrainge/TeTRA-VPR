#!/bin/bash 

#SBATCH --job-name=schedules
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/finetune/ablations/logs/schedules.log
#SBATCH --error=jobfiles/finetune/ablations/logs/schedules.err
#SBATCH --partition=swarm_h100

source tetraenv/bin/activate
export WANDB_MODE=offline

python finetune.py --agg_arch boq --backbone_arch dinot --image_size 322 322 --batch_size 200 --num_workers 12 --freeze_backbone True  --quant_schedule logistic
python finetune.py --agg_arch boq --backbone_arch dinot --image_size 322 322 --batch_size 200 --num_workers 12 --freeze_backbone False --quant_schedule logistic

python finetune.py --agg_arch boq --backbone_arch dinoboqt --image_size 322 322 --batch_size 200 --num_workers 12 --freeze_backbone True  --quant_schedule logistic
python finetune.py --agg_arch boq --backbone_arch dinoboqt --image_size 322 322 --batch_size 200 --num_workers 12 --freeze_backbone False --quant_schedule logistic

