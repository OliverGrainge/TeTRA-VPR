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

BACKBONE_WEIGHT_PATH="checkpoints/TeTRA-pretrain/Student[VitbaseT322]-Teacher[DinoV2]-Aug[Severe]/epoch=17-step=136250-train_loss=0.0460-qfactor=1.00.ckpt"
python finetune.py --agg_arch boq --backbone_arch vitbaset --image_size 322 322 --batch_size 200 --num_workers 12 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule linear 
python finetune.py --agg_arch boq --backbone_arch vitbaset --image_size 322 322 --batch_size 200 --num_workers 12 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule cosine

