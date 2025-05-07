#!/bin/bash 

#SBATCH --job-name=salad
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/finetune/logs/salad.log
#SBATCH --error=jobfiles/finetune/logs/salad.err
#SBATCH --partition=swarm_h100

source tetraenv/bin/activate
export WANDB_MODE=offline

BACKBONE_WEIGHT_PATH="checkpoints/TeTRA-pretrain/Student[VitbaseT322]-Teacher[DinoV2]-Aug[Severe]/epoch=17-step=136250-train_loss=0.0460-qfactor=1.00.ckpt"
QUANT_SCHEDULE="logistic"

python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 200 --num_workers 12 --desc_divider_factor 1 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule $QUANT_SCHEDULE
python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 200 --num_workers 12 --desc_divider_factor 2 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule $QUANT_SCHEDULE 
python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 200 --num_workers 12 --desc_divider_factor 4 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule $QUANT_SCHEDULE
python finetune.py --agg_arch salad --backbone_arch vitbaset --image_size 322 322 --batch_size 200 --num_workers 12 --desc_divider_factor 8 --freeze_backbone True --backbone_checkpoint $BACKBONE_WEIGHT_PATH --quant_schedule $QUANT_SCHEDULE

