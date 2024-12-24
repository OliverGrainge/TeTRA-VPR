#!/bin/bash 

#SBATCH --job-name=vitsmall_224
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/pretrain/boq/logs/vitsmall_224.log
#SBATCH --error=jobfiles/pretrain/boq/logs/vitsmall_224.err
#SBATCH --partition=swarm_a100

source tetraenv/bin/activate
python pretrain.py --backbone_arch vit_small_PLRBitLinear --agg_arch BoQ --teacher_preset DinoV2_BoQ --weight_decay_schedule sigmoid --image_size 224 224 --batch_size 64 --accumulate_grad_batches 4 --num_workers 16 --max_epochs 4 --use_attention True --augmentation_level Severe --val_dataset_dir "/scratch/oeg1n18/datasets/vpr/" --train_dataset_dir "/scratch/oeg1n18/datasets/vpr/sf_xl/raw/train/panoramas"