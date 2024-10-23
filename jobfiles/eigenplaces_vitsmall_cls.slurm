#!/bin/bash 

#SBATCH --job-name=eigenplaces_vitsmall_cls_lrlarge
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/eigenplaces_vitsmall_cls_lrlarge.log
#SBATCH --error=jobfiles/logs/eigenplaces_vitsmall_cls_lrlarge.err
#SBATCH --partition=a100


source activate venv
python train.py --training_method EigenPlaces --backbone_arch vit_small --agg_arch cls --image_size 224 224 --batch_size 256 --num_workers 16 --out_dim 512 --max_epochs 300