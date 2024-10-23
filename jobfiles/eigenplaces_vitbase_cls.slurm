#!/bin/bash 

#SBATCH --job-name=eigenplaces_vitbase_cls
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/eigenplaces_vitbase_cls.log
#SBATCH --error=jobfiles/logs/eigenplaces_vitbase_cls.err
#SBATCH --partition=a100


source activate venv
python train.py --training_method EigenPlaces --backbone_arch vit_base --agg_arch cls --image_size 224 224 --batch_size 128 --num_workers 16 --out_dim 512 --max_epochs 300