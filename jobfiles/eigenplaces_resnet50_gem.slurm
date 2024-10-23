#!/bin/bash 

#SBATCH --job-name=res_gem
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/res_gem.log
#SBATCH --error=jobfiles/logs/res_gem.err
#SBATCH --partition=a100


source activate venv
python train.py --training_method EigenPlaces --val_set_names sf_xl_small_val --batch_size 128 --backbone_arch resnet50 --agg_arch gem --num_workers 12 --image_size 512 512 --max_epochs 60