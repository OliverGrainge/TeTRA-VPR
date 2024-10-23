#!/bin/bash 

#SBATCH --job-name=eigenplaces_imagenetvitbase_cls
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=jobfiles/logs/eigenplaces_imagenetvitbase_cls.log
#SBATCH --error=jobfiles/logs/eigenplaces_imagenetvitbase_cls.err
#SBATCH --partition=swarm_a100


source activate venv
python train.py --training_method EigenPlaces --preset ImageNetViT --image_size 224 224 --batch_size 128 --num_workers 16 --max_epochs 300