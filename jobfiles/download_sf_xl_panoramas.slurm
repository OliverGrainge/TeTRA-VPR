#!/bin/bash 

#SBATCH --job-name=download_sf_xl_panoramas
#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node=1
#SBATCH --time=59:00:00
#SBATCH --mem=32G
#SBATCH --output=jobfiles/logs/download_sf_xl_panoramas.log
#SBATCH --error=jobfiles/logs/download_sf_xl_panoramas.err
#SBATCH --partition=amd


rsync -rvz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/raw/train/panoramas/ /scratch/oeg1n18/datasets/vpr/sf_xl/raw/train/panoramas/