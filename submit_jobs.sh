#! /bin/bash

# plain 
sbatch jobfiles/pretrain/vitbase_224.slurm
sbatch jobfiles/pretrain/vitbase_322.slurm
sbatch jobfiles/pretrain/vitsmall_224.slurm
sbatch jobfiles/pretrain/vitsmall_322.slurm

# progressive quant
sbatch jobfiles/pretrain/vitbasePT_224.slurm
sbatch jobfiles/pretrain/vitbasePT_322.slurm
sbatch jobfiles/pretrain/vitsmallPT_224.slurm
sbatch jobfiles/pretrain/vitsmallPT_322.slurm

# constant quant
sbatch jobfiles/pretrain/vitbaseT_224.slurm
sbatch jobfiles/pretrain/vitbaseT_322.slurm
sbatch jobfiles/pretrain/vitsmallT_224.slurm
sbatch jobfiles/pretrain/vitsmallT_322.slurm


