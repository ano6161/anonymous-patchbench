#!/bin/bash
#SBATCH --job-name=wildguard_label
#SBATCH --output=slurm/wildguard/logs/wildguard_%j.out
#SBATCH --error=slurm/wildguard/logs/wildguard_%j.err
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

mkdir -p slurm/wildguard/logs

source ~/.bashrc
conda activate steering-venv

python3 -u -m src.data.wildguard_label
