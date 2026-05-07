#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=4
#SBATCH --output=slurm-%A_%a.out.log
#SBATCH --error=slurm-%A_%a.err.log
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH -C a100

source ~/.bashrc
conda activate hf

cd "${SLURM_SUBMIT_DIR:-.}"

srun python3 -u -m src.data.llm_inference  --model meta-llama/Llama-4-Scout-17B-16E-Instruct --batch-size 16
