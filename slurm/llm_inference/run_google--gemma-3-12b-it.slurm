#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%A_%a.out.log
#SBATCH --error=slurm-%A_%a.err.log
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH -C a100

source ~/.bashrc
conda activate hf

cd "${SLURM_SUBMIT_DIR:-.}"

srun python3 -u -m src.data.llm_inference  --model google/gemma-3-12b-it --batch-size 64
