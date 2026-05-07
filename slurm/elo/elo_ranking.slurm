#!/bin/bash
#SBATCH --job-name=elo_llama70b
#SBATCH --output=slurm/elo/logs/elo_eval_%j.out
#SBATCH --error=slurm/elo/logs/elo_eval_%j.err
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

mkdir -p slurm/elo/logs

source ~/.bashrc
conda activate steering-venv

models=(
  'Qwen__Qwen2.5-14B-it'
  'Qwen__Qwen2.5-3B-it'
  'google__gemma-3-12b-it'
  'google__gemma-3-4b-it'
  'llama4_scout-it'
  'meta-llama__Llama-3.1-8B-it'
  'mistralai__Ministral-3-14B-2512-it'
  'mistralai__Mistral-7B-it'
)


for model in "${models[@]}"; do
	python -m src.data.elo_ranking --model_name "$model"
done
