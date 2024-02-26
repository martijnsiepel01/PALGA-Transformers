#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --mem=40000M

source ../.venv/bin/activate


python PALGA-Transformers/work_in_progress_code/find_similar_palga_terms.py