#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --cpus-per-task=18
#SBATCH --time=0:10:00
#SBATCH --mem=40000M

source ../.venv/bin/activate


python PALGA-Transformers/inference/calculate_test_scores_t5.py