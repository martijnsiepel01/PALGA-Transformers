#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --cpus-per-task=18
#SBATCH --time=5:00:00
#SBATCH --mem=40000M

source ../.venv/bin/activate


python PALGA-Transformers/train.py "$@"