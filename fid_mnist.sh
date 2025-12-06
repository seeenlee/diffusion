#!/bin/bash
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=J
#SBATCH --time=4:00:00
#SBATCH --mem=16G

source .venv/bin/activate
python3 -u evaluate.py