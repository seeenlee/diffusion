#!/bin/bash
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=J
#SBATCH --time=2:00:00

source .venv/bin/activate
python3 generate_mnist.py --generate_progress
