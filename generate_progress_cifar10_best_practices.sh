#!/bin/bash
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=J
#SBATCH --time=4:00:00

source .venv/bin/activate
python3 -u generate_cifar10.py --config config/cifar10_best_practices.json --generate_progress
