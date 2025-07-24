#!/bin/bash
#SBATCH --job-name md
#SBATCH --output=md.txt
#SBATCH --mem=2G
#SBATCH --time=7:59:00
#SBATCH --gres=gpu:1

python run.py