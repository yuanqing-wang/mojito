#!/bin/bash
#SBATCH --job-name=mojito
#SBATCH --output=%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=11:59:00
# Usage: sbatch --account=$ACCOUNT scripts/run.sbatch DATA.csv OUT_DIR [args]
set -euo pipefail
python scripts/run.py --data "$1" --out "$2" "${@:3}"
