#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=<name>.txt
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=8GB
#SBATCH --ntasks=1

export TOKENIZERS_PARALLELISM=true

python analysis.py
