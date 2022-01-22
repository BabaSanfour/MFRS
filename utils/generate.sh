#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --nodes=2
#SBATCH --time=24:10:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2

source /home/hamza97/venv/bin/activate
python  generate_csv.py
