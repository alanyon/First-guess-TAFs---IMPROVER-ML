#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=16G
#SBATCH --ntasks=8
#SBATCH --output=/home/h04/alanyon/first_guess_TAFs/improver_ml/ml.out
#SBATCH --time=4320
#SBATCH --error=/home/h04/alanyon/first_guess_TAFs/improver_ml/ml.err

# Data directories
export OUTPUT_DIR=/data/users/alanyon/tafs/improver/verification/20230805-20240805_ml

# Define pythonpath
export PYTHONPATH=$PYTHONPATH:~alanyon/first_guess_TAFs/improver_ml/

# Activate cloned sss environment
conda activate default_clone

# Navigate to code directory and run code
cd /net/home/h04/alanyon/first_guess_TAFs/improver_ml
python ml/train_busts.py

# Deactivate environment
conda deactivate
