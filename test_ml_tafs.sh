#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=4G
#SBATCH --ntasks=10
#SBATCH --output=/home/h04/alanyon/first_guess_TAFs/improver_ml/EGPA.out
#SBATCH --time=1200
#SBATCH --error=/home/h04/alanyon/first_guess_TAFs/improver_ml/EGPA.err

# Data directories
export OUTPUT_DIR=/data/users/alanyon/tafs/improver/verification/20230805-20241004_ml
export FAKE_DATE=20230908

# Define pythonpath
export PYTHONPATH=$PYTHONPATH:~alanyon/first_guess_TAFs/improver_ml/

# Activate cloned sss environment
conda activate default_clone

# Navigate to code directory and run code
cd /net/home/h04/alanyon/first_guess_TAFs/improver_ml
python ml/bust_adjust.py

# Deactivate environment
conda deactivate
