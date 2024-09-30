#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=1G
#SBATCH --ntasks=10
#SBATCH --output=/home/h04/alanyon/first_guess_TAFs/improver_ml/EGTK.out
#SBATCH --time=300
#SBATCH --error=/home/h04/alanyon/first_guess_TAFs/improver_ml/EGTK.err

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
