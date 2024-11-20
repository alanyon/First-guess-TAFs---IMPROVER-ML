#!/bin/bash -l

# Data directories
export OUTPUT_DIR=/data/users/alanyon/tafs/improver/verification/20230805-20241004_ml

# Define pythonpath
export PYTHONPATH=$PYTHONPATH:~alanyon/first_guess_TAFs/improver_ml/
export FAKE_DATE=None

# Activate cloned sss environment
conda activate default_clone

# Navigate to code directory and run code
cd /net/home/h04/alanyon/first_guess_TAFs/improver_ml
python ml/test_scores.py

# Deactivate environment
conda deactivate