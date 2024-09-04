#!/bin/bash -l
export TAF_START=2024060606
export VER_DIR=/data/users/alanyon/tafs/improver/verification/20230804-20240804_ml
export PYTHONPATH=$PYTHONPATH:~alanyon/first_guess_TAFs/improver_ml/
CODE_DIR=/home/h04/alanyon/first_guess_TAFs/improver_ml

# Activate cloned sss environment
conda activate default_clone

# Navigate to code directory and run code
cd ${CODE_DIR}
python master/taf_master.py

# Deactivate environment
conda deactivate
