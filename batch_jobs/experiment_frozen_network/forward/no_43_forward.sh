#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=10:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:4
source ../../../../../tfds2/bin/activate
pwd
python forward_without_two_and_three_last_blocks.py