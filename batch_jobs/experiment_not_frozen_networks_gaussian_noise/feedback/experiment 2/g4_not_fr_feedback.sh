#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=20:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:4
source ../../../../../../tfds2/bin/activate
pwd
python 4block_to_4block.py