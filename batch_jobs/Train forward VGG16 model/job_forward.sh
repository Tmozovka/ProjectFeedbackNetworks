#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=03:00:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:4
source ../../../../tfds2/bin/activate
python train_forward_model.py