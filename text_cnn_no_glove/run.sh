#!/usr/bin/env bash

#SBATCH --time=0-10:00:00
#SBATCH --output=job_output-%j.txt
#SBATCH --gres=gpu:1

python train.py
