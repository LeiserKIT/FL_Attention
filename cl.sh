#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<E-mail address>

## activate virtual environment here and execute .py file
source <path to env>/bin/activate
python centralized_learning.py

exit 0