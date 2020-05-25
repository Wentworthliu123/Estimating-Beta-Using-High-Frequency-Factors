#!/usr/bin/env sh

#SBATCH --partition=broadwl
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=nconcat
#SBATCH --output=nconcat.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=55g

module load python/3.7.0
python pre2step.py