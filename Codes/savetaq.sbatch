#!/usr/bin/env sh

#SBATCH --partition=broadwl
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=savetaq
#SBATCH --output=savetaq.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5g

module load python/3.7.0
python NewAPI_Save_TAQ.py