#!/bin/bash 
#BSUB -n 1
#BSUB -W 12:00
#BSUB -J pyKAN_surrogate
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=251]"
#BSUB -o stdout.%J
#BSUB -e stderr.%J

module load conda

source activate base
conda activate KAN 
python sampleKAN.py
