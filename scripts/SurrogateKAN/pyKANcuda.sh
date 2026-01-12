#!/bin/bash 
#BSUB -n 1
#BSUB -W 12:00
#BSUB -J pyKAN_surrogate
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q gpu 
#BSUB -R rusage[mem=124.00/task]
#BSUB -R "select[rtx2080 || gtx1080 || p100 || a30]"
#BSUB -gpu "num=6:mode=shared:mps=yes"

module load cuda/12.1
module load conda

source activate base
conda activate KAN 
python sampleKAN.py
