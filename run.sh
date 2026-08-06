#!/bin/bash
# qsub -q instructional -l select=1:ncpus=48:ngpus=1 run.sh
module load cuda11.8/toolkit
cd "${PBS_O_WORKDIR:-.}"

./build/heat2d --backend cuda-naive --verify