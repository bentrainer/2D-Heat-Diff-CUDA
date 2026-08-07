#!/bin/bash
# qsub -q instructional -l select=1:ncpus=48:ngpus=1 run.sh
module load cuda11.8/toolkit
cd "${PBS_O_WORKDIR:-.}"

# ./build/heat2d --backend cpu --steps 100 --Nx 8192 --Ny 8192

# ./build/heat2d --backend cuda-naive --verify
# ./build/heat2d --backend cuda-naive --Nx 1024 --Ny 2048 --verify

# ./build/heat2d --backend cuda-naive --steps 100 --Nx 4096 --Ny 4096

# ./build/heat2d --backend cuda-tiled --verify

# ./build/heat2d --backend cuda-naive --steps 100 --Nx 8192 --Ny 8192
# ./build/heat2d --backend cuda-tiled --steps 100 --Nx 8192 --Ny 8192

./build/heat2d --backend cuda-coarsen --verify

./build/heat2d --backend cuda-tiled --steps 100 --Nx 8192 --Ny 8192
./build/heat2d --backend cuda-coarsen --steps 100 --Nx 8192 --Ny 8192 --cfactor 1
./build/heat2d --backend cuda-coarsen --steps 100 --Nx 8192 --Ny 8192 --cfactor 2
./build/heat2d --backend cuda-coarsen --steps 100 --Nx 8192 --Ny 8192 --cfactor 3