#!/bin/bash

# Load CUDA module if not already loaded
module is-loaded CUDA/12.1.1 || module load CUDA/12.1.1

# Build the project
if make; then
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
         --partition=edu-short --job-name=homework --pty ./bin/main
else
    echo "Make failed"
fi

