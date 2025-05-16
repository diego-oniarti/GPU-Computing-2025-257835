#!/bin/bash

# Load CUDA module if not already loaded
module is-loaded CUDA/12.5.0 || module load CUDA/12.5.0

# Build the project
if ! make; then
    echo "Make failed"
    exit 1
fi

srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
    --output=logs/output.txt --error=logs/error.txt \
    --partition=edu-short --job-name=homework ./bin/main $1

echo "////ERRORS////"
cat logs/error.txt
echo "////OUTPUT////"
cat logs/output.txt

