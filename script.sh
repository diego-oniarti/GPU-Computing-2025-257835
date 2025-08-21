#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt
#SBATCH --partition=edu-short
#SBATCH --job-name=homework
#SBATCH --mail-type=END,FAIL

module is-loaded CUDA/12.5.0 || module load CUDA/12.5.0

if ! make; then
    echo "Make failed"
    exit 1
fi

if [ $# -gt 0 ]; then
    ./bin/main "$1"
else
    ./bin/main
fi
