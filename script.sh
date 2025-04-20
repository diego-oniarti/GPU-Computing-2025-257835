#!/bin/bash

make

if [[ $? -eq 0 ]]
then
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --partition=edu-short --job-name=homework --pty ./bin/main
else
    echo "Make failed"
fi


