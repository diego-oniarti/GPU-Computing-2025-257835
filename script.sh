make
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --partition=edu-short --job-name=homework --pty ./bin/main


