# Implementations
1. CPU
    - [x] CSR
2. GPU
    - [x] CSR 1 thread per row
    - [-] CSR 1 thread per val
    - [-] CSR 1 warp per row
    - [ ] add shared mem
    - [ ] add thread grid?

# Data
For each method:
- 3 warmup runs
- 10 timed runs
- mean and standard derivation
- uniform sparse matrix
- some funky distribution (e.g. only the bottom right corner filled in)

More complex to get:
- FLOPs
- Bottlenecks

# Rundown
The first implementation is just a control case running on the CPU.

## Threads per row
The first GPU implementation creates a thread per row. Each of these threads calculates
the result for its own row and stores it in the output vector.
It is already much faster than the CPU implementation, but each thread still has to
linearly iterate through the row one element at a time.

> Discussion about the block size

## Threads per value
My first idea for a better GPU implementation was to assign a thread to each
value, having the threads perform the multiplication with the vector, and then
coordinating them to reduce the result into a single value.
I was unable to implement this, as the synchronization of the threads between block
boundaries was too difficult.

## warp per row
A more structured approach is to assign each row to a warp. This solves the
synchronization problem encountered in the previous case, since we can define each warp
to be fully contained in the same block. 
Forthermore, each warp executes in lock-step, meaning that each row can execute
independently regardles of the number of elements contained
