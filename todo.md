# Implementations
1. CPU
    - [x] CSR
2. GPU
    - [x] CSR 1 thread per row
    - [-] CSR 1 thread per val
    - [-] CSR 1 warp per row
    - [x] add shared mem
        - [x] for the buffer
        - [ ] for the vector
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

From this point on I started noticing some small errors in the results, which I hope can
be attributed to rounding and floating point precision.
Some facts that make me believe this:
- The errors only manifest when the matrix is sufficiently big. This could mean a certain
  amount of operations is needed before the errors build up
- The errors become smaller when moving from flaot to double, and vanish when moving to
  int

## Warp per row with shared memory
There are two data structures that are accessed frequently and in a shared manner by
the threads:
1. The vector
2. The buffer for the sum
Hence, these should be stored in shared memory.

The elements in the matrix are only accessed once, so storing them would be a waste of
time.
