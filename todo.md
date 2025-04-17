# Implementations
1. CPU
        - [x] CSR
2. GPU
        - [ ] CSR 1 thread per row
        - [ ] CSR 1 thread per val
                - [ ] add thread grid?
        - [ ] add shared mem

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
