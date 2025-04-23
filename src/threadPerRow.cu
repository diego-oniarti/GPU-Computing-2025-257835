#include "common.h"
#include "timing.h"
#include "matrix.h"
#include <stdio.h>

__global__
void mult_per_row_kelner(data_t *vals, int *xs, int *ys, 
        data_t *vec, data_t *ret,
        int cols, int rows) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < rows) {
        data_t acc = 0;
        for (int i=ys[tid]; i<ys[tid+1]; i++) {
            acc += vals[i] * vec[xs[i]];
        }
        ret[tid] = acc;
    }
}

data_t* mult_per_row(MAT_CSR *csr, data_t *ones, int maxThreads) {
    int ROWS = csr->nrows;
    int COLS = csr->ncols;

    // Create as many threads as matrix rows
    int n_threads = csr->nrows;
    // Enough blocks to accomodate the threads
    int n_blocks = ceil((float)n_threads / maxThreads);

    // Put the data into managed memory to make it accessible by the GPU
    data_t *vals, *vec, *ret;
    int *xs, *ys;
    cudaMalloc(&vals, sizeof(data_t)*csr->nvals);
    cudaMemcpy(vals, csr->vals, sizeof(data_t)*csr->nvals, cudaMemcpyHostToDevice);
    cudaMalloc(&vec, sizeof(data_t)*COLS);
    cudaMemcpy(vec, ones, sizeof(data_t)*COLS, cudaMemcpyHostToDevice);
    cudaMallocManaged(&ret, sizeof(data_t)*ROWS);
    cudaMalloc(&xs, sizeof(int)*csr->nvals);
    cudaMemcpy(xs, csr->xs, sizeof(int)*csr->nvals, cudaMemcpyHostToDevice);
    cudaMalloc(&ys, sizeof(int)*(ROWS+1));
    cudaMemcpy(ys, csr->ys, sizeof(int)*(ROWS+1), cudaMemcpyHostToDevice);

    // Events and array for timing
    double times[RUNS];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int r=-PRERUNS; r<RUNS; r++) {
        cudaEventRecord(start);
        mult_per_row_kelner<<<n_blocks, maxThreads>>>(vals, xs, ys, vec, ret, COLS, ROWS);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaDeviceSynchronize();

        if (DOPRINTSINGLE) printf("--- Elapsed time: %lf\n", milliseconds);
        if (r>=0) { // Preruns
            times[r] = milliseconds;
        }
    }
    print_timing(times, RUNS, csr->nvals*2);

    cudaFree(vals);
    cudaFree(vec);
    cudaFree(xs);
    cudaFree(ys);

    return ret;
}
