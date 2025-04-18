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

data_t* mult_per_row(MAT_CSR *csr, data_t *ones) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;

    int n_threads = csr->nrows;
    int n_blocks = ceil((float)n_threads / maxThreads);

    data_t *vals, *vec, *ret;
    int *xs, *ys;
    cudaMallocManaged(&vals, sizeof(data_t)*csr->nvals);
    cudaMemcpy(vals, csr->vals, sizeof(data_t)*csr->nvals, cudaMemcpyHostToDevice);
    cudaMallocManaged(&vec, sizeof(data_t)*COLS);
    cudaMemcpy(vec, ones, sizeof(data_t)*COLS, cudaMemcpyHostToDevice);
    cudaMallocManaged(&ret, sizeof(data_t)*ROWS);
    cudaMallocManaged(&xs, sizeof(int)*csr->nvals);
    cudaMemcpy(xs, csr->xs, sizeof(int)*csr->nvals, cudaMemcpyHostToDevice);
    cudaMallocManaged(&ys, sizeof(data_t)*(ROWS+1));
    cudaMemcpy(ys, csr->ys, sizeof(data_t)*(ROWS+1), cudaMemcpyHostToDevice);

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

        printf("--- Elapsed time: %lf\n", milliseconds);
        if (r>=0) {
            times[r] = milliseconds;
        }
    }
    print_timing(times, RUNS);

    cudaFree(vals);
    cudaFree(vec);
    cudaFree(xs);
    cudaFree(ys);

    return ret;
}
