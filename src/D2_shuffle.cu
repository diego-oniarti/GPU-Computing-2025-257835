#include "common.h"
#include "timing.h"
#include "matrix.h"
#include "warpRow.h"
#include <stdio.h>

__global__
void D2_shuffle_kelner(data_t * __restrict__ vals,
        int * __restrict__ xs,
        int * __restrict__ ys, 
        data_t * __restrict__ vec,
        data_t * __restrict__ ret,
        int cols, int rows) {
    // Warp id
    int wid = threadIdx.x / 32;
    // Thread's place in the warp
    int friend_id = threadIdx.x % 32;
    // Row: blockid * rows_per_block + warpid
    int row = blockIdx.x * (blockDim.x / 32) + wid;

    if (row >= rows) return;

    int start = ys[row];
    int end = ys[row+1];
    data_t sum = 0; 
    for (int i=start+friend_id; i<end; i+=32) {
        // The access to the vector is not coalesced
        sum += vals[i] * __ldg(&vec[xs[i]]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (friend_id == 0) {
        ret[row] = sum;
    }
}

data_t* D2_shuffle(MAT_CSR *csr, data_t *ones, int threads_per_block) {
    int ROWS = csr->nrows;
    int COLS = csr->ncols;

    int warps_per_block = threads_per_block / 32;
    int n_blocks = (ROWS + warps_per_block - 1) / warps_per_block;

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
        D2_shuffle_kelner<<<n_blocks, threads_per_block>>>(vals, xs, ys, vec, ret, COLS, ROWS);
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
    print_timing(times, RUNS, csr->nvals*2, csr);

    cudaFree(vals);
    cudaFree(vec);
    cudaFree(xs);
    cudaFree(ys);

    return ret;
}
