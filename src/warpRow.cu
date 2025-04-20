#include "common.h"
#include "timing.h"
#include "matrix.h"
#include "warpRow.h"
#include <stdio.h>

__global__
void mult_warp_row_kelner(data_t *vals, int *xs, int *ys, 
        data_t *vec, data_t *ret,
        int cols, int rows, data_t *buffer) {
    // Thread id
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    // Warp id
    int wid = threadIdx.x / 32;
    // Thread's place in the warp
    int friend_id = threadIdx.x % 32;
    // Row: blockid * rows_per_block + warpid
    int row = blockIdx.x * (blockDim.x / 32) + wid;

    buffer[tid] = 0; // Should this be out of the if?
    if (row < rows) {
        int start = ys[row];
        int end = ys[row+1];

        // The sum of the elements taken by this thread. In case there are more values in
        // a row than threads in a warp
        data_t sum = 0; 
        for (int i=start+friend_id; i<end; i+=32) {
            // The access to the vector is not coalesced
            sum += vals[i] * vec[xs[i]];
        }

        buffer[tid] = sum;
    }

    for (int s=1; s<32; s<<=1) {
        __syncthreads();
        if (tid & ((s<<1)-1) == 0) {
            buffer[tid] += buffer[tid+s];
        }
    }

    if (friend_id==0) {
        ret[row] = buffer[tid];
    }
}

data_t* mult_warp_row(MAT_CSR *csr, data_t *ones, int threads_per_block) {
    int warps_per_block = threads_per_block / 32;
    int n_blocks = ceil((float)ROWS / warps_per_block);
    int n_threads = threads_per_block * n_blocks;       // R * 32?

    // Put the data into managed memory to make it accessible by the GPU
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

    data_t *buffer; // This should be shared memory
    cudaMallocManaged(&buffer, sizeof(data_t) * n_threads); // Can be not managed

    // Events and array for timing
    double times[RUNS];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int r=-PRERUNS; r<RUNS; r++) {
        cudaEventRecord(start);
        mult_warp_row_kelner<<<n_blocks, threads_per_block>>>(vals, xs, ys, vec, ret, COLS, ROWS, buffer);
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
    print_timing(times, RUNS, 0);

    cudaFree(vals);
    cudaFree(vec);
    cudaFree(xs);
    cudaFree(ys);

    return ret;
}
