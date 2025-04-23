#include "common.h"
#include "timing.h"
#include "matrix.h"
#include "warpRowShared.h"
#include <stdio.h>

__global__
void kernel(data_t *vals, int *xs, int *ys, 
        data_t *vec, data_t *ret,
        int cols, int rows) {
    // Thread id
    int tid = threadIdx.x;
    // Warp id
    int wid = tid / 32;
    // Thread's place in the warp
    int friend_id = tid % 32;
    // Row: blockid * rows_per_block + warpid
    int row = blockIdx.x * (blockDim.x / 32) + wid;

    extern __shared__ data_t smem[]; //One shared memory for both buffer and vals
    data_t *buffer = smem;
    //int offset = blockDim.x; // Size of the buffer
    // data_t *shared_vec = smem+offset;

    buffer[tid] = 0;
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
        if ((tid & ((s<<1)-1)) == 0) {
            buffer[tid] += buffer[tid+s];
        }
    }

    if (friend_id==0 && row<rows) {
        ret[row] = buffer[tid];
    }
}

data_t* mult_warp_row_shared(MAT_CSR *csr, data_t *ones, int threads_per_block) {
    int ROWS = csr->nrows;
    int COLS = csr->ncols;

    int warps_per_block = threads_per_block / 32;
    int n_blocks = ceil((float)ROWS / warps_per_block);

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
        size_t buffer_size = sizeof(data_t)*warps_per_block*32;
        // size_t vec_size = sizeof(data_t)*COLS;
        kernel<<<n_blocks, threads_per_block, buffer_size>>>(
                vals, xs, ys, vec, ret, COLS, ROWS
                );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("kernel launch failed: %s\n", cudaGetErrorString(err));
        }
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
