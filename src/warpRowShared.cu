#include "common.h"
#include "timing.h"
#include "matrix.h"
#include "warpRow.h"
#include <stdio.h>

__global__
void mult_warp_row_improved_kelner(
    const data_t * __restrict__ vals,
    const int * __restrict__ xs,
    const int * __restrict__ ys,
    const data_t * __restrict__ vec,
    data_t * __restrict__ ret,
    int cols, int rows
) {
    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int friend_id = threadIdx.x % 32;
    int row = blockIdx.x * warps_per_block + warp_id;

    extern __shared__ int s_ys[];
    int base_row = blockIdx.x * warps_per_block;

    if (friend_id <= warps_per_block) {
        int global_row = base_row + friend_id;
        int safe_idx = (global_row <= rows) ? global_row : rows;
        s_ys[friend_id] = ys[safe_idx];
    }

    __syncthreads();

    if (row >= rows) return;

    int start = s_ys[warp_id];
    int end = s_ys[warp_id + 1];

    data_t sum = 0;
    for (int i = start + friend_id; i < end; i += 32) {
        sum += vals[i] * __ldg(&vec[xs[i]]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (friend_id == 0) {
        ret[row] = sum;
    }
}

data_t* mult_warp_row_improved(MAT_CSR *csr, data_t *ones, int threads_per_block) {
    int ROWS = csr->nrows;
    int COLS = csr->ncols;

    int warps_per_block = threads_per_block / 32;
    int n_blocks = (ROWS + warps_per_block - 1) / warps_per_block;

    data_t *vals, *vec, *ret;
    int *xs, *ys;

    cudaMalloc(&vals, sizeof(data_t) * csr->nvals);
    cudaMemcpy(vals, csr->vals, sizeof(data_t) * csr->nvals, cudaMemcpyHostToDevice);

    cudaMalloc(&vec, sizeof(data_t) * COLS);
    cudaMemcpy(vec, ones, sizeof(data_t) * COLS, cudaMemcpyHostToDevice);

    cudaMallocManaged(&ret, sizeof(data_t) * ROWS);

    cudaMalloc(&xs, sizeof(int) * csr->nvals);
    cudaMemcpy(xs, csr->xs, sizeof(int) * csr->nvals, cudaMemcpyHostToDevice);

    cudaMalloc(&ys, sizeof(int) * (ROWS + 1));
    cudaMemcpy(ys, csr->ys, sizeof(int) * (ROWS + 1), cudaMemcpyHostToDevice);

    double times[RUNS];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int r = -PRERUNS; r < RUNS; r++) {
        cudaEventRecord(start);
        size_t shared_mem_size = (warps_per_block + 1) * sizeof(int);

        mult_warp_row_improved_kelner<<<n_blocks, threads_per_block, shared_mem_size>>>(
            vals, xs, ys, vec, ret, COLS, ROWS
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaDeviceSynchronize();

        if (DOPRINTSINGLE) printf("--- Elapsed time: %lf\n", milliseconds);
        if (r >= 0) {
            times[r] = milliseconds;
        }
    }

    print_timing(times, RUNS, csr->nvals * 2, csr);

    cudaFree(vals);
    cudaFree(vec);
    cudaFree(xs);
    cudaFree(ys);

    return ret;
}

