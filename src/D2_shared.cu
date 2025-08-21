#include "common.h"
#include "timing.h"
#include "matrix.h"
#include "warpRow.h"
#include <stdio.h>

#ifndef TILE
#define TILE 128
#endif

__global__ void D2_shared_kelner(
    const data_t * __restrict__ vals,
    const int    * __restrict__ xs,
    const int    * __restrict__ ys,
    const data_t * __restrict__ vec,
    data_t       * __restrict__ ret,
    int cols, int rows)
{
    // Warp id
    int wid  = threadIdx.x >> 5;
    // Thread's place in the warp
    int friend_id = threadIdx.x & 31;
    // Row: blockid * rows_per_block + warpid
    int row = blockIdx.x * (blockDim.x >> 5) + wid;

    if (row >= rows) return;

    extern __shared__ data_t smem[];
    // Warp's tile base
    data_t* s_vec = smem + (size_t)wid * TILE;

    int start = ys[row];
    int end = ys[row+1];

    data_t sum = 0;
    int cur = start;

    while (cur < end) {
        int tile_start = __ldg(&xs[cur]);

        // Allign with the tile
        int tileBase = (tile_start / TILE) * TILE;
        int tileEnd = tileBase + TILE;
        if (tileEnd > cols) tileEnd = cols;
        int tileW = tileEnd - tileBase;

        // stride 32 for coalesced access
        for (int t = friend_id; t < tileW; t += 32) {
            s_vec[t] = __ldg(&vec[tileBase + t]);
        }
        __syncwarp();

        // Iterate cooperatively through the row with stride 32
        // Start from cur and stop once you go out of the tile
        int j;
        for (j = cur+friend_id; j<end; j+=32) {
            int c = __ldg(&xs[j]);
            if (c >= tileEnd) break;
            int off = c - tileBase;
            sum += vals[j] * s_vec[off];
        }

        int next = j;
        for (int off = 16; off > 0; off >>= 1) {
            next = min(next, __shfl_down_sync(0xffffffff, next, off));
        }
        next = __shfl_sync(0xffffffff, next, 0);

        cur = next;
        __syncwarp();
    }

    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (friend_id == 0) ret[row] = sum;
}


data_t* D2_shared(MAT_CSR *csr, data_t *ones, int threads_per_block) {
    int ROWS = csr->nrows;
    int COLS = csr->ncols;

    int warps_per_block = threads_per_block / 32;
    int n_blocks = ceil((float)ROWS / warps_per_block);
    size_t shmem_bytes = (size_t)warps_per_block * TILE * sizeof(data_t);

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

    double times[RUNS];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int r = -PRERUNS; r < RUNS; r++) {
        cudaEventRecord(start);
        D2_shared_kelner<<<n_blocks, threads_per_block, shmem_bytes>>>(
                vals, xs, ys, vec, ret, COLS, ROWS);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaDeviceSynchronize();

        if (DOPRINTSINGLE) printf("--- Elapsed time: %lf\n", milliseconds);
        if (r >= 0) times[r] = milliseconds;
    }
    print_timing(times, RUNS, csr->nvals*2, csr);

    cudaFree(vals);
    cudaFree(vec);
    cudaFree(xs);
    cudaFree(ys);

    return ret;
}
