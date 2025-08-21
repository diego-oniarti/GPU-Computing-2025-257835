#include <stdio.h>
#include <stdlib.h>

#include "src/matrix.h"
#include "src/common.h"

#include "src/naiveCPU.h"
#include "src/threadPerRow.h"
#include "src/warpRow.h"
#include "src/warpRowShared.h"
#include "src/cpuChunks.h"
#include "src/D2_shuffle.h"
#include "src/D2_shared.h"
#include "src/D2_final.h"

#include <cusparse.h>
#include "src/timing.h"

int main(int argc, char **argv) {
    printf("argc: %d\n", argc);
    fflush(stdout);

    srand(time(NULL));

    MAT_CSR csr;
    if (argc != 2) {
        printf("No matrix specified. Generating one\n");
        generate_csr(&csr, 30000, 20000, 0.001);
    } else {
        read_mtx(&csr, argv[1]);
    }

    int ROWS = csr.nrows;
    int COLS = csr.ncols;

    data_t *vector = get_random_vec(COLS);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (DOPRINT) {
        printf("CSR:\n");
        print_CSR(&csr);
    }

    {
        // Calculate theoretical bandwidth
        float bandwidth = 
            (prop.memoryClockRate * 1000.0f * 2 *  // Convert kHz to Hz and double for DDR
             (prop.memoryBusWidth / 8.0f)) /        // Convert bits to bytes
            1e9;                                    // Convert to GB/s

        printf("Device: %s\n", prop.name);
        printf("Memory Clock Rate: %.1f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("Theoretical Memory Bandwidth: %.2f GB/s\n\n", bandwidth);
    }

    // Naive CPU implementation
    printf("------------------- CPU product: -------------------\n");
    data_t *prod_naive = multiply_naive(&csr, vector);
    if (DOPRINT) print_array(prod_naive, ROWS);

    // CPU chunked
    if (false) {
        printf("------------------- CPU + chunks: -------------------\n");
        data_t *prod = multiply_chunks(&csr, vector);
        if (DOPRINT) print_array(prod, ROWS);

        assert_correct(prod_naive, prod, ROWS);
        free(prod);
    }

    // Simple GPU implementation
    // The elements on each row are handled by their own thread
    if (false) {
        printf("-------------- GPU. One thread per row: -------------\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("\nThreads per block: %d\n", n_threads);
            data_t* prod = mult_per_row(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    // Improved GPU implementation
    // The elements on each row are handled by their own warp
    // Rows act more independently, not forcing locked step for different number
    // of elements
    if (true) {
        printf("-------------- GPU. One warp per row: ---------------\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("\nThreads per block: %d\n", n_threads);
            data_t* prod = mult_warp_row(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            // Misterious rounding errors?
            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    // Moved the buffer for the reduction to the shared memory
    if (false) {
        printf("----- GPU. One warp per row plus shared memory: -----\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("\nThreads per block: %d\n", n_threads);
            data_t* prod = mult_warp_row_improved(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    /// DELIVERABLE 2

    // Use bit shuffling to reduce instead of a buffer
    {
        printf("------------- GPU. shuffle + intrinsic: -------------\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("\nThreads per block: %d\n", n_threads);
            data_t* prod = D2_shuffle(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    // Use shuffle + shared memory for the Ys
    {
        printf("---------------- GPU. Shared memory: ----------------\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("\nThreads per block: %d\n", n_threads);
            data_t* prod = D2_shared(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    // CUSPARSE
    {
        printf("------------------- cuSPARSE: -------------------\n");

        cusparseHandle_t handle;
        cusparseCreate(&handle);

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void* dBuffer = NULL;
        size_t bufferSize = 0;

        int rows = csr.nrows;
        int cols = csr.ncols;
        int nnz  = csr.nvals;

        int *d_csrRowPtr, *d_csrColInd;
        data_t *d_csrVal, *d_x, *d_y;

        cudaMalloc((void**)&d_csrRowPtr, (rows + 1) * sizeof(int));
        cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
        cudaMalloc((void**)&d_csrVal,    nnz * sizeof(data_t));
        cudaMalloc((void**)&d_x,         cols * sizeof(data_t));
        cudaMalloc((void**)&d_y,         rows * sizeof(data_t));

        cudaMemcpy(d_csrRowPtr, csr.ys, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColInd, csr.xs, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrVal,    csr.vals,     nnz * sizeof(data_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, vector, cols * sizeof(data_t), cudaMemcpyHostToDevice);

        cusparseCreateCsr(&matA, rows, cols, nnz,
                d_csrRowPtr, d_csrColInd, d_csrVal,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F);

        cusparseCreateDnVec(&vecX, cols, d_x,
                CUDA_R_32F);
        cusparseCreateDnVec(&vecY, rows, d_y,
                CUDA_R_32F);

        data_t alpha = 1.0, beta = 0.0;
        cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

        cudaMalloc(&dBuffer, bufferSize);

        double times[RUNS];
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int r = -PRERUNS; r < RUNS; r++) {
            cudaEventRecord(start);
            cusparseSpMV(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY,
                    CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaDeviceSynchronize();

            if (DOPRINTSINGLE) printf("--- Elapsed time: %lf ms\n", milliseconds);
            if (r >= 0) {
                times[r] = milliseconds;
            }
        }

        print_timing(times, RUNS, csr.nvals * 2, &csr);

        data_t* prod_cusparse = (data_t*)malloc(rows * sizeof(data_t));
        cudaMemcpy(prod_cusparse, d_y, rows * sizeof(data_t), cudaMemcpyDeviceToHost);
        assert_correct(prod_naive, prod_cusparse, rows);
        if (DOPRINT) print_array(prod_cusparse, rows);

        free(prod_cusparse);
        cudaFree(d_csrRowPtr);
        cudaFree(d_csrColInd);
        cudaFree(d_csrVal);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(dBuffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);
    }

    free(prod_naive);
    free(vector);
    destroy_CSR(&csr);
}
