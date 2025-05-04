#include <stdio.h>
#include <stdlib.h>

#include "src/matrix.h"
#include "src/common.h"

#include "src/naiveCPU.h"
#include "src/threadPerRow.h"
#include "src/warpRow.h"
#include "src/warpRowShared.h"

int main(int argc, char **argv) {
    srand(time(NULL));

    MAT_CSR csr;
    if (argc != 2) {
        printf("No matrix specified. Generating one\n");
        data_t *mat = get_sparse_matrix(30000, 20000, 0.01);
        mat_to_CSR(&csr, mat, 30000, 20000);
        free(mat);
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

    // Simple GPU implementation
    // The elements on each row are handled by their own thread
    {
        printf("-------------- GPU. One thread per row: -------------\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("Threads per block: %d\n", n_threads);
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
    {
        printf("-------------- GPU. One warp per row: ---------------\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("Threads per block: %d\n", n_threads);
            data_t* prod = mult_warp_row(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            // Misterious rounding errors?
            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    // Moved the buffer for the reduction to the shared memory
    {
        printf("----- GPU. One warp per row plus shared memory: -----\n");
        for (int n_threads = 32; n_threads <= prop.maxThreadsPerBlock; n_threads<<=1) {
            printf("Threads per block: %d\n", n_threads);
            data_t* prod = mult_warp_row_shared(&csr, vector, n_threads);
            if (DOPRINT) print_array(prod, ROWS);

            assert_correct(prod_naive, prod, ROWS);
            cudaFree(prod);
        }
    }

    free(prod_naive);
    free(vector);
    destroy_CSR(&csr);
}
