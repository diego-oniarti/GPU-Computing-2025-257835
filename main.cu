#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "src/matrix.h"
#include "src/common.h"
#include "src/timing.h"

#define COLS 2000
#define ROWS 3000

#define PRERUNS 3
#define RUNS 10

data_t* multiply_naive(MAT_CSR *mat, data_t *vec);

// __global__
// void mult_per_row(data_t *vals, int *xs, int *ys, 
//         data_t *vec, data_t *ret,
//         int cols, int rows) {
//     int tid = blockIdx.x*blockDim.x + threadIdx.x;
// 
//     data_t acc = 0;
//     int n_elements = ys[tid+1] - ys[tid];
//     for (int i=0; i<)
// }

int main(void) {
    srand(time(NULL));

    data_t *mat = get_sparse_matrix(ROWS, COLS, 0.5);
    MAT_CSR csr;
    mat_to_CSR(&csr, mat, COLS, ROWS);
    data_t *ones = get_ones(COLS);

    data_t *prod_naive = multiply_naive(&csr, ones);

    // printf("CSR:\n");
    // print_CSR(&csr);
    // printf("CPU product:\n");
    // print_array(prod_naive, ROWS);

    free(mat);
    free(ones);
    free(prod_naive);
    destroy_CSR(&csr);
}

data_t* multiply_naive(MAT_CSR *mat, data_t *vec) {
    struct timeval temp_1={0,0}, temp_2={0,0};
    double times[RUNS];

    data_t *result;

    for (int r=-PRERUNS; r<RUNS; r++) {
        result = (data_t*)malloc(sizeof(data_t)*ROWS);
        gettimeofday(&temp_1, (struct timezone*)0);

        /// Actual multiplication
        for (int y=0; y<ROWS; y++) {
            data_t row_acc = 0;
            for (int i=mat->ys[y]; i<mat->ys[y+1]; i++) {
                int x = mat->xs[i];
                row_acc += mat->vals[i] * vec[x];
            }
            result[y] = row_acc;
        }
        ///

        gettimeofday(&temp_2, (struct timezone*)0);
        double time = ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1.tv_usec));
        printf("Elapsed time: %lf\n", time);
        if (r>=0) {
            times[r] = time;
        }

        // Don't free the memory the last time, so it can be returned
        if (r<RUNS-1) {
            free(result);
        }
    }

    print_timing(times, RUNS);
    return result;
}
