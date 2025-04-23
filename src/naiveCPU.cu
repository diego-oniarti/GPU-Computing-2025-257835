#include "common.h"
#include "timing.h"
#include <sys/time.h>
#include <stdio.h>

data_t* multiply_naive(MAT_CSR *mat, data_t *vec) {
    struct timeval temp_1={0,0}, temp_2={0,0};
    double times[RUNS];

    int ROWS = mat->nrows;

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
        if (DOPRINTSINGLE) printf("--- Elapsed time: %lf\n", time);
        if (r>=0) {
            times[r] = time;
        }

        // Don't free the memory the last time, so it can be returned
        if (r<RUNS-1) {
            free(result);
        }
    }

    print_timing(times, RUNS, mat->nvals*2);
    return result;
}

