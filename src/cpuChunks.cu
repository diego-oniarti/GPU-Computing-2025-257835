#include "common.h"
#include "timing.h"
#include <sys/time.h>
#include <stdio.h>

#define SPAN 4

data_t* multiply_chunks(MAT_CSR *mat, data_t *vec) {
    struct timeval temp_1={0,0}, temp_2={0,0};
    double times[RUNS];

    int ROWS = mat->nrows;
    int COLS = mat->ncols;

    data_t *result;

    for (int r=-PRERUNS; r<RUNS; r++) {
        result = (data_t*)malloc(sizeof(data_t)*ROWS);

        // Initialize the counters
        int *row_counters = (int*)malloc(sizeof(int)*ROWS);
        for (int i=0; i<ROWS; i++) {
            row_counters[i] = mat->ys[i];
        }

        gettimeofday(&temp_1, (struct timezone*)0);

        /// Actual multiplication
        for (int i=0; i<ROWS; i++) {
            result[i]=0;
        }
        for (int i=0; i<COLS; i+= SPAN) {
            for (int r=0; r<ROWS; r++) {
                int line_end = mat->ys[r+1];
                int *rc = &row_counters[r];
                while (*rc<line_end && mat->xs[*rc]>=i && mat->xs[*rc]<i+ SPAN) {
                    result[r] += mat->vals[*rc] * vec[mat->xs[*rc]];
                    (*rc)++;
                }
            }
        }
        ///

        gettimeofday(&temp_2, (struct timezone*)0);
        double time = ((temp_2.tv_sec-temp_1.tv_sec)*1.e3+(temp_2.tv_usec-temp_1.tv_usec)/1.e3);
        if (DOPRINTSINGLE) printf("--- Elapsed time: %lf\n", time);
        if (r>=0) {
            times[r] = time;
        }

        // Don't free the memory the last time, so it can be returned
        if (r<RUNS-1) {
            free(result);
        }
        free(row_counters);
    }

    print_timing(times, RUNS, mat->nvals*2, mat);
    return result;
}
