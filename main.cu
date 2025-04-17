#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "src/common.h"

#define COLS 2
#define ROWS 3

int main(void) {
    srand(time(NULL));

    data_t *mat = get_sparse_matrix(ROWS, COLS, 0.5);
    for (int y=0; y<ROWS; y++) {
        for (int x=0; x<COLS; x++) {
            printf("%f ", mat[y*COLS + x]);
        }
        printf("\n");
    }

    MAT_COO coo;
    mat_to_COO(&coo, mat, COLS, ROWS);

    MAT_CSR csr;
    mat_to_CSR(&csr, mat, COLS, ROWS);

    free(mat);
}
