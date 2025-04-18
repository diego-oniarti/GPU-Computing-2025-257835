#include <stdio.h>
#include <stdlib.h>

#include "src/matrix.h"
#include "src/common.h"

#include "src/naiveCPU.h"
#include "src/threadPerRow.h"

int main(void) {
    srand(time(NULL));

    data_t *mat = get_sparse_matrix(ROWS, COLS, 0.5);
    MAT_CSR csr;
    mat_to_CSR(&csr, mat, COLS, ROWS);
    data_t *ones = get_ones(COLS);

    if (DOPRINT) {
        printf("CSR:\n");
        print_CSR(&csr);
    }

    // Naive CPU implementation
    {
        printf("CPU product:\n");
        data_t *prod_naive = multiply_naive(&csr, ones);
        if (DOPRINT) print_array(prod_naive, ROWS);
        free(prod_naive);
    }

    // Simple GPU implementation
    // The elements on each row are handled by their own thread
    {
        printf("GPU. One thread per row:\n");
        data_t* prod = mult_per_row(&csr, ones);
        if (DOPRINT) print_array(prod, ROWS);
        cudaFree(prod);
    }


    free(mat);
    free(ones);
    destroy_CSR(&csr);
}
