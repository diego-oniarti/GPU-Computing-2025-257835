#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

/*
 * Generates a sparse matrix where each element has a
 * uniform probability `p` of being nonzero.
 * The values are in the range [0-10] and have 2 decimal points
 */
data_t* get_sparse_matrix(int rows, int cols, float p) {
    data_t *ret = (data_t*)malloc(sizeof(data_t) * cols * rows);
    for (int y=0; y<rows; y++) {
        for (int x=0; x<cols; x++) {
            ret[y*cols + x] = (rand()%100 < p*100) ? (data_t)(rand()%1000)/100. : 0;
            //printf("%f ", ret[y*rows + x]);
        }
    }

    return ret;
}

/*
 * Generates a vector of the given size filled with ones
 */
data_t* get_ones(int n) {
    data_t *ret = (data_t*)malloc(sizeof(data_t) * n);
    for (int i=0; i<n; i++) {
        ret[i] = 1;
    }

    return ret;
}

int count_non_zeros(data_t *mat, int rows, int cols) {
    int acc = 0;
    for (int y=0; y<rows; y++) {
        for (int x=0; x<cols; x++) {
            if (mat[y*cols+x]!=0) {
                acc++;
            }
        }
    }
    return acc;
}

/**
 * Populates a CSR matrix with the values from a matrix
 */
void mat_to_CSR(MAT_CSR *csr, data_t *mat, int cols, int rows) {
    int nvals  = count_non_zeros(mat, rows, cols);
    csr->nvals = nvals;
    csr->nrows = rows;
    csr->vals  = (data_t*)malloc(sizeof(data_t) * nvals);
    csr->xs    = (int*)malloc(sizeof(int) * nvals);
    csr->ys    = (int*)malloc(sizeof(int) * (rows+1));
    csr->ys[0]=0;
    int n=0;
    for (int y=0; y<rows; y++) {
        for (int x=0; x<cols && n<nvals; x++) {
            if (mat[y*cols+x]!=0) {
                csr->vals[n] = mat[y*cols+x];
                csr->xs[n] = x;
                n++;
            }
        }
        csr->ys[y+1] = n;
    }
}

void destroy_CSR(MAT_CSR *csr) {
    free(csr->vals);
    free(csr->xs);
    free(csr->ys);
}

void print_array(data_t *arr, int n) {
    for (int i=0; i<n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

void print_array_i(int *arr, int n) {
    for (int i=0; i<n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void print_CSR(MAT_CSR *csr) {
    printf("Vals:  |");
    print_array(csr->vals, csr->nvals);
    printf("Xs:    |");
    print_array_i(csr->xs, csr->nvals);
    printf("Ys:    |");
    print_array_i(csr->ys, csr->nrows+1);
}
