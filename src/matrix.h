#include "common.h"

/*
 * Generates a sparse matrix where each element has a
 * uniform probability `p` of being nonzero.
 * The values are in the range [0-10] and have 2 decimal points
 */
data_t* get_sparse_matrix(int rows, int cols, float p);

void read_mtx(MAT_CSR *mat, const char *path);

/*
 * Generates a vector of the given size filled with ones
 */
data_t* get_ones(int n);

int count_non_zeros(data_t *mat, int rows, int cols);

/**
 * Populates a CSR matrix with the values from a matrix
 */
void mat_to_CSR(MAT_CSR *csr, data_t *mat, int cols, int rows);

void destroy_CSR(MAT_CSR *csr);

void print_array(data_t *arr, int n);
void print_array_i(int *arr, int n);
void print_CSR(MAT_CSR *csr);

bool check_equal(data_t *m1, data_t *m2, int n);
void assert_correct(data_t *m1, data_t *m2, int n);
