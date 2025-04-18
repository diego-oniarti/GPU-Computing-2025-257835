#include "common.h"

__global__ void mult_per_row_kelner(data_t *vals, int *xs, int *ys, data_t *vec, data_t *ret, int cols, int rows);

data_t* mult_per_row(MAT_CSR *csr, data_t *ones);
