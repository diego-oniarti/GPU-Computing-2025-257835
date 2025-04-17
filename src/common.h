#define data_t float

typedef struct {
    data_t *vals;
    int *xs;
    int *ys;
    int nvals;
} MAT_COO;

typedef struct {
    data_t *vals;
    int *xs;
    int *ys;
    int nvals;
} MAT_CSR;

/*
 * Generates a sparse matrix where each element has a
 * uniform probability `p` of being nonzero.
 * The values are in the range [0-10] and have 2 decimal points
 */
data_t* get_sparse_matrix(int rows, int cols, float p);

/*
 * Generates a vector of the given size filled with ones
 */
data_t* get_ones(int n);

int count_non_zeros(data_t *mat, int rows, int cols);

/**
 * Populates a COO matrix with the values from a matrix
 */
void mat_to_COO(MAT_COO *coo, data_t *mat, int cols, int rows);

/**
 * Populates a CSR matrix with the values from a matrix
 */
void mat_to_CSR(MAT_CSR *csr, data_t *mat, int cols, int rows);

void destroy_CSR(MAT_CSR *csr);
void destroy_COO(MAT_COO *coo);
