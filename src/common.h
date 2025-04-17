#ifndef data_t
#define data_t float

typedef struct {
    data_t *vals;
    int *xs;
    int *ys;
    int nvals;
    int nrows;
} MAT_CSR;
#endif
