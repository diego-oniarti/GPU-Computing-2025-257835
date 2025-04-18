#ifndef data_t
#define data_t float

typedef struct {
    data_t *vals;
    int *xs;
    int *ys;
    int nvals;
    int nrows;
    int ncols;
} MAT_CSR;

#define PRERUNS 3
#define RUNS 10
#define COLS 2000
#define ROWS 3000

#define DOPRINT false

#endif
