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
#define COLS 200000
#define ROWS 300000

// print the results
#define DOPRINT false
// print the individual times for each run
#define DOPRINTSINGLE false

#endif
