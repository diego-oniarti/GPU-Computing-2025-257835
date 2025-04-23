#ifndef data_t
#define data_t double

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

// print the results
#define DOPRINT false
// print the individual times for each run
#define DOPRINTSINGLE false

#endif
