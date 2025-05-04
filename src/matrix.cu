#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
        ret[i] = (data_t)(rand()%1000)/100.;
    }

    return ret;
}

data_t* get_random_vec(int n) {
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
    csr->ncols = cols;
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

bool check_equal(data_t *m1, data_t *m2, int n) {
    for (int i=0; i<n; i++) {
        if (m1[i] != m2[i]) return false;
    }
    return true;
}

void assert_correct(data_t *m1, data_t *m2, int n) {
    data_t maxErr = 0;
    int n_errors = 0;
    for (int i=0; i<n; i++) {
        if (m1[i] != m2[i]) {
            data_t error = abs(m2[i]-m1[i]);
            if (error>maxErr) maxErr=error;
            n_errors++;
        }
    }
    if (maxErr != 0) {
        fprintf(stderr, "!!! - Num errors: %d | Max error: %.17g\n", n_errors, maxErr);
        // print error as binary
        long long *errInt = (long long*)&maxErr;
        for (int i=63; i>=0; i--) {
            long long b = (*errInt) & ((long long)1<<i);
            if (b==0) {
                fprintf(stderr, "0");
            }else{
                fprintf(stderr, "1");
            }
        }
        fprintf(stderr, "\n");
        
        // exit(1);
    }
}

void read_mtx(MAT_CSR *mat, const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return;

    char *line = (char*)malloc(sizeof(char)*255);
    size_t len = 255;

    int rows, cols, nonzeros;
    while (getline(&line, &len, file) != -1) {
        if (len==0 || line[0]=='%') continue;
        sscanf(line, "%d %d %d", &rows, &cols, &nonzeros);
        break;
    }
    printf("rows %d\ncols %d\nnonzeros %d\n", rows, cols, nonzeros);

    fpos_t pos;  // Declare a position holder
    fgetpos(file, &pos);  // Save current position

    mat->nvals = nonzeros;
    mat->ncols = cols;
    mat->nrows = rows;
    mat->vals  = (data_t*)malloc(sizeof(data_t) * nonzeros);
    mat->xs    = (int*)malloc(sizeof(int) * nonzeros);
    mat->ys    = (int*)calloc(rows+1, sizeof(int));
    int *row_counter = (int*)calloc(rows, sizeof(int));

    // First cycle to count the number of elements on each row
    for (int i=0; i<nonzeros; i++) {
        int y=0, x=0;
        data_t val=0;
        getline(&line, &len, file);
        sscanf(line, "%d %d %lf", &y, &x, &val);
        y--;
        x--;
        mat->ys[y+1]++;
    }
    // Incremental sum of the row pointer
    for (int i=1; i<=rows; i++) {
        mat->ys[i] += mat->ys[i-1];
    }

    // Reset the position in the file to read the lines again
    fsetpos(file, &pos);
    for (int i=0; i<nonzeros; i++) {
        int y=0, x=0;
        data_t val=0;
        getline(&line, &len, file);
        sscanf(line, "%d %d %lf", &y, &x, &val);
        y--;
        x--;

        //int p = mat->ys[y]+(row_counter[y]++);
        int base = mat->ys[y];
        int holding_x = x;
        data_t holding_val = val;
        for (int i=0; i<row_counter[y]; i++) {
            if ( holding_x < mat->xs[base+i] ) {
                int tmp_x = mat->xs[base+i];
                mat->xs[base+i] = holding_x;
                holding_x = tmp_x;

                data_t tmp_val = mat->vals[base+i];
                mat->vals[base+i] = holding_val;
                holding_val = tmp_val;
            }
        }
        mat->xs[base+row_counter[y]] = holding_x;
        mat->vals[base+row_counter[y]] = holding_val;
        row_counter[y] += 1;
    }

    fclose(file);
    free(line);
    free(row_counter);
}
