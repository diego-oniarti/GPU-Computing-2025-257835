#include "stdio.h"
#include "timing.h"

double mean(double *times, int n) {
    double acc = 0;
    for (int i=0; i<n; i++) {
        acc += times[i];
    }
    return acc/n;
}

double geom_mean(double *times, int n) {
    double acc = 1.0;
    for (int i=0; i<n; i++) {
        acc *= times[i];
    }
    return pow(acc, 1.0/n);
}

double derivation(double *times, int n) {
    double acc = 0;
    double m = mean(times, n);
    for (int i=0; i<n; i++) {
        acc += pow(times[i]-m, 2);
    }
    return sqrt(acc/(n-1));
}


void print_timing(double *times, int n) {
    // printf("               Mean: %.3f\n     Geometric mean: %.3f\nStandard derivation: %.3f\n",
    //         mean(times, n), geom_mean(times, n), derivation(times, n));
    printf("               Mean: %.3f ms\nStandard derivation: %.3f ms\n",
            mean(times, n), derivation(times, n));
}
