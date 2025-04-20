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


void print_timing(double *times, int n, int FLO) {
    double  avg = mean(times, n);
    printf("               Mean: %.3f ms\n", avg);
    printf("Standard derivation: %.3f ms\n", derivation(times, n));
    printf("              FLOPs: %.3f\n", (double)FLO/avg);
}
