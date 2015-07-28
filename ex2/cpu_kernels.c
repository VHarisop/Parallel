/*
 *  cpu_kernels.c -- Jacobi serial and OpenMP CPU kernel.
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Athena Elafrou
 */ 

#include <stdio.h>
#include <math.h>
#include "error.h"
#include "kernel.h"
#include "timer.h"

#include "omp.h"

void MAKE_KERNEL_NAME(jacobi, _cpu, _serial)(kernel_data_t *data)
{
    int N = data->N;
    REAL **A = data->A;
    REAL **A_prev = data->A_prev;

    for (int t = 0; t < T; t++) {
        for (int i = 1; i < (N-1); i++) {
            for (int j = 1; j < (N-1); j++) {
                A[i][j] = 0.25f * (A_prev[i-1][j] + A_prev[i+1][j] +
                                   A_prev[i][j-1] + A_prev[i][j+1]);
            }
        }

        // Swap pointers
        REAL **tmp = A_prev;
        A_prev = A;
        A = tmp;
    }
}

void MAKE_KERNEL_NAME(jacobi, _cpu, _omp)(kernel_data_t *data)
{
    int N = data->N;
    REAL **A = data->A;
    REAL **A_prev = data->A_prev;
    int i, j;

    REAL ** swap;

    // FILLME: copy your most efficient implementation in OpenMP
    #pragma omp parallel 
    {
        int t;
        for (t = 0; t < T; ++t)
        {
            #pragma omp for collapse(2) private(i,j)
            for (i = 1; i < N - 1; ++i) {
                for (j = 1; j < N - 1; ++j)
                    A[i][j] = 0.25f * (A_prev[i-1][j] + A_prev[i+1][j] + 
                                       A_prev[i][j-1] + A_prev[i][j+1]);
            }
            #pragma omp single nowait 
            {
                swap = A_prev; A_prev = A; A = swap;
            }
        }
    }
}
