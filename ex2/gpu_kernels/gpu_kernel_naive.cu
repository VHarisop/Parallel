// -*- c++ -*-
/*
 *  gpu_kernel_naive.cu -- Naive Jacobi GPU kernel.
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Athena Elafrou
 */ 

#include <stdio.h>
#include <cuda.h>
#include "error.h"
#include "gpu_util.h"
#include "kernel.h"
#include "timer.h"

#define GPU_KERNEL_NAME(name)   do_jacobi_gpu ## name
/*
 *  Naive GPU kernel: 
 *  Every thread updates a single matrix entry directly on global memory 
 *  with a 1-1 mapping of threads to matrix elements.
 */ 
__global__ void GPU_KERNEL_NAME(_naive)(REAL *input, REAL *output, int N)
{
    // FILLME: the naive GPU kernel code
    int i = threadIdx.x;
    int j = threadIdx.y;
    int I = blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x + j * N + i;

    if ((I > N) && (I < N*N - 1 - N) && (I % N != 0) && (I % N != N - 1))
    {
        output[I] = 0.25f * (input[I - 1] + input[I + 1] + input[I - N] + input[I + N]);
    }
}

void MAKE_KERNEL_NAME(jacobi, _gpu, _naive)(kernel_data_t *data)
{
    int N = data->N;
    REAL **A = data->A;
    REAL **A_prev = data->A_prev;
    REAL *dev_A_prev = NULL, *dev_A = NULL;
    xtimer_t compute_timer, transfer_timer;

    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    allocate_data_on_gpu(&dev_A_prev, &dev_A, N);
    copy_data_from_cpu(dev_A, A, N);
    copy_data_from_cpu(dev_A_prev, A_prev, N);
    timer_stop(&transfer_timer);

    // FILLME: set up grid and thread block dimensions
    // dim3 block(32, 6);
    dim3 block(32, 6);
    dim3 grid(int((N - 0.5)/block.x) + 1, int((N - 0.5)/block.y) + 1);

    timer_clear(&compute_timer);
    timer_start(&compute_timer);

#ifdef _CONV
#   undef T
#   define T 100000000
    int converged = 0;
    for (int t = 0; t < T && !converged; t++) {
#else
    for (int t = 0; t < T; t++) {
#endif
        // FILLME: launch the GPU kernel(s) for Jacobi computation
        GPU_KERNEL_NAME(_naive)<<< grid, block >>>(dev_A_prev, dev_A, N);

#ifdef _CONV
        if (t % C == 0) {
            // FILLME BONUS: launch the GPU kernel for convergence test
        }
#endif

        REAL *tmp = dev_A_prev;
        dev_A_prev = dev_A;
        dev_A = tmp;
    }

    // Wait for last kernel to finish, so as to measure correctly the
    // computation and transfer times
    cudaThreadSynchronize();
    timer_stop(&compute_timer);
    double jacobi = timer_elapsed_time(&compute_timer);

    // Copy back results to host
    timer_start(&transfer_timer);
    copy_data_to_cpu(A_prev, dev_A_prev, N);
    timer_stop(&transfer_timer);
    printf("N: %d Gx: %d Gy: %d\n", N, N/block.x, N/block.y);
    printf("Transfer time:      %lf s\n", timer_elapsed_time(&transfer_timer));
    printf("Computation time:   %lf s\n", jacobi);
    // Performance is only correct when there is no convergece test 
    size_t size = N*N;
    printf("Jacobi performance: %lf Gflops/s\n", (T*size*4*1.e-9)/jacobi);
}
