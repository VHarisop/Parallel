// -*- c++ -*-
/*
 *  gpu_kernel_time_tiled_shmem.cu -- Time-tiled Jacobi GPU kernel.
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

#define GPU_KERNEL_NAME(name)   do_jacobi_gpu_time_tiled ## name

__global__ void GPU_KERNEL_NAME(__time_tiled_shmem)(REAL *in, REAL *out, int N)
{
    // FILLME: the time-tiled GPU kernel code
    int i = threadIdx.x;
    int j = threadIdx.y;

    // block size (symmetrical)
    int BSZ = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int I_0 = by * N * BSZ + bx * BSZ;
    int Index = by * BSZ * N + bx * BSZ + (j + 1) * N + i + 1;

    int G_I = bx * BSZ + i + 1; // global I
    int G_J = by * BSZ + j + 1; // global J

    __shared__ float u_prev_sh[34][34];

    int ii = j * BSZ + i;   // thread indexing
    int I = ii % (BSZ + 2); // x-direction index
    int J = ii / (BSZ + 2); // y-direction index

    int I_G = I_0 + J * N + I;   // total index

    u_prev_sh[I][J] = in[I_G];

    int ii2 = BSZ * BSZ + j * BSZ + i;
    int I2 = ii2 % (BSZ + 2);
    int J2 = ii2 / (BSZ + 2);

    int I_G2 = I_0 + J2 * N + I2;

    if ((I2 < (BSZ + 2)) && (J2 < (BSZ + 2)) && (ii2 < N * N))
        u_prev_sh[I2][J2] = in[I_G2];

    __syncthreads();

    if ((G_J >= N - 1) || (G_I >= N - 1)) 
        return;

    out[Index] = 0.25f * (u_prev_sh[i+2][j+1] + u_prev_sh[i][j+1] + u_prev_sh[i+1][j+2] + u_prev_sh[i+1][j]);

}

void MAKE_KERNEL_NAME(jacobi, _gpu, _time_tiled_shmem)(kernel_data_t *data)
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

    dim3 block(32, 32);
    dim3 grid(int((N - 2 - 0.5)/block.x) + 1, int((N - 2 - 0.5)/block.y) + 1);

    timer_clear(&compute_timer);
    timer_start(&compute_timer);

#ifdef _CONV
#   undef T
#   define T 100000000
    int converged = 0;
    for (int t = 0; t < T && !converged; t+=TIME_STEP) {
#else
    for (int t = 0; t < T; t+=TIME_STEP) {
#endif
        // FILLME: launch the GPU kernel(s) for Jacobi computation
        GPU_KERNEL_NAME(__time_tiled_shmem)<<< grid, block >>>(dev_A_prev, dev_A, N); 

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
    printf("Transfer time:      %lf s\n", timer_elapsed_time(&transfer_timer));
    printf("Computation time:   %lf s\n", jacobi);
    // Performance is only correct when there is no convergece test 
    size_t size = N*N;
    printf("Jacobi performance: %lf Gflops/s\n", (T*size*4*1.e-9)/jacobi);
}
