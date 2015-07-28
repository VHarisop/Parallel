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
#include "alloc.h"

#define GPU_KERNEL_NAME(name)   do_jacobi_gpu_time_tiled ## name

// optimal: BX = 26, BY = 6, T_STEP = 3, TSZ = 3
// optimal: BX = 64, BY = 8, T_STEP = TSZ = 4

#define BLOCKSZ (24)
#define BSZX (88)
#define BSZY (6)
#define TSZ  (4)
#define TIME_STEP (4)

__global__ void GPU_KERNEL_NAME(__time_tiled_shmem)(REAL *in, REAL *out, int N)
{
    // FILLME: the time-tiled GPU kernel code
    int i = threadIdx.x;
    int j = threadIdx.y;
    bool x_check, y_check, ind_check;
    int q, t, ei, ej;
    
    // fast range checking
    int BY = BSZY * TSZ + 2 * TIME_STEP;
    int BX = BSZX + 2 * TIME_STEP;

    // BSZY * TSZ + 2 * T + 2 -> 2 additional places for "useless" elements
    __shared__ float u[BSZY * TSZ + 2 * TIME_STEP][BSZX + 2 * TIME_STEP];

    // array of index checks 
    bool ys[TSZ];
    bool is[TSZ];
    bool yind[TSZ];

    // local array of indices to update
    float vals[TSZ];

    // Index -> top left corner of block array indices
    int Index = (BSZY * TSZ) * blockIdx.y * N + blockIdx.x * BSZX + (j * TSZ) * N + i - (N + 1) * TIME_STEP;

    #pragma unroll
    for (q = 0; q < TSZ; ++q)
    {
        ej = j * TSZ + q;
        ei = Index + q * N;
        // update y-index checks to use later
        ys[q] = (ej > 0) && (ej < (BY - 1));

        yind[q] = (ej >= TIME_STEP) && (ej < (BY - TIME_STEP));

        // update general index checks to use later
        is[q] = (ei > N) && (ei < (N * N - N - 1)) && ((ei % N) != 0) && ((ei % N) != (N - 1));

        if ((ei >= 0) && (ei < (N * N)))
        {
            // update ej 
            u[ej][i] = in[ei];
        }
    }

    // wait for all elements to be loaded
    __syncthreads();

    x_check = (i > 0) && (i < (BX - 1));

    if (x_check)
    {
        // update elements here
        
        for (t = 1; t < TIME_STEP; ++t)
        {
            // T - 1 steps of updates

            #pragma unroll
            for (q = 0; q < TSZ; ++q)
            {
                ej = j * TSZ + q;
                // should we update this index;
                if (ys[q] && is[q])
                {
                    vals[q] = 0.25f * (u[ej][i-1] + u[ej][i+1] + u[ej-1][i] + u[ej+1][i]);
                }
                
            }
            __syncthreads();

            #pragma unroll
            for (q = 0; q < TSZ; ++q)
            {
                ej = j * TSZ + q;
                if (ys[q] && is[q])
                {
                    u[ej][i] = vals[q];
                }
            }
            __syncthreads();
        }
    }

    x_check = (i >= TIME_STEP) && (i < (BX - TIME_STEP));

    if (x_check)
    {
        for (q = 0; q < TSZ; ++q)
        {
            ej = j * TSZ + q;
            ei = Index + q * N;

            if (yind[q] && is[q])
            {
                out[ei] = 0.25f * (u[ej][i-1] + u[ej][i+1] + u[ej-1][i] + u[ej+1][i]);
            }
        }
    }
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


    // blocks: BX = bx + 2 * T, BY = by + (2 * T / ThreadLoad) -> load balance
    dim3 block(BSZX + 2 * TIME_STEP, BSZY + int((2 * TIME_STEP - 0.5) / TSZ) + 1);
    dim3 grid(int((N - 0.5)/BSZX) + 1, int((N - 0.5)/(BSZY * TSZ)) + 1);
    
    printf("N: %d, Grid: %d\n", N, int((N - 0.5)/BLOCKSZ) + 1);

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
