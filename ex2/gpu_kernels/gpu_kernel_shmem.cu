// -*- c++ -*-
/*
 *  gpu_kernel_shmem.cu -- Simple and improved Jacobi GPU kernels that
 *                         use shared memory.
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

// elements per thread: 4
#define TSZ (2)
#define BSZX (96)
#define BSZY (2)
#define BSZ (16)

/*
 *  Simple GPU kernel that uses shared memory:
 *  Every thread copies its corresponding element to shared memory.
 */
__global__ void GPU_KERNEL_NAME(_shmem_improved)(REAL *in, REAL *out, int N)
{
    // FILLME: the simple GPU kernel code that uses shared memory
    // multi-element computation per thread

    int i = threadIdx.x;
    int j = threadIdx.y; 

    // accurate position based on thread "size" 
    // threads access elements on column
    int I = blockIdx.y * blockDim.y * N * TSZ + blockIdx.x * blockDim.x + j * TSZ * N + i;

    // shared array of elements
    //__shared__ float u_prev_sh[32][32 * TSZ];
    __shared__ float u_prev_sh[BSZY * TSZ][BSZX];

    int ii, lcj, ei;

    // iterate on column index
    for (ii = 0; ii < TSZ; ++ii)
    {
        ei = I + ii * N;
        // warp
        if (ei < (N * N))
        {
            u_prev_sh[j * TSZ + ii][i]  = in[ei];
        }
    }

    __syncthreads();
    bool bound_check;
    bool block_check = ((i > 0) && (i < blockDim.y - 1) && (j > 0) && (j < blockDim.y - 1));
    ei = I;

    for (ii = 0; ii < TSZ; ++ii) 
    {
        ei += ii * N;
        lcj = j * TSZ + ii;
        bound_check = ((ei > N) && (ei < N * N - 1 - N) && (ei % N != 0) && (ei % N != N - 1));
        if (block_check && bound_check) 
        {
            out[ei] = 0.25f * (u_prev_sh[lcj-1][i] + u_prev_sh[lcj+1][i] + u_prev_sh[lcj][i-1] + u_prev_sh[lcj][i+1]);
        }
        else if (bound_check) 
        {
            out[ei] = 0.25f * (in[ei - 1] + in[ei + 1] + in[ei - N] + in[ei + N]);
        }
    }
}

/*
 *  Improved GPU kernel that uses shared memory.
 */
__global__ void GPU_KERNEL_NAME(_shmem)(REAL *in, REAL *out, int N)
{
    // FILLME: improved GPU kernel code that uses shared memory
    int i = threadIdx.x;
    int j = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int I = (BSZ - 2) * bx + i;
    int J = (BSZ - 2) * by + j;

    int Index = I + J*N;

    if (I >= N || J >= N)
        return;

    __shared__ float u_prev_sh[BSZ][BSZ];

    u_prev_sh[i][j] = in[Index];

    __syncthreads();

    bool bound_check = ((I > 0) && (I < (N - 1)) && (J > 0) && (J < (N-1)));
    bool block_check = ((i > 0) && (i < (BSZ - 1)) && (j > 0) && (j < (BSZ - 1)));

    if (block_check && bound_check) 
    {
        out[Index] = 0.25f * (u_prev_sh[i-1][j] + u_prev_sh[i+1][j] + u_prev_sh[i][j-1] + u_prev_sh[i][j+1]);
    }
}

void MAKE_KERNEL_NAME(jacobi, _gpu, _shmem)(kernel_data_t *data)
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


    dim3 block(BSZ, BSZ);
    dim3 grid(int((N - 0.5)/(block.x - 2)) + 1, int((N - 0.5)/(block.y - 2)) + 1);

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
        GPU_KERNEL_NAME(_shmem)<<< grid, block >>>(dev_A_prev, dev_A, N);

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

void MAKE_KERNEL_NAME(jacobi, _gpu, _shmem_improved)(kernel_data_t *data)
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
    dim3 block(BSZX, BSZY);
    dim3 grid(int((N - 0.5)/block.x) + 1, int((N - 0.5)/(block.y * TSZ)) + 1);
    //
//    dim3 block(BSZX, BSZY); //3 x THREADS x BSZy size
//    dim3 grid(int((N - 0.5) / (block.x * TSZ)) + 1, int((N-0.5)/block.y) + 1); // N threads on column

    printf("N: %d, Block %d, Grid: %d\n", N, block.x * TSZ, int(N / (block.x * TSZ)));

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
        GPU_KERNEL_NAME(_shmem_improved)<<< grid, block >>>(dev_A_prev, dev_A, N);

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
