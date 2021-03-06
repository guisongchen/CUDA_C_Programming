#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This code implements the interleaved and neighbor-paired approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used. A
 * variety of optimizations on parallel reduction aimed at reducing divergence
 * are also demonstrated, such as unrolling.
 */

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with less divergence
// basic idea: let tid 0,1,2,3 act as tid 0,2,4,6 in reduceNeighbored
// unlike reduceNeighbored, even tid like 0,2,4 working while tid 1,3,5 at rest
// in reduceNeighboredLess, all the scheduled threads have work to do
__global__ void reduceNeighboredLess (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2*stride*tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Unrolling loop Implementation with less divergence
__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    // unrolling neighbore
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // unrolling current and neighbore block(+blockDim.x)
    // at g_idata !!!
    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx+blockDim.x];
    }
    __syncthreads();

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // in-place reduction in global memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Unrolling loop Implementation with less divergence
__global__ void reduceUnrolling4 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    // unrolling neighbore
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling current and neighbore block(+blockDim.x)
    // at g_idata !!!
    if (idx + blockDim.x*3 < n) {
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+blockDim.x*2];
        int d = g_idata[idx+blockDim.x*3];
        g_idata[idx] = a+b+c+d;
    }
    __syncthreads();

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // in-place reduction in global memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Unrolling loop Implementation with less divergence
// different from reduceUnrolling4:
// 1. stop unroll warps when stride < 32, a.k.a. stride=32, 16, 8, 4, 2, 1 
//    because left threads are not enough for a warp
//    if keep unroll, the left threads must be padded into a warp(32 threads)
//    so some threads in the last warp will be halting, warp divergence happens
// 2. instead of putting the left threads in a warp, unroll them explicitly
//    usin stride 32, 16, 8, 4, 2, 1
__global__ void reduceUnrollWarps4 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    // unrolling neighbore
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling current and neighbore block(+blockDim.x)
    // at g_idata !!!
    if (idx + blockDim.x*3 < n) {
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+blockDim.x*2];
        int d = g_idata[idx+blockDim.x*3];
        g_idata[idx] = a+b+c+d;
    }
    __syncthreads();

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // in-place reduction in global memory
    // notice: stride > 32 !!!
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1)
    {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps4 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    // unrolling neighbore
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling current and neighbore block(+blockDim.x)
    // at g_idata !!!
    if (idx + blockDim.x*3 < n) {
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+blockDim.x*2];
        int d = g_idata[idx+blockDim.x*3];
        g_idata[idx] = a+b+c+d;
    }
    __syncthreads();

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // in-place reduction in global memory
    // unroll explicitly when stride > 32, start from 1024/2
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid+512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid+256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid+128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid+64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum_brutle = 0;
    for (int i = 0; i < size; i++) {
        cpu_sum_brutle += h_idata[i];
    }
    iElaps = seconds() - iStart;
    printf("cpu brutle      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum_brutle);

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);


    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrollWarps4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp4 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 7: reduceCompleteUnrollWarsp4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll4  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
