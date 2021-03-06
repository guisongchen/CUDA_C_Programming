#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *in, const int size) {
    for (int i = 0; i < size; i++) {
        in[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for( int iy = 0; iy < ny; ++iy)
    {
        for( int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyRow(float *out, float *in, const int nx, const int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < nx && iy < ny) {
        out[ix + iy*nx] = in[ix + iy*nx];
    }
}

__global__ void copyCol(float *out, float *in, const int nx, const int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < nx && iy < ny) {
        out[iy + ix*ny] = in[iy + ix*ny];
    }
}

// read in rows and write in columns
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < nx && iy < ny) {
        out[iy + ix*ny] = in[ix + iy*nx];
    }
}

// read in columns and write in rows
__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < nx && iy < ny) {
        out[ix + iy*nx] = in[iy + ix*ny];
    }
}

// read in rows and write in columns, unroll blockIdx.x
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    int idx_out = iy + ix*ny;
    int idx_in = ix + iy*nx;

    if (ix + blockDim.x*3 < nx && iy < ny) {
        out[idx_out] = in[idx_in];
        out[idx_out +   blockDim.x*ny] = in[idx_in + blockDim.x];
        out[idx_out + 2*blockDim.x*ny] = in[idx_in + 2*blockDim.x];
        out[idx_out + 3*blockDim.x*ny] = in[idx_in + 3*blockDim.x];
    }
}

// read in columns and write in rows, unroll blockIdx.x
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    int idx_out = ix + iy*nx;
    int idx_in = iy + ix*ny;

    if (ix + blockDim.x*3 < nx && iy < ny) {
        out[idx_out] = in[idx_in];
        out[idx_out +   blockDim.x] = in[idx_in +   blockDim.x*ny];
        out[idx_out + 2*blockDim.x] = in[idx_in + 2*blockDim.x*ny];
        out[idx_out + 3*blockDim.x] = in[idx_in + 3*blockDim.x*ny];
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("device %d: %s ", dev, prop.name);
    cudaSetDevice(dev);

    // set up array size
    int nx = 1<<11;
    int ny = 1<<11;

    // kernel config
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc>1) iKernel = atoi(argv[1]);
    if (argc>2) blockx = atoi(argv[2]);
    if (argc>3) blocky = atoi(argv[3]);
    if (argc>4) nx = atoi(argv[4]);
    if (argc>5) ny = atoi(argv[5]);
    printf(" with matrix nx %d ny %d with kernel %d\n", nx,ny,iKernel);
    
    size_t nBytes = nx * ny * sizeof(float); // 16MB

    dim3 block(blockx, blocky);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx*ny);

    // transpose at host side
    transposeHost(hostRef,h_A, nx,ny);

    // allocate device memory
    float *d_A,*d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // warmup to avoide startup overhead
    double iStart = seconds();
    warmup <<< grid, block >>> (d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup elapsed %f sec\n",iElaps);

    // kernel
    void (*kernel)(float *, float *, int, int);
    char *kernelName;

    // set up kernel
    switch (iKernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow       ";
        break;

    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol       ";
        break;

    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "NaiveRow      ";
        break;

    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "NaiveCol      ";
        break;

    case 4:
        kernel = &transposeUnroll4Row;
        kernelName = "Unroll4Row    ";
        grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
        break;

    case 5:
        kernel = &transposeUnroll4Col;
        kernelName = "Unroll4Col    ";
        grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
        break;

    // case 6:
    //     kernel = &transposeDiagonalRow;
    //     kernelName = "DiagonalRow   ";
    //     break;

    // case 7:
    //     kernel = &transposeDiagonalCol;
    //     kernelName = "DiagonalCol   ";
    //     break;
    }

    // run kernel
    iStart = seconds();
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    // calculate effective_bandwidth
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> effective "
           "bandwidth %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);
    CHECK(cudaGetLastError());

    // check kernel results
    if (iKernel > 1)
    {
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
        checkResult(hostRef, gpuRef, nx * ny, 1);
    }

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}