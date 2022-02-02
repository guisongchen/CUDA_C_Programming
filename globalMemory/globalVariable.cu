#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData[5];

__global__ void checkGlobalVal() {
    devData[threadIdx.x] *= threadIdx.x;
    printf("device threadIdx:%d, val:%f\n", threadIdx.x, devData[threadIdx.x]);
}

int main(int argc, char **argv) {
    float val[5];
    for (int i = 0; i < 5; i++)
        val[i] = 3.14f;
    cudaMemcpyToSymbol(devData, val, 5*sizeof(float));
    for (int i = 0; i < 5; i++)
        printf("host idx:%d, val:%f\n", i, val[i]);

    checkGlobalVal<<<1,5>>>();

    cudaMemcpyFromSymbol(val, devData, 5*sizeof(float));
    for (int i = 0; i < 5; i++)
        printf("host idx:%d, changed val:%f\n", i, val[i]);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}