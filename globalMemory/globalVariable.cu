#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVal() {
    printf("device: %f\n", devData);
    devData += 2.0f;
}

int main(int argc, char **argv) {
    float val = 3.14f;
    cudaMemcpyToSymbol(devData, &val, sizeof(float));
    printf("host: %f\n", val);

    checkGlobalVal<<<1,1>>>();

    cudaMemcpyFromSymbol(&val, devData, sizeof(float));
    printf("host, changed value: %f\n", val);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}