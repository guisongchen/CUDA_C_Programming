#include <stdio.h>

__global__ void helloFromGPU(void) {
    if (threadIdx.x == 5) {
        printf("Hello from GPU thread %d\n", threadIdx.x);
    }
}

int main(void) {
    printf("Hello from CPU\n");

    helloFromGPU<<<1, 10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();

    return 0;
}
