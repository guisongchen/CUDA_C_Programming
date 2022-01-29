#include <cuda_runtime.h>
#include <stdio.h>

void initData(float *dataPtr, int size);
void sumArrayOnHost(float *pa, float *pb, float *pc, const int size);
void checkResult(float *hRet, float *dRet, const int size);
__global__ void sumArrayOnGPU(float *pa, float *pb, float *pc);

int main(int argc, char **argv) {
    int dev = 0;
    cudaSetDevice(dev);

    int num = 32;
    size_t nBytes = num * sizeof(float);
    printf("vector size:%d\n", num);

    float *h_a, *h_b, *hRet, *dRet;
    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    hRet = (float *)malloc(nBytes);
    dRet = (float *)malloc(nBytes);

    initData(h_a, num);
    initData(h_b, num);

    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, nBytes);
    cudaMalloc((float**)&d_b, nBytes);
    cudaMalloc((float**)&d_c, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    dim3 block(num);
    dim3 grid(num/block.x);
    sumArrayOnGPU<<<grid, block>>>(d_a, d_b, d_c);
    printf("config: block=%d, thread=%d\n", grid.x, block.x);

    // copy kernel result back to host
    cudaMemcpy(dRet, d_c, nBytes, cudaMemcpyDeviceToHost);
    
    // host side result
    sumArrayOnHost(h_a, h_b, hRet, num);

    // check result
    checkResult(hRet, dRet, num);

    free(h_a);
    free(h_b);
    free(hRet);
    free(dRet);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}

void initData(float *dataPtr, int size) {
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++) {
        dataPtr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArrayOnHost(float *pa, float *pb, float *pc, const int size) {
    for (int i = 0; i < size; i++) {
        pc[i] = pa[i] + pb[i];
    }
}

__global__ void sumArrayOnGPU(float *pa, float *pb, float *pc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    pc[i] = pa[i] + pb[i];
}

void checkResult(float *hRet, float *dRet, const int size) {
    double epsilon = 1e-8;
    bool match = 1;
    for (int i = 0; i < size; i++) {
        if (abs(hRet[i]-dRet[i]) > epsilon) {
            match = 0;
            printf("Array not match! idx=%d, host=%5.2f, gpu=%5.2f\n", i, hRet[i], dRet[i]);
            break;
        }
    }

    if (match) {
        printf("Array match!\n");
    }
}



