#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) \
{   \
    const cudaError_t error = call; \
    if (error != cudaSuccess)    \
    {   \
    printf("Error: %s:%d, ", __FILE__, __LINE__);    \
    printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));    \
    exit(1);    \
    }   \
}   \

void initData(float *dataPtr, int size);
void sumArrayOnHost(float *pa, float *pb, float *pc, const int size);
void checkResult(float *hRet, float *dRet, const int size);
double cpuSecond();
__global__ void sumArrayOnGPU(float *pa, float *pb, float *pc, const int size);

int main(int argc, char **argv) {
    printf("starting...\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    int num = 1 << 24;
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
    int threadNum = 256;
    dim3 block(threadNum);
    dim3 grid((num + block.x -1)/block.x);

    double start = cpuSecond();
    sumArrayOnGPU<<<grid, block>>>(d_a, d_b, d_c, num);
    cudaDeviceSynchronize();
    double during = cpuSecond() - start;
    printf("GPU config:block=%d, thread=%d, time elapsed %f\n", grid.x, block.x, during);

    // copy kernel result back to host
    cudaMemcpy(dRet, d_c, nBytes, cudaMemcpyDeviceToHost);
    
    // host side result
    start = cpuSecond();
    sumArrayOnHost(h_a, h_b, hRet, num);
    during = cpuSecond() - start;
    printf("CPU time elapsed %f\n", during);

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

__global__ void sumArrayOnGPU(float *pa, float *pb, float *pc, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
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

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}



