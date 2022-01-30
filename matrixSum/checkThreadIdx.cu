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

void initData(int *dataPtr, int size);
double cpuSecond();
void printMatrix(int *mat, const int nx, const int ny);
__global__ void printfThreadIndex(int *mat, const int nx, const int ny);

int main(int argc, char **argv) {
    printf("starting...\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // notice: x,y is more like image coorinate
    // origin is at top left corner
    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);

    int *h_a;
    h_a = (int*)malloc(nBytes);
    initData(h_a, nxy);
    printMatrix(h_a, nx, ny);

    int *d_a;
    cudaMalloc((void**)&d_a, nBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    printfThreadIndex<<<grid, block>>>(d_a, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    free(h_a);

    cudaDeviceReset();

    return 0;
}

void initData(int *dataPtr, int size) {
    for (int i = 0; i < size; i++) {
        dataPtr[i] = i;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

void printMatrix(int *mat, const int nx, const int ny) {
    printf("matrix: (%d,%d)\n", nx, ny);
    for (int i = 0; i < nx*ny; i++) {
        printf("%3d", mat[i]);
        if (i%nx == nx-1) {
            printf("\n");
        }
    }
    printf("\n");
}

__global__ void printfThreadIndex(int *mat, const int nx, const int ny) {
    int ix =  threadIdx.x + blockDim.x * blockIdx.x;
    int iy =  threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy*nx;

    printf("%d %d %d\n", ix, iy, mat[idx]);
}
