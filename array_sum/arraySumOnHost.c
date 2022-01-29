#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

void initData(float *dataPtr, int size);
void sumArrayOnHost(float *pa, float *pb, float *pc, const int size);

int main(int argc, char **argv) {
    clock_t start, stop;

    start = clock();

    int num = 1024;
    size_t nBytes = num * sizeof(float);

    float *h_a, *h_b, *h_c;
    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_c = (float *)malloc(nBytes);

    initData(h_a, num);
    initData(h_b, num);
    sumArrayOnHost(h_a, h_b, h_c, num);

    free(h_a);
    free(h_b);
    free(h_c);

    stop = clock();

    printf("duration: %lf\n", ((double)(stop - start)) / CLOCKS_PER_SEC);
    
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

