#include<iostream>
#include<cuda_runtime.h>

__global__ void kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    kernel<<<1, 1024>>>();
    cudaDeviceSynchronize();
    return 0;
}   