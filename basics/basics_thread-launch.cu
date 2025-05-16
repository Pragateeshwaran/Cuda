#include<iostream>
#include<cuda_runtime.h>

__global__ void kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    kernel<<<1, 1024>>>(); // if 1025 will not produce any output because the kernel have only 1024 threads
    cudaDeviceSynchronize();
    kernel<<<1, 1025>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}   