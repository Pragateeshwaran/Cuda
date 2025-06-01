#include <stdio.h>
#include <cuda_runtime.h>
#define N 8

__global__ void getBitFlags(int *input, int *flags, int bit) {
    int i = threadIdx.x;
    flags[i] = (input[i] >> bit) & 1;
}

__global__ void reorder(int *input, int *output, int *flags) {
    int i = threadIdx.x;
    __shared__ int zeroPos, onePos;
    
    if (i == 0) { zeroPos = 0; onePos = 0; }
    __syncthreads();
    
    if (flags[i] == 0) {
        int pos = atomicAdd(&zeroPos, 1);
        output[pos] = input[i];
    }
    __syncthreads();
    
    if (flags[i] == 1) {
        int pos = zeroPos + atomicAdd(&onePos, 1);
        output[pos] = input[i];
    }
}

void radixSort(int *input) {
    int *d_input, *d_output, *d_flags;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMalloc(&d_flags, N * sizeof(int));
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    for (int bit = 0; bit < 32; bit++) {
        getBitFlags<<<1, N>>>(d_input, d_flags, bit);
        reorder<<<1, N>>>(d_input, d_output, d_flags);
        int *temp = d_input; d_input = d_output; d_output = temp;
    }
    
    cudaMemcpy(input, d_input, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_flags);
}

int main() {
    int input[N] = {7, 3, 2, 8, 1, 4, 6, 5};
    radixSort(input);
    for (int i = 0; i < N; i++) printf("%d ", input[i]);
    return 0;
}