#include <cuda_runtime.h>
#include <iostream>

__global__ void reverseArray(float *arr, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = N - i - 1; // Mirror index

    if (i < j) { // Only process the first half
        float temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void reverseArrayHost(float *arr, int N) {
    float *d_arr;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    reverseArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    const int N = 10;
    float arr[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::cout << "Original Array: ";
    for (int i = 0; i < N; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;

    reverseArrayHost(arr, N);

    std::cout << "Reversed Array: ";
    for (int i = 0; i < N; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;

    return 0;
}
