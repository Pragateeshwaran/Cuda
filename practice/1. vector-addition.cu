#include<cuda_runtime.h>
#include<stdio.h>

__global__  void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
    printf("%f ", C[i]);    
}

int main() {
     
}