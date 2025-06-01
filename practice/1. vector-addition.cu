#include<iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void vectorAdd(float *A, float *B, float *C, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N)
        C[idx] = A[idx] + B[idx]; 
}

int main(){
    float *A, *B, *C;
    int N = 100;
    A = (float*)malloc(sizeof(float) * N);
    B = (float*)malloc(sizeof(float) * N);
    C = (float*)malloc(sizeof(float) * N);
    for(int i=0; i<N; i++){
        A[i] = i;
        B[i] = i;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * N);
    cudaMalloc((void**)&d_B, sizeof(float) * N);
    cudaMalloc((void**)&d_C, sizeof(float) * N);

    cudaMemcpy(d_A, A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N, cudaMemcpyHostToDevice);

    vectorAdd<<<1, 100>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<100; i++){
        cout<<C[i] <<"\t";
    }
}