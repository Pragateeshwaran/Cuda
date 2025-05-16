#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    return 0;
}

