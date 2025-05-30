#include<iostream>
#include<cuda_runtime.h>
using namespace std;
int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout<<deviceCount<<endl;
    for(int i=0; i<deviceCount; i++){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        cout<<"Device name: " <<deviceProp.name<<endl;
        cout<<"Total global memory: " <<deviceProp.totalGlobalMem<<endl;
        cout<<"Shared memory per block: " <<deviceProp.sharedMemPerBlock<<endl;
        cout<<"Registers per block: " <<deviceProp.regsPerBlock<<endl;
        cout<<"Warp size: " <<deviceProp.warpSize<<endl;
        cout<<"Memory pitch: " <<deviceProp.memPitch<<endl;
        cout<<"Max threads per block: " <<deviceProp.maxThreadsPerBlock<<endl;
        cout<<"Max threads dim: (" <<deviceProp.maxThreadsDim[0]<<", " 
            <<deviceProp.maxThreadsDim[1]<<", " <<deviceProp.maxThreadsDim[2]<<")"<<endl;
        cout<<"Max grid size: (" <<deviceProp.maxGridSize[0]<<", "
            <<deviceProp.maxGridSize[1]<<", " <<deviceProp.maxGridSize[2]<<")"<<endl;
        cout<<"Clock rate: " <<deviceProp.clockRate<<endl;
        cout<<"Total constant memory: " <<deviceProp.totalConstMem<<endl;
        cout<<"Compute capability: " <<deviceProp.major<<"."<<deviceProp.minor<<endl;
        cout<<"Texture alignment: " <<deviceProp.textureAlignment<<endl;
        cout<<"Texture pitch alignment: " <<deviceProp.texturePitchAlignment<<endl;
        cout<<"Device overlap: " <<deviceProp.deviceOverlap<<endl;
        cout<<"Multi-processor count: " <<deviceProp.multiProcessorCount<<endl;
    }
}