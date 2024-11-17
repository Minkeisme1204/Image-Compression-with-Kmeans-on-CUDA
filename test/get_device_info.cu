#include "cuda_runtime.h"
#include <iostream>

void printDeviceProperties(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    int device;
    cudaGetDevice(&device);

    std::cout << "Device #" << deviceId << ": " << deviceProp.name << std::endl;
    std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "Max grid dimensions: (" 
          << deviceProp.maxGridSize[0] << ", "
          << deviceProp.maxGridSize[1] << ", "
          << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << std::endl;
    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
    std::cout << "Shared memory per block: " << sharedMemPerBlock << " bytes" << std::endl;

    int sharedMemPerSM;
    cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    std::cout << "Shared memory per SM: " << sharedMemPerSM << " bytes" << std::endl;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
    } else {
        std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl;
        for (int i = 0; i < deviceCount; ++i) {
            printDeviceProperties(i);
        }
    }

    return 0;
}
