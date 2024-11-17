#include <iostream>
#include "kmeans_gpu.h"
#include <string>
#include <filesystem>
#include "config.h"
#include <chrono>

int main(int argc, char **argv) {
    std::string data_dir = DATA_PATH; 
    data_dir = data_dir + "/" + "image1.jpg";

    auto start = std::chrono::high_resolution_clock::now();

    Kmeans* kmeans = new Kmeans(data_dir, 200);
    std::cout << "Starting testing...\n";
    kmeans->build(5000); 
    kmeans->compile();
    kmeans->export_image("compressd_gpu.jpg");
    std::cout << "Done\n";

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time running parallel: " << duration.count() << " ms" << std::endl;
    cudaDeviceReset();
}