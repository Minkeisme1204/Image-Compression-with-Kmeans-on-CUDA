#include "kmeans.h"
#include <filesystem>
#include <string> 
#include "config.h"
#include <chrono>

// #define DEBUG
int main(int argc, char **argv) {
    std::string data_dir = DATA_PATH; 
    std::string output = OUTPUT_PATH; 
    for (const auto& entry: std::filesystem::directory_iterator(data_dir)) {
        auto start = std::chrono::high_resolution_clock::now();

        std::string path = entry.path().string();
        Kmeans* kmeans = new Kmeans(path, 200);
        kmeans->build(5000);
        kmeans->compile("compressed.jpg");
        // std::cout << "Loss value: " << loss << "\n";
        // std::cout << kmeans->writeToFileTXT("compressed.txt") << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time running non_parallel: " << duration.count() << " ms" << std::endl;
    }
}