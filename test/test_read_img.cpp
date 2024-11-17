#include "img_preprocess.h"
#include <filesystem>
#include <string> 
#include "config.h"
 
int main() {
    std::string data_dir = DATA_PATH; 
    for (const auto& entry: std::filesystem::directory_iterator(data_dir)) {
        std::string path = entry.path().string();
        std::cout << path << std::endl;
        std::cout << "Image: " << path << std::endl;
        ImageRGB *image = new ImageRGB(path);
        image->printPixel();
        std::cout << (int)image->pixel[59503].B <<std::endl;
        delete image;  // Release memory
    }
}