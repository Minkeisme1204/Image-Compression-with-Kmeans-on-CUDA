#include "img_preprocess.h"
#include <filesystem>
#include <string> 
#include "config.h"
 
int main(int argc, char **argv) {
    std::string data_dir = DATA_PATH;
    std::string output = OUTPUT_PATH; 
    for (const auto& entry: std::filesystem::directory_iterator(data_dir)) {
        std::string path = entry.path().string();
        std::cout << "Image: " << path << std::endl;
        ImageRGB *image = new ImageRGB(path);

        image->saveImage(output + "/" + argv[1]);
        delete image;  // Release memory
    }
}