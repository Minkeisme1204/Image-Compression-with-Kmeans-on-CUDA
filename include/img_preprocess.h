#ifndef OPENCV
#define OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <iostream>

class Pixel {
    public: 
        unsigned char R;
        unsigned char G;
        unsigned char B;
};

class ImageRGB {
    public: 
        Pixel *pixel; // Array contains RGB values of each pixel of the image
        ImageRGB(const std::string &imgPath);
        ~ImageRGB();
        ImageRGB(int height, int width);
        void printPixel();
        void saveImage(std::string savePath);
        void saveImageCuda(std::string savePath);
        int getWidth() { return this->shape[1]; }
        int getHeight() { return this->shape[0]; }

    private: 
        int shape[2]; // [0] is height, [1] is width, [2] is channels
};

unsigned char *flattenReadImage(const std::string &filename, int &width, int &height);


