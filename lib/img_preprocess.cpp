#include "img_preprocess.h"

ImageRGB::ImageRGB(int height, int width) {
    this->pixel = new Pixel[height * width];
    this->shape[0] = height;
    this->shape[1] = width;
}

ImageRGB::ImageRGB(const std::string &imgPath) {
    cv::Mat img = cv::imread(imgPath);

    // Init infor of image size
    this->shape[0] = img.size().height;
    this->shape[1] = img.size().width;
    this->shape[2] = img.channels();

    // Init color values of the image
    
    this->pixel= new Pixel[shape[0] * shape[1]];
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            this->pixel[i * shape[1] + j].R = pixel[2];
            this->pixel[i * shape[1] + j].G = pixel[1];
            this->pixel[i * shape[1] + j].B = pixel[0];
            
        }
    }
}

ImageRGB::~ImageRGB() {
    if (this->pixel != NULL)  delete[] pixel;
}

void ImageRGB::printPixel() {
    std::cout << "Image size: " << this->shape[0] << " x " << this->shape[1] << std::endl;
    for (int i = 0; i < this->shape[0]; i++) {
        for (int j = 0; j < this->shape[1]; j++) {
            std::cout << "(" << (int)this->pixel[i * this->shape[1] + j].R 
                    << ", " << (int)this->pixel[i * this->shape[1] + j].G 
                    << ", " << (int)this->pixel[i * this->shape[1] + j].B 
                    << ") ";
        }
        std::cout << std::endl;
    }
}

void ImageRGB::saveImage(std::string savePath) {
    cv::Mat img(this->shape[0], this->shape[1], CV_8UC3);
    for (int i = 0; i < this->shape[0]; i++) {
        for (int j = 0; j < this->shape[1]; j++) {
            img.at<cv::Vec3b>(i, j)[2] = this->pixel[i * this->shape[1] + j].R;
            img.at<cv::Vec3b>(i, j)[1] = this->pixel[i * this->shape[1] + j].G;
            img.at<cv::Vec3b>(i, j)[0] = this->pixel[i * this->shape[1] + j].B;
        }
    }
    cv::imwrite(savePath, img);
}

unsigned char *flattenReadImage(const std::string &filename, int &width, int &height) {
    cv::Mat img = cv::imread(filename);
    width = img.size().width; 
    height = img.size().height;

    std::cout << width << " " << height << std::endl;

    unsigned char *data = new unsigned char[width * height * 3];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            data[height*width*2 + i*width + j] = pixel[0];
            data[height*width*1 + i*width + j] = pixel[1];
            data[height*width*0 + i*width + j] = pixel[2];
            if (pixel[2] > 255 or pixel[1] > 255 or pixel[0] > 255) std::cout << "error" << std::endl;
        }
    }
    return data; 
}