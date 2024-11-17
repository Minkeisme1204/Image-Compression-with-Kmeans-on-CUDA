#ifndef KMEANS
#define KMEANS
    #include "img_preprocess.h"
    #include <iostream>
    #include <random>
    #include <cmath>
    #include <fstream>
    #include "config.h"

// #define DEBUG

class Kmeans {
    private: 
        int k; // Input compression parameters
        int iters; // iterations for finding the best K
        ImageRGB* image; // Image Object for later calculations ** Size of memory allocated H x W x 3   // Input compression parameters (function .getHeight(), .getWidth())

        Pixel *centroid; // Array of pixel value of each centroids ** Size of memory allocated K   // Input compression parameters 
        int *image_cluster; // Matrix of centroids that each pixel on the image belongs to ** Size of memory allocated H x W   // Input compression parameters 
        void find_closest_centroid(); // To update the image_cluster array
        void update_centroid(); // To update the centroid array

    public: 
        Kmeans(const std::string &filename, int k);
        ~Kmeans();
        void build(int iters);
        double compile(const std::string &filename);
        int getK() { return this->k; }
        int writeToFileTXT(const std::string &filename); 
        Pixel *getCentroid() {return this->centroid;}
};
#endif