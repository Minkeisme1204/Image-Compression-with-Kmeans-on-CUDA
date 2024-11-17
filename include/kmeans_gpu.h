#ifndef KMEANS_GPU
#define KMEANS_GPU
    #include "img_preprocess.h"
    #include <iostream>
    #include <random>
    #include <cmath>
    #include <fstream>
    #include "config.h"
    #include "cuda_tools.h"
    #include <cmath>

class Kmeans {
    private: 
        int k; // Input compression parameters
        int iters; // iterations for finding the best K
        int height, width;
        unsigned char* imageRGB; // Image Object for later calculations ** Size of memory allocated H x W x 3   // Input compression parameters (function .getHeight(), .getWidth())

        unsigned char* centroidRGB; // Array of pixel value of each centroids ** Size of memory allocated K   // Input compression parameters 
        int *image_cluster; // Matrix of centroids that each pixel on the image belongs to ** Size of memory allocated H x W   // Input compression parameters 
         // To update the image_cluster array
         // To update the centroid array

    public: 
        Kmeans(const std::string &filename, int k);
        ~Kmeans();
        void build(int iters);
        double compile();
        int getK() { return this->k; }
        void find_closest_centroid();
        void update_centroid();
        void export_image(const std::string &filename);
};
#endif