#ifndef CUDA
#define CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "config.h"
#include <iostream>

#define BLOCKS 1024
#define THREADS 512

#define R_FLAG 0
#define G_FLAG 1
#define B_FLAG 2

// all the API designed following architectures 2D Grid, 3D Blocks

// Calculate the distance between a pixel with every single centroid

// Paralell functions 
/*
1. Calculate the distance between a pixel with every single centroids in X/Y/Z axis
2. Sum all the distances above to calculate the real distance 
3. Check all the distances from a pixel with every single centroids and assign the minimun distanced centroids
4. Calculate Avarage value RGB and update new centroids'value
5. Update the new value of image 
*/
__global__ void findTheClosestCentroidKernel(unsigned char *image, unsigned char* centroids, int *result, int *image_cluster, int height, int width, int k);
__global__ void sumPixelCentroid(unsigned char *image, int *image_cluster, int* sum_centroid, int *coutn, int height, int width, int k);
__global__ void updateCentroidKernel(unsigned char* centroids, int *sum_centroid, int *count, int height, int width, int k); 

#endif