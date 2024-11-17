#include "cuda_tools.h"
#define DEBUG
// #define DEBUG_GPU

__global__ void findTheClosestCentroidKernel(unsigned char *image, unsigned char* centroids, int *result, int *image_cluster, int height, int width, int k) {
    /*
        Input: A 3D matrix from the preCalculateDistanceKernel 
        Output: A 2D matrix of the distance from each pixels to every single centroids
        Grid size = max(height, width), k, 1
        Block size = min(height, width), 1, 1
        Note: The block size should be adjusted to fit the GPU memory
    */
    extern __shared__ unsigned char centroidValue[];

    int ox = blockDim.x*gridDim.x;

    int image_R_id = 0*ox + blockDim.x*blockIdx.x + threadIdx.x; 
    int image_G_id = 1*ox + blockDim.x*blockIdx.x + threadIdx.x; 
    int image_B_id = 2*ox + blockDim.x*blockIdx.x + threadIdx.x; 

    int resultId = blockDim.x*blockIdx.x + threadIdx.x;
    __syncthreads();

    for (int i = 0; i < k; i++) {
        centroidValue[0*k + i] = centroids[0*k + i];
        centroidValue[1*k + i] = centroids[1*k + i];
        centroidValue[2*k + i] = centroids[2*k + i];    
    }
    __syncthreads();

    int min_distance = MAXN; 
    for (int i = 0; i < k; i++) {
        int d =   (image[image_R_id] - centroidValue[0*k + i])*(image[image_R_id] - centroidValue[0*k + i])
                + (image[image_G_id] - centroidValue[1*k + i])*(image[image_G_id] - centroidValue[1*k + i])
                + (image[image_B_id] - centroidValue[2*k + i])*(image[image_B_id] - centroidValue[2*k + i]);
        if (d < min_distance) {
            min_distance = d;
            image_cluster[resultId] = i;  // Update the minimum distance and the corresponding cluster id for the current pixel.
        }
    }
    __syncthreads();

    
    #ifdef DEBU
        if ((blockDim.x*blockIdx.x + threadIdx.x == 98883 or blockDim.x*blockIdx.x + threadIdx.x == 98882)  && (blockDim.y*blockIdx.y + threadIdx.y == 9 or blockDim.y*blockIdx.y + threadIdx.y ==8) ) {
            printf("Distance pixel %d (%d, %d, %d), centroid %d (%d, %d, %d): %d\n", 
            blockDim.x*blockIdx.x + threadIdx.x, image[image_R_id], image[image_G_id], image[image_B_id],
            blockDim.y*blockIdx.y + threadIdx.y, centroidValue[centroid_R_id], centroidValue[centroid_G_id], centroidValue[centroid_B_id],
            result[resultId]);
        }
    #endif
}

__global__ void sumPixelCentroid(unsigned char *image, int *image_cluster, int* sum_centroid, int *count, int height, int width, int k) {
    /*
    Input: The value of the image, the image_cluster (pixels labels)
    Output: The sum of pixel value each centroids, the number of pixels in a centroids

    Block size: min(height, width), 1, 1 
    Grid size: max(height, width), 1, 1
    */
    int pixelId = blockDim.x*blockIdx.x + threadIdx.x; 
    int n = height*width; 
    if (pixelId < n) {
        atomicAdd(&count[image_cluster[pixelId]], 1);
        int centroidId_R = k*0 + image_cluster[pixelId]; 
        int centroidId_G = k*1 + image_cluster[pixelId];
        int centroidId_B = k*2 + image_cluster[pixelId];
        atomicAdd(&sum_centroid[centroidId_R], (int)image[n*0 + pixelId]);
        atomicAdd(&sum_centroid[centroidId_G], (int)image[n*1 + pixelId]);
        atomicAdd(&sum_centroid[centroidId_B], (int)image[n*2 + pixelId]);
    }
    __syncthreads();
}


__global__ void updateCentroidKernel(unsigned char* centroids, int* sum_centroid,int *count, int height, int width, int k) {
    /*
    Input: Sum of the centroids, Count of pixels in centroid
    Output: centroid, updated
    
    Block Size: 1, 1, 1
    Grid Size: k, 1, 1
    */
    int centroidId = blockIdx.x; 
  
    if (count[centroidId] != 0) {
        centroids[0*k + centroidId] = sum_centroid[0*k + centroidId] / count[centroidId];
        centroids[1*k + centroidId] = sum_centroid[1*k + centroidId] / count[centroidId];
        centroids[2*k + centroidId] = sum_centroid[2*k + centroidId] / count[centroidId];
    }
    else {
        centroids[0*k + centroidId] = 0;
        centroids[1*k + centroidId] = 0;
        centroids[2*k + centroidId] = 0;
    }
    __syncthreads();

} 

