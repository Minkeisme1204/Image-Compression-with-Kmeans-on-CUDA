#include "kmeans_gpu.h"

#define DEBUG
#define RANDOM

Kmeans::Kmeans(const std::string &filename, int k) {
    // Initialize the centroid values
    this->k = k;
    this->imageRGB = flattenReadImage(filename, this->width, this->height);
    printf("The image size: %d x %d\n", this->height, this->width);
    // Init random generator and seed  
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_h(0, this->height - 1);
    std::uniform_int_distribution<int> dist_w(0, this->width - 1);

    this->centroidRGB = new unsigned char [this->k*3];

    for (int i = 0; i < this->k; i++) {
        #ifdef RANDOM
        int r_id = (this->height * this->width * 0 + dist_h(gen) * this->width + dist_w(gen));
        int g_id = (this->height * this->width * 1 + dist_h(gen) * this->width + dist_w(gen));
        int b_id = (this->height * this->width * 2 + dist_h(gen) * this->width + dist_w(gen));
        #else
        int r_id = (this->height * this->width * 0 + 40*i * this->width + 23*i);
        int g_id = (this->height * this->width * 1 + 40*i * this->width + 23*i);
        int b_id = (this->height * this->width * 2 + 40*i * this->width + 23*i);  
        #endif

        #ifdef DEBUG_Succeeded
        printf("(%d, %d, %d)", i, r_id, g_id, b_id);
        #endif

        this->centroidRGB[this->k * 0 + i] = this->imageRGB[r_id]; 
        this->centroidRGB[this->k * 1 + i] = this->imageRGB[g_id];
        this->centroidRGB[this->k * 2 + i] = this->imageRGB[b_id];
        std::cout << "Initial Cluster " << i  << " " << (int)this->centroidRGB[this->k * 0 + i] << " " << (int)this->centroidRGB[this->k * 1 + i] << " " << (int)this->centroidRGB[this->k* 2 + i] << std::endl; 
    }

    this->image_cluster = new int[this->height * this->width];

    #ifdef DEBUG_Succeeded
    std::cout << "Read infor of image. Size of image: " << this->height << " " << this->width << std::endl;
    for (int i = 0; i < this->height*this->width; i++) {
            printf("Pixel %d: (%d, %d, %d)\n ",i ,this->imageRGB[this->width*this->height*0 + i], this->imageRGB[this->width*this->height*1 + i], this->imageRGB[this->width*this->height*2 + i]);
        std::cout << std::endl;
    }
    #endif
}

Kmeans::~Kmeans() {
    if (this->centroidRGB != NULL) delete[] this->centroidRGB;
    if (this->image_cluster!= NULL) delete[] this->image_cluster;
    if (this->imageRGB != NULL) delete[] this->imageRGB;
}

void Kmeans::find_closest_centroid() {
    unsigned char *gpu_image, *gpu_centroids;
    cudaMalloc((void **)&gpu_image, this->height*this->width*3*sizeof(unsigned char));
    cudaMalloc((void **)&gpu_centroids, this->k*3*sizeof(unsigned char));

    cudaMemcpy(gpu_image,this->imageRGB, this->height*this->width*3, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids, this->centroidRGB, this->k*3, cudaMemcpyHostToDevice);

    // Call the GPU kernel
    // Function 1 Find the closest centroid to each pixel
    int *gpu_distance, *image_cluster_gpu;
    cudaMalloc((void **)&gpu_distance, this->height*this->width*sizeof(int));
    cudaMalloc((void **)&image_cluster_gpu, this->height*this->width*sizeof(int));

    dim3 blockDim(min(this->height, this->width), 1, 1);
    dim3 gridDim((int)ceil((double)this->height *this->width / blockDim.x), 1, 1);

    // Run the kernel 
    findTheClosestCentroidKernel <<<gridDim, blockDim, this->k*3>>> (gpu_image, gpu_centroids, gpu_distance, image_cluster_gpu, this->height, this->width, this->k);
    cudaDeviceSynchronize();
    cudaMemcpy(this->image_cluster, image_cluster_gpu, this->height*this->width*sizeof(int), cudaMemcpyDeviceToHost);
    #ifdef DEBUG_Succeeded
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        int *distance = new int[this->height*this->width*this->k]; 
        cudaMemcpy(distance, gpu_distance, this->height*this->width*this->k*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < this->height*this->width; i++) {
            for (int j = 0; j < this->k; j++) {
                printf("Distance pixel %d, centroid %d: %d\n", i, j, distance[this->height*this->width*j + i]);
            }
        }
    #endif
    #ifdef DEBUG_Succeeded
    std::cout << "Image cluster: " << std::endl;
    for (int i = 0; i < this->height*this->width; i++) {
        printf("Cluster of Pixel %d: centroid %d\n", i, this->image_cluster[i]);
    }
    #endif 
    #ifdef DEBUG_Succeeded
        std::cout << "Kernel 1: findTheClosestCentroid is running" << std::endl;
        std::cout << "BlockSize: " << blockDim.x << " " << blockDim.y << " " << blockDim.z << std::endl;
        std::cout << "GridSize: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
        printf("GridSize: %d, %d, %d\n", gridDim.x*blockDim.x, gridDim.y*blockDim.y, gridDim.z*blockDim.z);
    #endif
    // Free 4 arrays in Cuda global memory
    cudaFree(gpu_image);
    cudaFree(gpu_centroids);
    cudaFree(gpu_distance);
    cudaFree(image_cluster_gpu);
}


void Kmeans::update_centroid() {
    /*
    Input: this->centroids, this->image_cluster
    Output: this->centroids updated
    */

    // Fucntion 2: Get sum and number of pixels in a centroid cluster
    int *sum_centroid_gpu, *count_gpu, *image_cluster_gpu; 
    unsigned char *image_gpu; 

    cudaMalloc((void **)&image_gpu, sizeof(unsigned char)*this->width*this->height*3); 
    cudaMalloc((void **)&image_cluster_gpu, sizeof(int)*this->width*this->height); 
    cudaMalloc((void **)&sum_centroid_gpu, sizeof(int)*this->k*3);
    cudaMalloc((void **)&count_gpu, sizeof(int)*this->k);

    cudaMemcpy(image_gpu, this->imageRGB, sizeof(unsigned char)*this->width*this->height*3, cudaMemcpyHostToDevice);
    cudaMemcpy(image_cluster_gpu, this->image_cluster, sizeof(int)*this->width*this->height, cudaMemcpyHostToDevice);

    dim3 gridDim1(max(this->height, this->width), 1, 1); 
    dim3 blockDim1(min(this->height, this->width), 1, 1);

    sumPixelCentroid <<<gridDim1, blockDim1>>> (image_gpu, image_cluster_gpu, sum_centroid_gpu, count_gpu, this->height, this->width, this->k); 
    cudaDeviceSynchronize();
    #ifdef DEBUG_Succeeded
    int *sum_centroid, *count; 
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    count = new int[this->k]; 
    cudaMemcpy(count, count_gpu, sizeof(int)*this->k, cudaMemcpyDeviceToHost);

    sum_centroid = new int[this->k*3];
    cudaMemcpy(sum_centroid, sum_centroid_gpu, sizeof(int)*this->k*3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < this->k; i++) {
        printf("Centroid %d has total pixels: %d\t(%d, %d, %d)\n", i, count[i], sum_centroid[this->k*0 + i], sum_centroid[this->k*1 + i], sum_centroid[this->k*2 + i]); 
    }
    #endif 
    #ifdef DEBUG_Succeeded
        std::cout << "Kernel 2: preCalculateDistance is running" << std::endl;
        std::cout << "BlockSize: " << blockDim1.x << " " << blockDim1.y << " " << blockDim1.z << std::endl;
        std::cout << "GridSize: " << gridDim1.x << " " << gridDim1.y << " " << gridDim1.z << std::endl;
        printf("GridSize: %d, %d, %d\n", gridDim1.x*blockDi1.x, gridDim1.y*blockDim1.y, gridDim1.z*blockDim1.z);
    #endif
    cudaFree(image_cluster_gpu);

    // Function 2: Calculate new centroids
    unsigned char* centroid_gpu; 
    cudaMalloc((void**)&centroid_gpu, sizeof(unsigned char)*this->k*3);
   
    // cudaMemcpy(centroid_gpu, this->centroidRGB, sizeof(int)*this->k*3, cudaMemcpyHostToDevice);

    dim3 gridDim2(this->k, 1, 1); 
    dim3 blockDim2(1, 1, 1);

    #ifdef DEBUG_Succeeded
        std::cout << "Centroids before updated" << std::endl;
        for (int i = 0; i < this->k; i++) {
            std::cout << "Cluster " << i << ": (" << (int)this->centroidRGB[k*0 + i] << ", " << (int)this->centroidRGB[k*1 + i] << ", " << (int)this->centroidRGB[k*2 + i] << ")" << std::endl; 
        } 
    #endif

    updateCentroidKernel<<<gridDim2, blockDim2>>> (centroid_gpu, sum_centroid_gpu, count_gpu, this->height, this->width, this->k);
    cudaDeviceSynchronize();
    cudaMemcpy(this->centroidRGB, centroid_gpu, sizeof(unsigned char)*this->k*3, cudaMemcpyDeviceToHost);
    
    #ifdef DEBUG_Succeeded
        std::cout << "Centroids after updated" << std::endl;
        for (int i = 0; i < this->k; i++) {
            printf("Centroid %d after updated: %d, %d, %d\n", i, this->centroidRGB[this->k*0 + i], this->centroidRGB[this->k*1 + i], this->centroidRGB[this->k*2 + i]); 
        } 
    #endif
    // Free 4 arrays in Cuda global memory
    cudaFree(centroid_gpu);
    cudaFree(sum_centroid_gpu);
    cudaFree(count_gpu);
    cudaFree(image_gpu);

    cudaDeviceSynchronize(); 
    // cudaDeviceReset();
}

void Kmeans::build(int iters=20) {
    this->iters = iters; 
}

double Kmeans::compile() { // Function return the loss value compared to the raw image
    for (int i = 0; i < this->iters; i++) {
        std::cout << "Iteration " << i << "\n";
        this->find_closest_centroid();
        this->update_centroid();
    }
    return 0;
}

void Kmeans::export_image(const std::string &filename) {
    std::string output = OUTPUT_PATH; 

    ImageRGB *img = new ImageRGB(this->height, this->width);
    std::cout << "Output size: " << img->getHeight() << " " << img->getWidth() << " " << std::endl;

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            img->pixel[i * this->width + j].R = this->centroidRGB[this->k*0 + this->image_cluster[this->width*i + j]];
            img->pixel[i * this->width + j].G = this->centroidRGB[this->k*1 + this->image_cluster[this->width*i + j]];
            img->pixel[i * this->width + j].B = this->centroidRGB[this->k*2 + this->image_cluster[this->width*i + j]];
        }
    } 

    img->saveImage(output + "/" + filename);
    delete img;
}

