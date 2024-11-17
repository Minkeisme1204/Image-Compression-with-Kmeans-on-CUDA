#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

// Constants
#define MAX_ITERATIONS 100
#define TOLERANCE 1e-4

// CUDA kernel for assigning each pixel to the closest centroid
__global__ void assign_clusters(const unsigned char *image, int *labels, float *centroids, int num_pixels, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    // Load pixel color
    float r = image[3 * idx];
    float g = image[3 * idx + 1];
    float b = image[3 * idx + 2];

    // Find nearest centroid
    float min_dist = 12122004;
    int closest_centroid = 0;
    for (int i = 0; i < k; ++i) {
        float r_centroid = centroids[3 * i];
        float g_centroid = centroids[3 * i + 1];
        float b_centroid = centroids[3 * i + 2];

        float dist = (r - r_centroid) * (r - r_centroid) +
                     (g - g_centroid) * (g - g_centroid) +
                     (b - b_centroid) * (b - b_centroid);
        if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = i;
        }
    }
    labels[idx] = closest_centroid;
}

// CUDA kernel for updating centroids
__global__ void update_centroids(const unsigned char *image, int *labels, float *centroids, int *cluster_sizes, int num_pixels, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int cluster = labels[idx];
    atomicAdd(&centroids[3 * cluster], image[3 * idx]);
    atomicAdd(&centroids[3 * cluster + 1], image[3 * idx + 1]);
    atomicAdd(&centroids[3 * cluster + 2], image[3 * idx + 2]);
    atomicAdd(&cluster_sizes[cluster], 1);
}

// Host function to initialize centroids randomly
void initialize_centroids(float *centroids, const unsigned char *image, int num_pixels, int k) {
    srand(0);
    for (int i = 0; i < k; ++i) {
        int idx = rand() % num_pixels;
        centroids[3 * i] = image[3 * idx];
        centroids[3 * i + 1] = image[3 * idx + 1];
        centroids[3 * i + 2] = image[3 * idx + 2];
    }
}

// Host function to perform K-means clustering
void kmeans_cuda(unsigned char *image, int width, int height, int k) {
    int num_pixels = width * height;
    int image_size = num_pixels * 3 * sizeof(unsigned char);
    int centroid_size = k * 3 * sizeof(float);
    int label_size = num_pixels * sizeof(int);

    // Allocate memory on device
    unsigned char *d_image;
    int *d_labels, *d_cluster_sizes;
    float *d_centroids;
    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_labels, label_size);
    cudaMalloc(&d_centroids, centroid_size);
    cudaMalloc(&d_cluster_sizes, k * sizeof(int));

    // Copy image data to device
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    // Initialize centroids on host and copy to device
    std::vector<float> h_centroids(k * 3);
    initialize_centroids(h_centroids.data(), image, num_pixels, k);
    cudaMemcpy(d_centroids, h_centroids.data(), centroid_size, cudaMemcpyHostToDevice);

    // Define CUDA kernel launch parameters
    int threads_per_block = 256;
    int blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    // Run K-means iterations
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // Reset cluster sizes and centroids
        cudaMemset(d_centroids, 0, centroid_size);
        cudaMemset(d_cluster_sizes, 0, k * sizeof(int));

        // Step 1: Assign clusters
        assign_clusters<<<blocks, threads_per_block>>>(d_image, d_labels, d_centroids, num_pixels, k);
        cudaDeviceSynchronize();

        // Step 2: Update centroids
        update_centroids<<<blocks, threads_per_block>>>(d_image, d_labels, d_centroids, d_cluster_sizes, num_pixels, k);
        cudaDeviceSynchronize();

        // Copy updated centroids and sizes back to host
        cudaMemcpy(h_centroids.data(), d_centroids, centroid_size, cudaMemcpyDeviceToHost);
        std::vector<int> h_cluster_sizes(k);
        cudaMemcpy(h_cluster_sizes.data(), d_cluster_sizes, k * sizeof(int), cudaMemcpyDeviceToHost);

        // Normalize centroids
        bool convergence = true;
        for (int i = 0; i < k; ++i) {
            if (h_cluster_sizes[i] > 0) {
                h_centroids[3 * i] /= h_cluster_sizes[i];
                h_centroids[3 * i + 1] /= h_cluster_sizes[i];
                h_centroids[3 * i + 2] /= h_cluster_sizes[i];
            }
        }
        cudaMemcpy(d_centroids, h_centroids.data(), centroid_size, cudaMemcpyHostToDevice);
    }

    // Step 3: Apply new colors to image based on final centroids
    for (int i = 0; i < num_pixels; ++i) {
        int cluster = h_centroids[d_labels[i]];
        image[3 * i] = h_centroids[3 * cluster];
        image[3 * i + 1] = h_centroids[3 * cluster + 1];
        image[3 * i + 2] = h_centroids[3 * cluster + 2];
    }

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_labels);
    cudaFree(d_centroids);
    cudaFree(d_cluster_sizes);
}

#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <num_clusters>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    int k = std::stoi(argv[2]);

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return 1;
    }

    if (img.channels() == 4) {
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    int width = img.cols;
    int height = img.rows;

    // Tạo một mảng mới cho dữ liệu ảnh
    unsigned char *image_data = new unsigned char[width * height * 3];
    std::memcpy(image_data, img.data, width * height * 3);

    // Gọi hàm K-means nén ảnh
    kmeans_cuda(image_data, width, height, k);

    // // Chuyển dữ liệu kết quả từ CUDA trở lại OpenCV
    // std::memcpy(img.data, image_data, width * height * 3);

    // // Lưu ảnh nén
    // cv::imwrite("compressed_image.png", img);
    // std::cout << "Image saved as compressed_image.png" << std::endl;

    delete[] image_data; // Giải phóng bộ nhớ

    return 0;
}
