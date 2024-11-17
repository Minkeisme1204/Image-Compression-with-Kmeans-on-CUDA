#include "kmeans.h"
#define RANDOM
#define DEBUG
Kmeans::Kmeans(const std::string &filename, int k) {
    this->image = new ImageRGB(filename);
    // Initialize the centroid values
    this->k = k;
    this->centroid = new Pixel[this->k];

    // Init random generator and seed  
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_h(0, image->getHeight());
    std::uniform_int_distribution<int> dist_w(0, image->getWidth());

    #ifdef DEBUG
    std::cout << "Centroid value: \n";
    #endif


    for (int i = 0; i < this->k; i++) {
        #ifndef RANDOM
        this->centroid[i].R = image->pixel[i*40 * image->getWidth() + 23*i].R; 
        this->centroid[i].G = image->pixel[i*40 * image->getWidth() + 23*i].G;
        this->centroid[i].B = image->pixel[i*40 * image->getWidth() + 23*i].B;
        #else
        this->centroid[i].R = image->pixel[dist_h(gen) * image->getWidth() + dist_w(gen)].R; 
        this->centroid[i].G = image->pixel[dist_h(gen) * image->getWidth() + dist_w(gen)].G;
        this->centroid[i].B = image->pixel[dist_h(gen) * image->getWidth() + dist_w(gen)].B;
        #endif

        #ifdef DEBUG
        std::cout << (int)this->centroid[i].R << " " << (int)this->centroid[i].G << " " << (int)this->centroid[i].B << std::endl;
        #endif 
    }

    #ifdef DEBUG_Succeeded
    for (int i = 0; i < image->getHeight(); i++) {
        for (int j = 0; j < image->getWidth(); j++) {
            printf("Pixel %d: (%d, %d, %d)\n", i*image->getWidth() + j, image->pixel[i * image->getWidth() + j].R, image->pixel[i * image->getWidth() + j].G, image->pixel[i * image->getWidth() + j].B);
        }
    }
    #endif

    this->image_cluster = new int[this->image->getWidth() * this->image->getHeight()];
}

Kmeans::~Kmeans() {
    if (this->centroid!= NULL) delete[] this->centroid;
    if (this->image_cluster!= NULL) delete[] this->image_cluster;
    if (this->image != NULL) delete this->image;
}

void Kmeans::find_closest_centroid() {
    int n = this->image->getWidth() * this->image->getHeight();
    for (int i = 0; i < n; i++) {
        int dist = 12122004;

        // Compute the distance between the centroid and the pixel on image
        // Able to be Paralell 
        for (int j = 0; j < this->k; j++) {
            int deltaB = (image->pixel[i].B - centroid[j].B);
            int deltaG = (image->pixel[i].G - centroid[j].G);
            int deltaR = (image->pixel[i].R - centroid[j].R);
            
            #ifdef DEBUG_Succeeded
                printf("Delta pixel %d, centroid %d: (%d, %d, %d)\n ", i, j , abs(deltaR), abs(deltaG), abs(deltaB));
            #endif

            int d = deltaB*deltaB + deltaG*deltaG + deltaR*deltaR;
            #ifdef DEBU
                printf("Distance pixel %d, centroid %d: %d\t(%d, %d, %d)\n", i, j, d, deltaR, deltaG, deltaB);
            #endif
            if (d < dist) {
                dist = d;
                this->image_cluster[i] = j;
            }
        }
        #ifdef DEBU
            printf("Cluster of Pixel %d: centroid %d\n", i, this->image_cluster[i]);
        #endif 
    }
}

void Kmeans::update_centroid() {
    int n = this->image->getWidth() * this->image->getHeight();
    int *count = new int[this->k];
    for (int i = 0; i < this->k; i++) count[i] = 0;

    // int tempB[this->k]; 
    // int tempG[this->k];
    // int tempR[this->k];

    int *tempB = new int[this->k]; 
    int *tempG = new int[this->k];
    int *tempR = new int[this->k];  

    for (int i = 0; i < this->k; i++) {
        tempB[i] = 0;
        tempG[i] = 0;
        tempR[i] = 0;
    }

    // This code can be parallel
    for (int i = 0; i < n; i++) { 
        tempB[this->image_cluster[i]] += this->image->pixel[i].B;
        tempG[this->image_cluster[i]] += this->image->pixel[i].G;
        tempR[this->image_cluster[i]] += this->image->pixel[i].R;
        count[this->image_cluster[i]]++;
    }

    // This code can be parallel
    for (int i = 0; i < this->k; i++) {
        #ifdef DEBUG_Succeeded
            printf("Centroid %d has total pixels: %d\t(%d, %d, %d)\n", i, count[i], tempR[i], tempG[i], tempB[i]); 
        #endif
        if (count[i]!= 0) {
            this->centroid[i].B = tempB[i] / count[i];
            this->centroid[i].G = tempG[i] / count[i];
            this->centroid[i].R = tempR[i] / count[i];
        }
        else {
            // std::cout << "centroid " << i << " has no pixels\n";
            this->centroid[i].B = 0;
            this->centroid[i].G = 0;
            this->centroid[i].R = 0;
            continue;  // Skip centroids with no pixels, they will not affect the loss value and centroid computation.
        }
        #ifdef DEBUG_Succeeded
            printf("Centroid %d after updated: %d, %d, %d\n", i, this->centroid[i].R, this->centroid[i].G, this->centroid[i].B);
        #endif
    }
    delete[] tempB;
    delete[] tempG;
    delete[] tempR;
}

void Kmeans::build(int iters=20) {
    this->iters = iters; 
}

double Kmeans::compile(const std::string &filename) { // Function return the loss value compared to the raw image
    for (int i = 0; i < this->iters; i++) {
        #ifndef DEBU 
        std::cout << "Iteration " << i << "\n";
        #endif

        this->find_closest_centroid();
        this->update_centroid();
    }

    // Save the image 
    int n = this->image->getWidth() * this->image->getHeight();
    // Replace RGB values with centroids

    double loss = 0; 

    for (int i = 0; i < n; i++) {
        // Paralell able
        // loss += (double)(this->image->pixel[i].R - this->centroid[this->image_cluster[i]].R) * (this->image->pixel[i].R - this->centroid[this->image_cluster[i]].R)
        //         + (double)(this->image->pixel[i].B - this->centroid[this->image_cluster[i]].B) * (this->image->pixel[i].B - this->centroid[this->image_cluster[i]].B)
        //         + (double)(this->image->pixel[i].G - this->centroid[this->image_cluster[i]].G) * (this->image->pixel[i].G - this->centroid[this->image_cluster[i]].G);

        this->image->pixel[i].R = this->centroid[this->image_cluster[i]].R;
        this->image->pixel[i].G = this->centroid[this->image_cluster[i]].G;
        this->image->pixel[i].B = this->centroid[this->image_cluster[i]].B;
        #ifdef DEBUG_Succeeded
        printf("Pixel %d after updated: centroid %d (%d, %d, %d)\n", i, image_cluster[i],  this->image->pixel[i].R, this->image->pixel[i].G, this->image->pixel[i].B);
        #endif
    }                                                            
    std::string output = OUTPUT_PATH; 
    this->image->saveImage(output + "/" + filename);
    return loss / n;
}

int Kmeans::writeToFileTXT(const std::string &filename) {
    int size = 0; 
    // Save the centroids into a file
    std::string path = OUTPUT_PATH; 
    path = path + "/" + filename;
    std::ofstream outfile(path, std::ios::out);
    outfile << (int)this->getK() << " "; size += 1; 
    outfile << (int)this->image->getHeight() << " "; size += 4; 
    outfile << (int)this->image->getWidth() << " "; size += 4; 
    for (int i = 0; i < this->k; i++) {
        outfile << (int)this->centroid[i].R << " "; 
        outfile << (int)this->centroid[i].G << " ";
        outfile << (int)this->centroid[i].B << " ";
        size += 3; 
    }

    int n = this->image->getWidth() * this->image->getHeight();
    for (int i = 0; i < n; i++) {
        outfile << (int)this->image_cluster[i] << " ";
        size += 1; 
    }
    outfile.close();
    return size; 
}