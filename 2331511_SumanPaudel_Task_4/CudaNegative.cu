// Include necessary standard input/output and memory allocation libraries
#include <stdio.h>
#include <stdlib.h>
// Include the PNG image handling library
#include "lodepng.h"

/*
 * Overall Program Workflow:
 * 1. Program reads PNG image using lodepng
 * 2. Allocates memory on both CPU and GPU
 * 3. Transfers image data to GPU
 * 4. Applies 3x3 box blur using CUDA kernel
 * 5. Transfers processed image back to CPU
 * 6. Saves the blurred image as PNG
 */

// Define a macro to make CUDA error checking easier to read and maintain
// This will print the error message and return -1 if any CUDA operation fails
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        return -1; \
    }

/*
 * Function: getPixelIndex
 * Purpose: Converts 2D coordinates to 1D array index for RGBA image
 * Parameters:
 *   - x: x-coordinate of pixel
 *   - y: y-coordinate of pixel
 *   - width: image width
 * Returns: Index in 1D array where pixel RGBA values start
 * Note: Each pixel uses 4 array elements (R,G,B,A)
 */
// Helper function that runs on the GPU to calculate the position of a pixel in memory
// Since images are stored as 1D arrays, this converts 2D coordinates (x,y) to the correct position
__device__ int getPixelIndex(int x, int y, int width) {
    return (y * width + x) * 4;  // Multiply by 4 because each pixel has RGBA values
}

/*
 * CUDA Kernel: applyBoxBlur
 * Purpose: Applies 3x3 box blur to input image
 * Workflow:
 * 1. Each thread processes one pixel
 * 2. Gets neighboring pixel values (3x3 matrix)
 * 3. Calculates average RGB values
 * 4. Writes blurred values to output
 * 5. Preserves original alpha channel
 */
// The main GPU kernel that performs the blur operation
__global__ void applyBoxBlur(unsigned char* output, const unsigned char* input, 
                            int width, int height) {
    // Calculate which pixel this thread should process based on thread and block IDs
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process if this thread maps to a valid pixel in the image
    if (x < width && y < height) {
        // Initialize running totals for RGB channels
        int sumR = 0, sumG = 0, sumB = 0;
        int count = 0;
        
        // Define the 3x3 grid of neighboring pixels to look at
        int neighbors[9][2] = {
            {-1,-1}, {0,-1}, {1,-1},  // Top row
            {-1, 0}, {0, 0}, {1, 0},  // Middle row
            {-1, 1}, {0, 1}, {1, 1}   // Bottom row
        };
        
        // Look at each neighbor pixel
        for (int i = 0; i < 9; i++) {
            int newX = x + neighbors[i][0];
            int newY = y + neighbors[i][1];
            
            // Check if the neighbor pixel is actually in the image
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                // Get the position of this neighbor in memory
                int idx = getPixelIndex(newX, newY, width);
                // Add this neighbor's RGB values to our running totals
                sumR += input[idx];
                sumG += input[idx + 1];
                sumB += input[idx + 2];
                count++;
            }
        }
        
        // Calculate where to write our output pixel
        int outputIdx = getPixelIndex(x, y, width);
        
        // Write the average values for RGB channels
        output[outputIdx] = sumR / count;     // Average Red
        output[outputIdx + 1] = sumG / count; // Average Green
        output[outputIdx + 2] = sumB / count; // Average Blue
        output[outputIdx + 3] = input[outputIdx + 3]; // Keep alpha unchanged
    }
}

int main(int argc, char **argv) {
    // Variables for image handling
    unsigned int error;
    unsigned char* image;
    unsigned int width, height;
    const char* filename = "hck.png";
    const char* newFileName = "Output.png";

    // Load the PNG image into memory
    error = lodepng_decode32_file(&image, &width, &height, filename);
    if(error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    // Calculate the size needed for the image data
    const int ARRAY_SIZE = width * height * 4;  // 4 channels (RGBA) per pixel
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

    // Allocate memory on the CPU for the output image
    unsigned char* host_output = (unsigned char*)malloc(ARRAY_BYTES);

    // Allocate memory on the GPU for input and output images
    unsigned char *d_in, *d_out;
    cudaError_t cudaStatus;
    
    // Allocate GPU memory for input image
    cudaStatus = cudaMalloc((void**)&d_in, ARRAY_BYTES);
    CHECK_CUDA_ERROR(cudaStatus);
    
    // Allocate GPU memory for output image
    cudaStatus = cudaMalloc((void**)&d_out, ARRAY_BYTES);
    CHECK_CUDA_ERROR(cudaStatus);

    // Copy the input image to the GPU
    cudaStatus = cudaMemcpy(d_in, image, ARRAY_BYTES, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(cudaStatus);

    // Set up the GPU processing grid
    dim3 threadsPerBlock(16, 16);  // Each block will be 16x16 threads
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Run the blur operation on the GPU
    applyBoxBlur<<<numBlocks, threadsPerBlock>>>(d_out, d_in, width, height);
    
    // Check if the kernel launch was successful
    cudaStatus = cudaGetLastError();
    CHECK_CUDA_ERROR(cudaStatus);

    // Copy the processed image back from the GPU
    cudaStatus = cudaMemcpy(host_output, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaStatus);

    // Save the processed image to a new PNG file
    error = lodepng_encode32_file(newFileName, host_output, width, height);
    if(error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }

    // Free all allocated memory (both CPU and GPU)
    free(image);
    free(host_output);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
