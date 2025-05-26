#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    // Check command line arguments
    if (argc != 2) {
        printf("Usage: %s <input_image.png>\n", argv[0]);
        return 1;
    }

    unsigned int error;
    unsigned int encError;
    unsigned char* image = NULL;
    unsigned char* newImage = NULL;
    unsigned int width, height;
    const char* filename = argv[1];
    const char* newFileName = "generated.png";

    // Decode the image
    error = lodepng_decode32_file(&image, &width, &height, filename);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    // Allocate memory for the new image
    newImage = (unsigned char*)malloc(height * width * 4);
    if (newImage == NULL) {
        printf("Failed to allocate memory for output image\n");
        free(image);
        return 1;
    }

    printf("width = %d height = %d\n", width, height);
    // Process the image
    for (int i = 0; i < height * width * 4; i += 4) {
        newImage[i] = 255 - image[i];        // R
        newImage[i + 1] = 255 - image[i + 1];// G
        newImage[i + 2] = 255 - image[i + 2];// B
        newImage[i + 3] = image[i + 3];      // A
    }

    // Encode the new image
    encError = lodepng_encode32_file(newFileName, newImage, width, height);
    if (encError) {
        printf("error %u: %s\n", encError, lodepng_error_text(encError));
    }

    // Cleanup
    free(image);
    free(newImage);

    return encError ? 1 : 0;
}

