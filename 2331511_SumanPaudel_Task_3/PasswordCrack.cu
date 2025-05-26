/*
 * Password Encryption and Decryption using CUDA
 * 
 * Description:
 * This program demonstrates password encryption and decryption using CUDA parallel processing.
 * It takes a 4-character password from the user (2 letters + 2 numbers), encrypts it,
 * and then uses GPU parallel processing to crack the encrypted password.
 * 
 *  * Program Workflow:
 * 1. Gets 4-character password input (2 letters + 2 numbers)
 * 2. Encrypts password on CPU for verification
 * 3. Uses GPU parallel processing to crack the encrypted password
 * 4. Returns decrypted password to CPU
 * 
 * How to compile: nvcc PasswordCrack.cu -o PasswordCrack
 * How to run: ./PasswordCrack
 * 
 * Input format: LLNN (L=letter A-Z, N=number 0-9)
 * Example input: AZ12
 */




#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <ctype.h>
#include <string.h>

// Define constants for array sizes
#define ALPHABET_SIZE 52  // Include both upper and lowercase letters
#define NUMBER_SIZE 10
#define PASSWORD_LENGTH 4
#define ENCRYPTED_LENGTH 11


/*
 * Function: getPasswordInput
 * Purpose: Validates and gets password input from user
 * Workflow:
 * 1. Prompts user for password
 * 2. Validates length is exactly 4 characters
 * 3. Checks first 2 chars are letters and last 2 are numbers
 * 4. Repeats until valid input received
 */

void getPasswordInput(char* password) {
    while (1) {
        printf("Enter 4-character password (2 letters followed by 2 numbers, e.g., AZ12): ");
        scanf("%4s", password);
        
        // Validate input length
        if (strlen(password) != 4) {
            printf("Error: Password must be exactly 4 characters\n");
            continue;
        }
        
        // Validate format (2 letters + 2 numbers)
        if (!(isalpha(password[0]) && isalpha(password[1]) && 
              isdigit(password[2]) && isdigit(password[3]))) {
            printf("Error: Format must be 2 letters followed by 2 numbers\n");
            continue;
        }
        
        break;
    }
}

// Device-side character checking functions
__device__ bool isUpperCase(char c) {
    return (c >= 'A' && c <= 'Z');
}

__device__ bool isLowerCase(char c) {
    return (c >= 'a' && c <= 'z');
}

/*
 * Function: CudaCrypt (Device-side)
 * Purpose: Encrypts raw password using provided algorithm
 * Workflow:
 * 1. Applies encryption rules to each character
 * 2. Handles character wrapping for letters (A-Z, a-z)
 * 3. Handles wrapping for numbers (0-9)
 * 4. Produces 10-char encrypted string + null terminator
 */

__device__ void CudaCrypt(const char* raw, char* encrypted) {
    // Apply encryption rules
    encrypted[0] = raw[0] + 2;
    encrypted[1] = raw[0] - 2;
    encrypted[2] = raw[0] + 1;
    encrypted[3] = raw[1] + 3;
    encrypted[4] = raw[1] - 3;
    encrypted[5] = raw[1] - 1;
    encrypted[6] = raw[2] + 2;
    encrypted[7] = raw[2] - 2;
    encrypted[8] = raw[3] + 4;
    encrypted[9] = raw[3] - 4;
    encrypted[10] = '\0';

    // Handle wrapping for letters and numbers
    for (int i = 0; i < 6; i++) { // First 6 are letters
        char c = raw[i/3]; // Get original character
        if (isUpperCase(c)) {
            if (encrypted[i] > 'Z') encrypted[i] = 'A' + (encrypted[i] - 'Z' - 1);
            if (encrypted[i] < 'A') encrypted[i] = 'Z' - ('A' - encrypted[i] - 1);
        } else {
            if (encrypted[i] > 'z') encrypted[i] = 'a' + (encrypted[i] - 'z' - 1);
            if (encrypted[i] < 'a') encrypted[i] = 'z' - ('a' - encrypted[i] - 1);
        }
    }
    
    for (int i = 6; i < 10; i++) { // Last 4 are numbers
        if (encrypted[i] > '9') encrypted[i] = '0' + (encrypted[i] - '9' - 1);
        if (encrypted[i] < '0') encrypted[i] = '9' - ('0' - encrypted[i] - 1);
    }
}

/*
 * Function: crack (Kernel)
 * Purpose: GPU parallel password cracking
 * Workflow:
 * 1. Each thread tries different character combinations
 * 2. Uses block indices for letters (x,y)
 * 3. Uses thread indices for numbers (x,y)
 * 4. Compares generated encrypted password with target
 * 5. Stores found password in result
 */

__global__ void crack(const char* alphabet, const char* numbers, 
                     const char* target, char* result, bool* found) {
    // Add support for both upper and lowercase in alphabet indexing
    int letterIndex = blockIdx.x;
    char firstLetter = alphabet[letterIndex];
    int secondLetterIdx = blockIdx.y;
    char secondLetter = alphabet[secondLetterIdx];
    
    char generatedPassword[4] = {
        firstLetter,
        secondLetter,
        numbers[threadIdx.x],
        numbers[threadIdx.y]
    };
    
    char encrypted[11];
    CudaCrypt(generatedPassword, encrypted);
    
    // Compare with target
    bool match = true;
    for (int i = 0; i < 10 && match; i++) {
        if (encrypted[i] != target[i]) match = false;
    }
    
    if (match && !*found) {
        atomicExch((int*)found, 1);  // Thread-safe update
        for (int i = 0; i < 4; i++) {
            result[i] = generatedPassword[i];
        }
    }
}

// Modified host-side test function
void testEncryptionOnHost(const char* raw, char* encrypted) {
    // Apply encryption rules
    encrypted[0] = raw[0] + 2;
    encrypted[1] = raw[0] - 2;
    encrypted[2] = raw[0] + 1;
    encrypted[3] = raw[1] + 3;
    encrypted[4] = raw[1] - 3;
    encrypted[5] = raw[1] - 1;
    encrypted[6] = raw[2] + 2;
    encrypted[7] = raw[2] - 2;
    encrypted[8] = raw[3] + 4;
    encrypted[9] = raw[3] - 4;
    encrypted[10] = '\0';

    // Handle wrapping for letters and numbers with proper case handling
    for (int i = 0; i < 6; i++) {
        char c = raw[i/3];
        if (isupper(c)) {
            if (encrypted[i] > 'Z') encrypted[i] = 'A' + (encrypted[i] - 'Z' - 1);
            if (encrypted[i] < 'A') encrypted[i] = 'Z' - ('A' - encrypted[i] - 1);
        } else {
            if (encrypted[i] > 'z') encrypted[i] = 'a' + (encrypted[i] - 'z' - 1);
            if (encrypted[i] < 'a') encrypted[i] = 'z' - ('a' - encrypted[i] - 1);
        }
    }
    
    // Handle numbers
    for (int i = 6; i < 10; i++) {
        if (encrypted[i] > '9') encrypted[i] = '0' + (encrypted[i] - '9' - 1);
        if (encrypted[i] < '0') encrypted[i] = '9' - ('0' - encrypted[i] - 1);
    }
    printf("CPU-side encryption complete\n");
}

int main() {
    // Initialize CPU arrays
    char cpuAlphabet[ALPHABET_SIZE];
    for(int i = 0; i < 26; i++) {
        cpuAlphabet[i] = 'A' + i;        // Uppercase letters
        cpuAlphabet[i+26] = 'a' + i;     // Lowercase letters
    }
    char cpuNumbers[NUMBER_SIZE] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    
    // Get password from user
    char rawPassword[PASSWORD_LENGTH + 1];
    getPasswordInput(rawPassword);
    
    // Generate encrypted password on CPU for verification
    char targetPassword[ENCRYPTED_LENGTH];
    testEncryptionOnHost(rawPassword, targetPassword);
    printf("Original password: %s\nEncrypted password: %s\n", rawPassword, targetPassword);
    
    char result[4] = {0};
    bool found = false;

    // Allocate GPU memory
    char* gpuAlphabet;
    char* gpuNumbers;
    char* gpuTargetPassword;
    char* gpuResult;
    bool* gpuFound;
    cudaMalloc((void**)&gpuAlphabet, sizeof(char) * ALPHABET_SIZE);
    cudaMalloc((void**)&gpuNumbers, sizeof(char) * NUMBER_SIZE);
    cudaMalloc((void**)&gpuTargetPassword, sizeof(char) * 11);
    cudaMalloc((void**)&gpuResult, sizeof(char) * 4);
    cudaMalloc((void**)&gpuFound, sizeof(bool));

    // Copy data to GPU
    cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * ALPHABET_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * NUMBER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTargetPassword, targetPassword, sizeof(char) * 11, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuFound, &found, sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel with multi-dimensional configuration
    dim3 blocks(ALPHABET_SIZE, ALPHABET_SIZE, 1);
    dim3 threads(NUMBER_SIZE, NUMBER_SIZE, 1);
    crack<<<blocks, threads>>>(gpuAlphabet, gpuNumbers, gpuTargetPassword, gpuResult, gpuFound);

    // Wait for GPU to complete
    cudaDeviceSynchronize();

    // Add error checking for CUDA operations
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to CPU
    cudaMemcpy(result, gpuResult, sizeof(char) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&found, gpuFound, sizeof(bool), cudaMemcpyDeviceToHost);

    // Print the decrypted password if found
    if (found) {
        printf("Password found! Raw: %c%c%c%c\n", result[0], result[1], result[2], result[3]);
    } else {
        printf("Password not found.\n");
    }

    // Free GPU memory
    cudaFree(gpuAlphabet);
    cudaFree(gpuNumbers);
    cudaFree(gpuTargetPassword);
    cudaFree(gpuResult);
    cudaFree(gpuFound);

    return 0;
}
