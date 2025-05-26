///////////////////////////////////////////////////////////////////////////////
// Program Description:
// This program performs matrix multiplication using multithreading for optimization. 
// It reads pairs of matrices from an input file, computes their product, and writes
// the resulting matrices to an output file named "output.txt". The program dynamically
// allocates memory for matrices and ensures efficient computation by distributing rows
// across multiple threads.
//
// How the Program Works:
// 1. Matrices are read from an input file.
// 2. Threads are created to compute the product of each row of the first matrix with the
//    columns of the second matrix.
// 3. The resulting matrix is written to an output file.
//
// How to Run the Program:
// 1. Compile the code using a C compiler that supports pthreads (e.g., `gcc`):
//    gcc -o matrix_mult matrix_mult.c -pthread
// 2. Run the program with the input file name and number of threads as arguments:
//    ./matrix_mult input.txt 4
//    Here, "input.txt" is the input file containing the matrices, and "4" is the number
//    of threads to use.
// 3. Check the "output.txt" file for the resulting matrices.
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_LINE_LENGTH 1024

// Struct to store thread data for matrix multiplication
typedef struct {
    double **A;        // Pointer to matrix A
    double **B;        // Pointer to matrix B
    double **C;        // Pointer to result matrix C
    int row;           // Current row of matrix A being processed
    int cols;          // Number of columns in matrix B
    int common_dim;    // Common dimension (columns of A / rows of B)
} ThreadData;

// Function: multiply_row
// Description: Multiplies a single row of matrix A with all columns of matrix B,
//              storing the result in matrix C.
void *multiply_row(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int j = 0; j < data->cols; j++) {
        data->C[data->row][j] = 0.0;
        for (int k = 0; k < data->common_dim; k++) {
            data->C[data->row][j] += data->A[data->row][k] * data->B[k][j];
        }
    }
    return NULL;
}

// Function: allocate_matrix
// Description: Dynamically allocates memory for a matrix with the specified
//              number of rows and columns.
double **allocate_matrix(int rows, int cols) {
    double **matrix = malloc(rows * sizeof(double *));
    if (!matrix) {
        fprintf(stderr, "Failed to allocate matrix rows\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
        if (!matrix[i]) {
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            fprintf(stderr, "Failed to allocate matrix columns\n");
            return NULL;
        }
    }
    return matrix;
}

// Function: free_matrix
// Description: Frees the memory allocated for a matrix.
void free_matrix(double **matrix, int rows) {
    if (matrix) {
        for (int i = 0; i < rows; i++) {
            if (matrix[i]) free(matrix[i]);
        }
        free(matrix);
    }
}

// Function: read_matrix
// Description: Reads a matrix from a file, dynamically allocating memory and
//              parsing the data into the allocated matrix.
double **read_matrix(FILE *file, int *rows, int *cols) {
    char line[MAX_LINE_LENGTH];

    // Skip empty lines
    do {
        if (!fgets(line, sizeof(line), file)) {
            return NULL;  // End of file
        }
    } while (line[0] == '\n' || line[0] == '\r');

    // Parse dimensions
    if (sscanf(line, "%d,%d", rows, cols) != 2) {
        fprintf(stderr, "Error: Invalid matrix dimensions\n");
        return NULL;
    }

    if (*rows <= 0 || *cols <= 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions: %dx%d\n", *rows, *cols);
        return NULL;
    }

    double **matrix = allocate_matrix(*rows, *cols);
    if (!matrix) return NULL;

    // Read matrix data
    for (int i = 0; i < *rows; i++) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error: Incomplete matrix data\n");
            free_matrix(matrix, *rows);
            return NULL;
        }

        char *token = strtok(line, ",");
        for (int j = 0; j < *cols; j++) {
            if (!token) {
                fprintf(stderr, "Error: Insufficient values in row %d\n", i + 1);
                free_matrix(matrix, *rows);
                return NULL;
            }
            matrix[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    return matrix;
}

// Function: multiply_matrices
// Description: Multiplies two matrices A and B, using the specified number
//              of threads to perform row-by-row computations.
int multiply_matrices(double **A, int A_rows, int A_cols, 
                      double **B, int B_rows, int B_cols, 
                      double ***C, int num_threads) {
    // Check if matrices can be multiplied
    if (A_cols != B_rows) {
        printf("Cannot multiply matrices of dimensions %dx%d and %dx%d\n", 
               A_rows, A_cols, B_rows, B_cols);
        return 0;  // Return 0 to indicate failure
    }

    // Allocate result matrix
    *C = allocate_matrix(A_rows, B_cols);
    if (!*C) return 0;

    // Create thread data and thread array
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = malloc(num_threads * sizeof(ThreadData));

    if (!threads || !thread_data) {
        free_matrix(*C, A_rows);
        *C = NULL;
        free(threads);
        free(thread_data);
        return 0;
    }

    // Distribute work among threads
    for (int i = 0; i < A_rows; i++) {
        int thread_idx = i % num_threads;
        thread_data[thread_idx] = (ThreadData){
            .A = A,
            .B = B,
            .C = *C,
            .row = i,
            .cols = B_cols,
            .common_dim = A_cols
        };

        pthread_create(&threads[thread_idx], NULL, multiply_row, &thread_data[thread_idx]);

        // Join threads when we've used maximum threads or reached last row
        if (thread_idx == num_threads - 1 || i == A_rows - 1) {
            for (int j = 0; j <= thread_idx; j++) {
                pthread_join(threads[j], NULL);
            }
        }
    }

    free(threads);
    free(thread_data);
    return 1;  // Return 1 to indicate success
}

// Function: write_matrix
// Description: Writes a matrix to a file in a comma-separated format.
void write_matrix(FILE *file, double **matrix, int rows, int cols) {
    fprintf(file, "%d,%d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.2f", matrix[i][j]);
            if (j < cols - 1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");  // Add blank line between matrices
}

// Function: main
// Description: Entry point of the program. Parses arguments, reads matrices
//              from an input file, performs matrix multiplication, and writes
//              the results to an output file.
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_threads = atoi(argv[2]);
    if (num_threads <= 0) {
        fprintf(stderr, "Error: Number of threads must be positive\n");
        return EXIT_FAILURE;
    }

    FILE *input_file = fopen(argv[1], "r");
    if (!input_file) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    FILE *output_file = fopen("output.txt", "w");
    if (!output_file) {
        perror("Error opening output file");
        fclose(input_file);
        return EXIT_FAILURE;
    }

    int pair_count = 0;
    while (!feof(input_file)) {
        int A_rows, A_cols, B_rows, B_cols;
        pair_count++;

        // Try to read first matrix of the pair
        double **A = read_matrix(input_file, &A_rows, &A_cols);
        if (!A) {
            if (feof(input_file)) break;  // Normal end of file
            printf("Skipping invalid matrix A in pair %d\n", pair_count);
            continue;
        }

        // Try to read second matrix of the pair
        double **B = read_matrix(input_file, &B_rows, &B_cols);
        if (!B) {
            if (feof(input_file)) {
                printf("Warning: Odd number of matrices in input file\n");
                free_matrix(A, A_rows);
                break;
            }
            printf("Skipping invalid matrix B in pair %d\n", pair_count);
            free_matrix(A, A_rows);
            continue;
        }

        // Try to multiply matrices
        double **C = NULL;
        int max_dim = A_rows > B_cols ? A_rows : B_cols;
        int actual_threads = num_threads > max_dim ? max_dim : num_threads;

        printf("Processing pair %d: Matrix A(%dx%d) * Matrix B(%dx%d)\n", 
               pair_count, A_rows, A_cols, B_rows, B_cols);

        if (multiply_matrices(A, A_rows, A_cols, B, B_rows, B_cols, &C, actual_threads)) {
            printf("Successfully multiplied pair %d\n", pair_count);
            write_matrix(output_file, C, A_rows, B_cols);
            free_matrix(C, A_rows);
        } else {
            printf("Skipping multiplication of pair %d due to incompatible dimensions\n", pair_count);
        }

        // Clean up matrices
        free_matrix(A, A_rows);
        free_matrix(B, B_rows);
    }

    printf("\n=== Processing Complete ===\n");
    printf("\nResults have been written to: output.txt\n");
    printf("===============================\n\n");

    fclose(input_file);
    fclose(output_file);
    return EXIT_SUCCESS;
}
