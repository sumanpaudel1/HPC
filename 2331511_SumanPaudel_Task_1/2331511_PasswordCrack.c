/**
  This program is designed to crack an encrypted password by trying all possible combinations of uppercase letters 
  and digits for a 5-character password. The password is tested using multiple threads, which help speed up the process.
  It prints all the combinations tested and once the correct password is found, it stops and displays the result.
  
  This code uses standard C libraries for input/output, string manipulation, threading, and cryptography.
  We use pthread for multithreading and crypt for password encryption.
  
  To run this code  : 
  gcc -o password_cracker password_cracker.c -lpthread -lcrypt
  ./password_cracker
  This code is Written by Suman Paudel  in 2024/12/18  # Thank You #
 */

#include <stdio.h>     // Library for input and output operations
#include <stdlib.h>    // Library for general utilities like memory allocation
#include <string.h>    // Library for string manipulation functions
#include <pthread.h>   // Library to use POSIX threads for multithreading
#include <crypt.h>     // Library for cryptographic hash function
#include <unistd.h>    // Library for POSIX functions
#include <time.h>      // Library for measuring time

// Define constants
#define SALT_LENGTH 7      // Length of the salt used for encryption
#define PASSWORD_LENGTH 5  // Length of the password (5 characters)
#define NUM_LETTERS 26     // Number of uppercase letters (A-Z)
#define NUM_DIGITS 10      // Number of digits (0-9)

// Global variables
int password_found = 0;               // Flag to check if the password is found
char discovered_password[PASSWORD_LENGTH];   // Variable to store the discovered password
char final_password[PASSWORD_LENGTH];       // Store the final found password separately
pthread_mutex_t mutex;                 // Mutex for thread safety
pthread_cond_t cond;                   // Condition variable for threads to synchronize

// Struct for passing thread arguments
typedef struct {
    char *encrypted_password;  // The encrypted password to match
    char salt[SALT_LENGTH];    // The salt used for encryption
    int start_index;           // Starting index for the current thread's range
    int end_index;             // Ending index for the current thread's range
    int thread_id;             // ID to identify the thread
} ThreadArgs;

// Function prototypes
void *crack_thread(void *args);   // Function that runs the cracking algorithm in threads
void substr(char *dest, const char *src, int start, int length);  // Function to extract substring

/**
 * Main function: This is where the program starts.
 */
int main() {
    char encrypted_password[128];  // To store the encrypted password input by the user
    int thread_count;             // To store the number of threads the user wants to use

    // Input encrypted password from the user
    printf("Enter the encrypted password: ");
    if (scanf("%127s", encrypted_password) != 1) {  // Safe reading of the password
        fprintf(stderr, "Error reading encrypted password.\n");
        return 1;
    }

    // Input number of threads to use for cracking the password
    printf("Enter the number of threads to use (1-128): ");
    if (scanf("%d", &thread_count) != 1 || thread_count < 1 || thread_count > 128) {
        fprintf(stderr, "Invalid number of threads. Please enter a value between 1 and 128.\n");
        return 1;
    }

    // Extract salt from the encrypted password
    char salt[SALT_LENGTH];
    substr(salt, encrypted_password, 0, SALT_LENGTH - 1);  // Copy the first 7 characters as salt
    printf("Debug: Extracted salt: %s\n", salt);  // Print the salt for debugging

    // Initialize threads, thread arguments, mutex, and condition variable
    pthread_t threads[thread_count];
    ThreadArgs thread_args[thread_count];
    pthread_mutex_init(&mutex, NULL);  // Initialize the mutex
    pthread_cond_init(&cond, NULL);    // Initialize the condition variable

    // Record start time to measure execution time
    clock_t start_time = clock();

    // Calculate the workload distribution across threads
    int range_per_thread = NUM_LETTERS / thread_count;  // How many letters each thread will process
    int remainder = NUM_LETTERS % thread_count;  // Remainder letters to distribute among threads

    // Assign work to threads
    int current_index = 0;
    for (int i = 0; i < thread_count; i++) {
        thread_args[i].encrypted_password = encrypted_password;  // Pass encrypted password to each thread
        strcpy(thread_args[i].salt, salt);  // Pass salt to each thread

        // Assign range to threads
        thread_args[i].start_index = current_index;
        thread_args[i].end_index = current_index + range_per_thread - 1;

        // Distribute the remainder to the first few threads
        if (i < remainder) {
            thread_args[i].end_index++;
        }

        current_index = thread_args[i].end_index + 1;  // Move to the next index for the next thread

        thread_args[i].thread_id = i;  // Assign thread ID

        // Create the thread to run the cracking algorithm
        if (pthread_create(&threads[i], NULL, crack_thread, &thread_args[i]) != 0) {
            perror("Error creating thread");
            return 1;
        }
    }

    // Join threads (wait for all threads to finish)
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up mutex and condition variable
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    // Record end time
    clock_t end_time = clock();

    // Print all combinations tested and the password result
    printf("\nAll combinations tested.\n");

    if (password_found) {
        printf("Password found: %s\n", final_password);
    } else {
        printf("Password not found.\n");
    }

    // Calculate and display the time taken to execute the program
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken to execute the program: %.4f seconds\n", time_taken);

    return 0;
}

/**
 * crack_thread: This function runs the password cracking algorithm in each thread.
 * It tries all combinations of the first character being uppercase letters (A-Z)
 * and the next four being digits (0-9). The thread will test its assigned range
 * and stop if it finds the correct password.
 */
void *crack_thread(void *args) {
    ThreadArgs *thread_args = (ThreadArgs *)args;  // Get the thread arguments
    char plain[PASSWORD_LENGTH];  // String to store the plain password
    plain[PASSWORD_LENGTH - 1] = '\0';  // Null-terminate the password string

    // Iterate through all combinations in the assigned range
    for (int i = thread_args->start_index; i <= thread_args->end_index && !password_found; i++) {
        plain[0] = 'A' + i;  // First character is an uppercase letter
        for (int j = 0; j < NUM_LETTERS && !password_found; j++) {
            plain[1] = 'A' + j;  // Second character is an uppercase letter
            for (int k = 0; k < NUM_DIGITS && !password_found; k++) {
                plain[2] = '0' + k;  // Third character is a digit
                for (int l = 0; l < NUM_DIGITS && !password_found; l++) {
                    plain[3] = '0' + l;  // Fourth character is a digit

                    // Print all combinations being tested by the thread
                    pthread_mutex_lock(&mutex);
                    printf("Thread %d: Testing password: %s\n", thread_args->thread_id, plain);
                    pthread_mutex_unlock(&mutex);

                    // Encrypt the password using the crypt function with the salt
                    pthread_mutex_lock(&mutex);  // Lock to ensure thread safety
                    char *encrypted = crypt(plain, thread_args->salt);
                    pthread_mutex_unlock(&mutex);

                    // Check if the encrypted password matches the target encrypted password
                    pthread_mutex_lock(&mutex);
                    if (!password_found && strcmp(encrypted, thread_args->encrypted_password) == 0) {
                        password_found = 1;  // Set flag if password is found
                        strcpy(final_password, plain);  // Store the found password
                        printf("Thread %d: Found password: %s\n", thread_args->thread_id, final_password);
                    }
                    pthread_mutex_unlock(&mutex);
                }
            }
        }
    }

    // Print debug message when the thread finishes its range
    pthread_mutex_lock(&mutex);
    printf("Thread %d: Finished testing range [%d - %d]\n", thread_args->thread_id, thread_args->start_index, thread_args->end_index);
    pthread_mutex_unlock(&mutex);

    return NULL;
}

/**
 * substr: This function extracts a substring from the source string starting
 * from a specified index and for a specified length.
 */
void substr(char *dest, const char *src, int start, int length) {
    memcpy(dest, src + start, length);  // Copy part of the source string
    dest[length] = '\0';  // Null-terminate the destination string
}
