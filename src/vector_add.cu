// This CUDA program implements vector addition on both the CPU & GPU

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Function declarations
float *CPU_add_vectors(float *A, float *B, int N);
float *GPU_add_vectors(float *A, float *B, int N);
float *get_random_vector(int N);
long long start_timer();
long long stop_timer(long long start_time, const char *name);
void die(const char *message);
void check_error(cudaError e);

// The number of blocks and threads per blocks in the GPU kernel. If we define them as constant as being 
// done here, then we can use its value in the kernel, for example to statically declare an array in 
// shared memory. Note that to determine the best block count and threads per block for a particular GPU 
// you should check its hardware specification. You can loose performance substantially due to a wrong 
// choice for these parameters. 
const int BLOCK_COUNT = 14;
const int THREADS_PER_BLOCK = 256;

int main(int argc, char **argv) {

	// Seed the random generator (use a constant here for repeatable results)
	srand(5);

	// Determine the vector length
	int N = 100000;  // default value
	if (argc > 1) N = atoi(argv[1]); // user-specified value

	// Generate two random vectors
	long long vector_start_time = start_timer();
	float *A = get_random_vector(N);
	float *B = get_random_vector(N);
	stop_timer(vector_start_time, "Vector generation");
	
	// Compute their sum on the CPU
	long long CPU_start_time = start_timer();
	float *C_CPU = CPU_add_vectors(A, B, N);
	long long CPU_time = stop_timer(CPU_start_time, "\nCPU");
	
	// Compute their sum on the GPU
	long long GPU_start_time = start_timer();
	float *C_GPU = GPU_add_vectors(A, B, N);
	long long GPU_time = stop_timer(GPU_start_time, "\tTotal");
	
	// Compute the speedup or slowdown
	if (GPU_time > CPU_time) {
		printf("\nCPU outperformed GPU by %.2fx\n", (float) GPU_time / (float) CPU_time);
	} else { 
		printf("\nGPU outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);
	}
	
	// Check the correctness of the GPU results
	int num_wrong = 0;
	for (int i = 0; i < N; i++) {
		if (fabs(C_CPU[i] - C_GPU[i]) > 0.0001) {
			printf("Values differs at index %d CPU:%f\tGPU:%f\n", i, C_CPU[i], C_GPU[i]);
			num_wrong++;
		}
	}
	
	// Report the correctness results
	if (num_wrong) {
		printf("\n%d / %d values incorrect\n", num_wrong, N);
	} else {          
		printf("\nAll values correct\n");
	}
}


// A GPU kernel that computes the vector sum A + B
__global__ void add_vectors_kernel(float *A, float *B, float *C, int N) {

	// determine the index of the thread among all GPU threads	
	int blockId = blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	int threadCount = gridDim.x * blockDim.x; 

	// calculate the vector sum for the indexes of vector the current thread is responsible for
	for (int i = threadId; i < N; i += threadCount) {	
		C[i] = A[i] + B[i];
	}
}


// Returns the vector sum A + B (computed on the GPU)
float *GPU_add_vectors(float *A_CPU, float *B_CPU, int N) {
	
	long long memory_start_time = start_timer();

	// Allocate GPU memory for the inputs and the result
	int vector_size = N * sizeof(float);
	float *A_GPU, *B_GPU, *C_GPU;
	check_error(cudaMalloc((void **) &A_GPU, vector_size));
	check_error(cudaMalloc((void **) &B_GPU, vector_size));
	check_error(cudaMalloc((void **) &C_GPU, vector_size));
	
	// Transfer the input vectors to GPU memory
	check_error(cudaMemcpy(A_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice));
	check_error(cudaMemcpy(B_GPU, B_CPU, vector_size, cudaMemcpyHostToDevice));
	
	stop_timer(memory_start_time, "\nGPU:\tTransfer to GPU");
	
	// Execute the kernel to compute the vector sum on the GPU
	long long kernel_start_time = start_timer();

	// Note that we are using a one dimensional grid in this calculation as that is ideal for this
	// particular problem. For some other problem, a 2D or even a 3D grid may be appropriate. The
	// dimensionality of the grid is supposed to help you decompose the algorithmic logic inside the
	// GPU kernel. In particular, how you decide what thread should do what instruction. It does not 
	// affect the performance of the kernel.
	add_vectors_kernel <<<BLOCK_COUNT, THREADS_PER_BLOCK>>> (A_GPU, B_GPU, C_GPU, N);
	
	// make the CPU main thread waite for the GPU kernel call to complete
	cudaThreadSynchronize();  // This is only needed for timing and error-checking purposes
	stop_timer(kernel_start_time, "\tKernel execution");
	
	// Check for kernel errors
	check_error(cudaGetLastError());
	
	// Allocate CPU memory for the result
	float *C_CPU = (float *) malloc(vector_size);
	if (C_CPU == NULL) die("Error allocating CPU memory");
	
	// Transfer the result from the GPU to the CPU
	memory_start_time = start_timer();
	check_error(cudaMemcpy(C_CPU, C_GPU, vector_size, cudaMemcpyDeviceToHost));
	stop_timer(memory_start_time, "\tTransfer from GPU");
	
	// Free the GPU memory
	check_error(cudaFree(A_GPU));
	check_error(cudaFree(B_GPU));
	check_error(cudaFree(C_GPU));
	
	return C_CPU;
}


// Returns the vector sum A + B
float *CPU_add_vectors(float *A, float *B, int N) {	
	
	// Allocate memory for the result
	float *C = (float *) malloc(N * sizeof(float));
	if (C == NULL) die("Error allocating CPU memory");

	// Compute the sum;
	for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
	
	// Return the result
	return C;
}


// Returns a randomized vector containing N elements
float *get_random_vector(int N) {
	if (N < 1) die("Number of elements must be greater than zero");
	
	// Allocate memory for the vector
	float *V = (float *) malloc(N * sizeof(float));
	if (V == NULL) die("Error allocating CPU memory");
	
	// Populate the vector with random numbers
	for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();
	
	// Return the randomized vector
	return V;
}


// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *label) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", label, ((float) (end_time - start_time)) / (1000 * 1000));
	return end_time - start_time;
}


// Prints the specified message and quits
void die(const char *message) {
	printf("%s\n", message);
	exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
	if (e != cudaSuccess) {
		printf("\nCUDA error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
}
