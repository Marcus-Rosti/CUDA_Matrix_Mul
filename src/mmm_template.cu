#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<iostream>

using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------

typedef struct {
	int width;
	int height;	
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory  
float *A, *B, *C, *C_CPU;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error); 
void check_error(cudaError e);
__global__ void kernel(float * A_GPU, float * B_GPU, float * C_GPU, ArrayMetadata2D A_gpu_md, ArrayMetadata2D B_gpu_md);
//----------------------------------- CUDA function definitions -----------------------------------------

#define BLOCK_SIZE 10

//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
	
	A_MD.width = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.height = (argc > 2) ? atoi(argv[2]) : A_MD.width;
	B_MD.width = (argc > 3) ? atoi(argv[3]) : A_MD.height;
	B_MD.height = (argc > 4) ? atoi(argv[4]) : B_MD.width;
	C_MD.width = A_MD.width;
	C_MD.height = B_MD.height;

	printf("Matrix A is %d-by-%d\n", A_MD.width, A_MD.height);
	printf("Matrix B is %d-by-%d\n", B_MD.width, B_MD.height);
	printf("Matrix C is %d-by-%d\n", C_MD.width, C_MD.height);

	allocateAndInitializeAB();

	// matrix matrix multiplication in the CPU
	clock_t start = clock();	
	computeCpuMMM();
	clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        printf("Computation time in the CPU: %f seconds\n", elapsed);
	

	copyMatricesToGPU();
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	// width of b / BS, height of A / BS
	dim3 dimGrid(B_MD.width/ BLOCK_SIZE, A_MD.height / BLOCK_SIZE);
	kernel <<< dimGrid, dimBlock >>>  (A_GPU, B_GPU, C_GPU, A_MD, B_MD);
			// num blocks, num threads/block 
	return 0;
}

// allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {
	
	size_t sizeofA = A_MD.width * A_MD.height * sizeof(float);
	A = (float*) malloc(sizeofA);
	
	srand(time(NULL));
  	for (int i = 0; i < A_MD.width; i++) {
		for (int j = 0; j < A_MD.height; j++) {
			int index = i * A_MD.height + j;
			A[index] = (rand() % 1000) * 0.001; 
		}
	}
	
	size_t sizeofB = B_MD.width * B_MD.height * sizeof(float);
	B = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.width; i++) {
		for (int j = 0; j < B_MD.height; j++) {
			int index = i * B_MD.height + j;
			B[index] = (rand() % 1000) * 0.001; 
		}
	}
}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	
	size_t sizeofA = A_MD.width * A_MD.height * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));
	
	size_t sizeofB = B_MD.width * B_MD.height * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));
	
	size_t sizeofC = C_MD.width * C_MD.height * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	size_t sizeofC = C_MD.width * C_MD.height * sizeof(float);
	C_CPU = (float*) malloc(sizeofC);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {
	
	// allocate the result matrix for the CPU computation
	size_t sizeofC = C_MD.width * C_MD.height * sizeof(float);
	C = (float*) malloc(sizeofC);
	
	// compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.width; i++) {
		int a_i = i * A_MD.height;
		int c_i = i * C_MD.height;
		for (int j = 0; j < B_MD.height; j++) {
			int c_index = c_i + j;
			C[c_index] = 0;
			for (int k = 0; k < B_MD.width; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.height + j;
				C[c_index] += A[a_index] * B[b_index];
			}
		}
	}
}

__global__ void kernel(float * A_GPU, float * B_GPU, float * C_GPU, ArrayMetadata2D A_gpu_md, ArrayMetadata2D B_gpu_md) {
	////////////////////////////////////
	// Marcus's idea of how it should work
	const int blockY = blockIdx.y; // the global block indexes
	const int blockX = blockIdx.x;	
	
	// Get the reference to C starting at the row and column
	// Essentially this is the whole block
	// I've probably fixed up the index
	float * C_block = &C_GPU[blockY * BLOCK_SIZE + blockX * BLOCK_SIZE];
	
	const int sub_row = threadIdx.y; // valued from 0:blocksize-1
	const int sub_col = threadIdx.x; // valued from 0:blocksize-1
	
	// Th value we're going to shove into the final array
	volatile float my_final_value = 0.0f;
	
	// Let's loop over each block!
	for (int i = 0; i < A_gpu_md.height / BLOCK_SIZE; i++) {
		// Get the sub block
		float * A_block = &A_GPU[i*BLOCK_SIZE];
		float * B_block = &B_GPU[i*BLOCK_SIZE];
	
		// Here's all the shared memory we'll need
		__shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

		// Fill in that shared array with my column
		sharedA[sub_row][sub_col] = A_block[i*BLOCK_SIZE];
		sharedB[sub_row][sub_col] = B_block[i*BLOCK_SIZE];

		// Sum up all the elements that go from 0:BLOCKSIZE
		// So the row of A and the column of B for 0 to BLOCKSIZE
		for (int j = 0; j < BLOCK_SIZE; j++) my_final_value += sharedA[sub_row][j] * sharedB[j][sub_col];
	}
	
	C_block[sub_row * B_gpu_md.width + sub_col] = my_final_value;
	//
	////////////////////////////////////////////////	
	int srow = 0;
	int scol = 0;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	const int sizeOfWork = 10;
	const int sizeOfBlock = 100;
	// Where to start in the GPU matrix
	int mIndex = threadId * sizeOfWork;

	// copy the submatrix into shared memory
	__shared__ float blockA[10][10]; 
	__shared__ float blockB[10][10]; 
	int blockIndex = mIndex;
	for (int i = 0; i < sizeOfBlock; i++) {
		blockA[srow][scol] = A_GPU[blockIndex];
		blockB[srow][scol] = B_GPU[blockIndex];
		// Jump a row when finished copying column
		if (i == sizeOfWork) {
			srow++;
			blockIndex *= sizeOfWork;
		}
		scol++;
	}

	// Compute a partial row of C
	int aRow = threadId;
	int cIndex = mIndex;
	// TODO: Transpose B for better load times
	// Will need to switch order to keep coalesced 

	// Multiply a row of A 
	for (int aCol = 0; aCol < sizeOfWork; aCol++) {
		// with each column of B
		for (int bCol = 0; bCol < sizeOfWork; bCol++) {
			float cell = 1;
			for (int bRow = 0; bRow < sizeOfWork; bRow++) {
				cell += (blockA[aRow][aCol] * blockB[bRow][bCol]);
			}
			// Store the result in C
			C_GPU[cIndex] = cell;
			cIndex++;
		}
	}



}



// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
// from the CPU
void compareHostAndGpuOutput() {
	int totalElements = C_MD.width * C_MD.height;
	int missmatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C[i] - C_CPU[i]) > 0.01) {
			missmatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
	}
	if (missmatchCount > 0) {
		printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
	} else {
		printf("Computation is correct: CPU and GPU outputs match\n");
	}
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

