/* 
 * Name: Simple cuRand based random number generator
 * File: simpleRandomNumbers.cu 
 * Description: This file contains a simple CUDA kernel to generate
 *              a matrix of distinct random numbers
 * Author: kmmankad (kmmankad@gmail.com kmankad@ncsu.edu)
 * License: MIT License
 *
 */
#include <stdio.h>

// Pull in the curand headers
#include <curand.h>
#include <curand_kernel.h>

// We'll use the time as seed
#include <ctime>

// The all-important CUDA error
// checking macros
#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
		__FILE__,__LINE__); exit(-1);} 
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
		__FILE__,__LINE__-1); exit(-1);} 
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK() 
#endif

// X, Y dimensions of the output matrix
#ifndef NumOfRand_X
#define NumOfRand_X 32
#endif

#ifndef NumOfRand_Y
#define NumOfRand_Y 32
#endif

#define NumOfRand (NumOfRand_X * NumOfRand_Y)

// Block Size
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 32

// CUDA Kernel to initialize the random generator 'states'
__global__ void InitRandGen (int RandSeed, curandState_t* RandStates){
	int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_num = thread_x * NUM_THREADS_X + thread_y;
	if (thread_num < NumOfRand) {
		// Initialization is much faster if sequence number and offset
		// are kept at zero, and instead a different seed is used.
		// See - https://devtalk.nvidia.com/default/topic/480586/curand-initialization-time/?offset=4
		curand_init(RandSeed+thread_num, /* sequence number */ 0, /* sequence offset */ 0, &RandStates[thread_num]);
	}
}

__global__ void RandGen (int* GPUNums, curandState_t* RandStates){
	int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_num = thread_x * NUM_THREADS_X + thread_y;
	if (thread_num < NumOfRand){
		GPUNums[thread_num] = curand(&RandStates[thread_num]) % 100;
	}
}


int main(){
	// Allocate memory for the array of
	// random numbers that we want
	int CPUNums[NumOfRand];
	int* GPUNums;

	// Define a pointer for the cuRandStates
	curandState_t* RandStates;

	// Allocate the memory for the output nums
	CUDA_CALL(cudaMalloc((void**) &GPUNums, sizeof(int) * NumOfRand));

	// Allocate memory for the different curandStates on each core
	CUDA_CALL(cudaMalloc((void**) &RandStates, sizeof(curandState) * NumOfRand));

	// Launch params
	dim3 BlockSize (NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 GridSize((NumOfRand_X/NUM_THREADS_X)+1, (NumOfRand_Y/NUM_THREADS_Y)+1, 1);

	// Launch the Initialization kernel
	InitRandGen<<<GridSize,BlockSize>>>(10, RandStates);
	CUDA_CHECK();
	CUDA_CALL( cudaDeviceSynchronize() );

	// Launch the actual generator kernel
	RandGen<<<GridSize, BlockSize>>> (GPUNums, RandStates);
	CUDA_CHECK();
	CUDA_CALL( cudaDeviceSynchronize());

	// Get the results back to the host mem
	CUDA_CALL(cudaMemcpy(CPUNums, GPUNums, NumOfRand*sizeof(int), cudaMemcpyDeviceToHost));

	// Just print some for examination
	for (int i=0; i<40; i++){
		printf ("%0d ", CPUNums[i]);
		if(i%10 == 9)  {
			printf(" \n");
		}
	}    
	return 0;
}

