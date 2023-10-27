/*
 * kernel.h
 *
 *  Created on: 24.10.2010
 *      Author: ahlers
 */

#ifndef KERNEL_H_
#define KERNEL_H_

/*
 * type definitions
 */

typedef float* float_pt;
typedef struct cudaGraphicsResource* cudaGraphicsRes_pt;

/*
 * function prototypes
 */

// initialize CUDA
// (using CUDA device 0)
void initCUDA();

// check for CUDA errors, exit in case of errors
void checkCUDAError(const char* functionName);

// print CUDA version and device information
void printCUDAVersion();

// launch N-body dynamics kernel
void launchCudaKernel(cudaGraphicsRes_pt& positionCudaVBO, float_pt velocityCudaPtr,
    size_t nParticles, float maxPosition);

#endif /* KERNEL_H_ */
