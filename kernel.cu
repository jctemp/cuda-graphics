/*
 * kernel.cpp
 *
 *  Created on: 24.10.2010
 *      Author: ahlers
 */

#include "kernel.h"
#include <ctime>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

struct ModelParams {
  GLsizei nParticles;
  float maxPosition;
  float mass;
  float g;
  float epsSqr;
};

const ModelParams params{2048, 10.0f, 1.0f, 0.001f, 0.0001f};

/*
 * global variables
 */

constexpr float EPSILON{1e-5};
constexpr float G{6.6743e-11};
const size_t nBlocks = 16;

/*
 * local function prototypes
 */

// simple kernel for testing
__global__ void setPositionsKernel(float4 *positions, float time,
                                   float maxPosition);

// kernel for N-body dynamics
__global__ void updatePositionsKernel(float4 *positions, float *velocityPtr,
                                      float time, float maxPosition);

// integrate ODE of a given particle
__device__ void stepIntegration(float4 *position, float4 *velocity,
                                float timeDiff);

// apply reflective boundary conditions to a given particle
__device__ void applyReflectiveBoundaryConditions(float4 *position,
                                                  float4 *velocity,
                                                  float maxPosition);

// apply reflective boundary conditions to a given particle
__device__ void applyPeriodicBoundaryCondition(float4 *position,
                                               float4 *velocity,
                                               float maxPosition);

// compute the body-body interactions
__device__ void computeBodyBodyInteractions(float3 *acceleration,
                                            float4 const *position,
                                            float4 const *positions,
                                            size_t nBodies);

// compute the gravitation
__device__ void addBodyBodyGravitation(float3 *acceleration,
                                       float4 const *position,
                                       float4 const *otherPosition);

__device__ void stepIntegrationLeapfrog(float4 *position, float4 *velocity,
                                        float3 *acceleration, float timeDiff);

/*
 * global function definitions
 */

void initCUDA() {
  // check for CUDA devices
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  checkCUDAError("initCUDA()");
  if (deviceCount == 0) {
    cerr << "Error: no CUDA device found" << endl;
    exit(1);
  }

  // use CUDA device 0
  // cudaGLSetGLDevice(0);
  checkCUDAError("initCUDA()");

  printCUDAVersion();
}

void checkCUDAError(const char *functionName) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    cerr << endl
         << "CUDA error in " << functionName << ": "
         << cudaGetErrorString(error) << endl;
    exit(1);
  }
}

void printCUDAVersion() {
  int driverVersion = 0, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  cout << "CUDA driver version: " << driverVersion / 1000 << "."
       << driverVersion % 100 << endl;
  cout << "CUDA runtime version: " << runtimeVersion / 1000 << "."
       << runtimeVersion % 100 << endl;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cout << "CUDA device 0: " << deviceProp.name << endl;
  cout << "  Compute capability: " << deviceProp.major << "."
       << deviceProp.minor << endl;
  cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount
       << endl;
}

void launchCudaKernel(cudaGraphicsRes_pt &positionCudaVBO,
                      float *velocityCudaPtr, size_t nParticles,
                      float maxPosition) {
  static float timeOld = clock() / static_cast<float>(CLOCKS_PER_SEC);
  size_t nBytes;
  float4 *positions;
  cudaGraphicsMapResources(1, &positionCudaVBO, 0);
  checkCUDAError("launchCudaKernel()");
  cudaGraphicsResourceGetMappedPointer((void **)&positions, &nBytes,
                                       positionCudaVBO);
  checkCUDAError("launchCudaKernel()");

  float time = clock() / static_cast<float>(CLOCKS_PER_SEC);
  // setPositionsKernel<<<nBlocks, nParticles / nBlocks>>>(positions, time,
  // maxPosition);
  updatePositionsKernel<<<nBlocks, nParticles / nBlocks>>>(
      positions, velocityCudaPtr, time - timeOld, maxPosition);
  checkCUDAError("launchCudaKernel()");
  timeOld = time;
  cudaGraphicsUnmapResources(1, &positionCudaVBO, 0);
  checkCUDAError("launchCudaKernel()");
}

/*
 * local function definitions
 */

// simple kernel for testing
__global__ void setPositionsKernel(float4 *positions, float time,
                                   float maxPosition) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int nParticles = gridDim.x * blockDim.x;

  positions[idx].x =
      maxPosition * 0.5f *
      (__cosf(3 * (3.14159f * idx / (float)nParticles + time)) + 1.0f);
  positions[idx].y =
      maxPosition * 0.5f *
      (__cosf(4 * (3.14159f * idx / (float)nParticles + time)) + 1.0f);
  positions[idx].z = maxPosition * idx / (float)nParticles;
}

// kernel for N-body dynamics
__global__ void updatePositionsKernel(float4 *positions, float *velocityPtr,
                                      float timeDiff, float maxPosition) {
  unsigned int idx{blockIdx.x * blockDim.x + threadIdx.x};
  unsigned int nParticles = gridDim.x * blockDim.x;
  float4 position = positions[idx];
  float4 *velocities{(float4 *)velocityPtr};
  float3 acceleration{0.0f};

  computeBodyBodyInteractions(&acceleration, &position, positions, nParticles);

  // stepIntegration(&(positions[idx]), &(velocities[idx]), timeDiff);
  stepIntegrationLeapfrog(&(positions[idx]), &(velocities[idx]), &acceleration,
                          timeDiff);

  // applyReflectiveBoundaryConditions(&(positions[idx]), &(velocities[idx]),
  // maxPosition);
  applyPeriodicBoundaryCondition(&(positions[idx]), &(velocities[idx]),
                                 maxPosition);
}

__device__ void stepIntegration(float4 *position, float4 *velocity,
                                float timeDiff) {
  position->x += timeDiff * velocity->x;
  position->y += timeDiff * velocity->y;
  position->z += timeDiff * velocity->z;
}

__device__ void stepIntegrationLeapfrog(float4 *position, float4 *velocity,
                                        float3 *acceleration, float timeDiff) {
  velocity->x += timeDiff * acceleration->x;
  velocity->y += timeDiff * acceleration->y;
  velocity->z += timeDiff * acceleration->z;
  position->x += timeDiff * velocity->x;
  position->y += timeDiff * velocity->y;
  position->z += timeDiff * velocity->z;
}

__device__ void applyReflectiveBoundaryConditions(float4 *position,
                                                  float4 *velocity,
                                                  float maxPosition) {
  if (position->x < 0.0f) {
    position->x *= -1;
    velocity->x *= -1;
  } else if (position->x > maxPosition) {
    position->x = maxPosition - (position->x - maxPosition);
    velocity->x *= -1;
  }
  if (position->y < 0.0f) {
    position->y *= -1;
    velocity->y *= -1;
  } else if (position->y > maxPosition) {
    position->y = maxPosition - (position->y - maxPosition);
    velocity->y *= -1;
  }
  if (position->z < 0.0f) {
    position->z *= -1;
    velocity->z *= -1;
  } else if (position->z > maxPosition) {
    position->z = maxPosition - (position->z - maxPosition);
    velocity->z *= -1;
  }
}

__device__ void applyPeriodicBoundaryCondition(float4 *position,
                                               float4 *velocity,
                                               float maxPosition) {
  position->x = fmodf(position->x + maxPosition, maxPosition);
  position->y = fmodf(position->y + maxPosition, maxPosition);
  position->z = fmodf(position->z + maxPosition, maxPosition);
}

__device__ void computeBodyBodyInteractions(float3 *acceleration,
                                            float4 const *position,
                                            float4 const *positions,
                                            size_t nBodies) {
  for (int i = 0; i < nBodies; i++) {
    float4 otherPosition = positions[i];
    addBodyBodyGravitation(acceleration, position, &otherPosition);
  }
}

__device__ void addBodyBodyGravitation(float3 *acceleration,
                                       float4 const *position,
                                       float4 const *otherPosition) {
  float3 direction = {
      otherPosition->x - position->x,
      otherPosition->y - position->y,
      otherPosition->z - position->z,
  };
  float distSqrWEps = direction.x * direction.x + direction.y * direction.y +
                      direction.z * direction.z + EPSILON * EPSILON;
  float factor =
      G * otherPosition->w / sqrtf(distSqrWEps * distSqrWEps * distSqrWEps);
  acceleration->x += factor * direction.x;
  acceleration->y += factor * direction.y;
  acceleration->z += factor * direction.z;
}
