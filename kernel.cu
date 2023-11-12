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

/*
 * global variables
 */

constexpr float EPSILON{1e-5};
constexpr float EPSILON_SQR{EPSILON * EPSILON};
constexpr float SOFTENING{1e-5};
constexpr float SOFTENING_SQR{SOFTENING * SOFTENING};
constexpr float SIGMA{1e-5};
constexpr float SIGMA_POW_SIX{SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA};
constexpr float G{6.6743e-11};
constexpr size_t gridSize = 4;
constexpr size_t blockSize = 1024;

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
                                            size_t nBodies, float maxPosition);

// compute the gravitation
__device__ void addBodyBodyGravitation(float3 *acceleration,
                                       float4 const *position,
                                       float4 const *otherPosition,
                                       float maxPosition);

__device__ void addVanDerWaalsForces(float3 *acceleration,
                                     float4 const *position,
                                     float4 const *otherPosition,
                                     float maxPosition);

// euler integration
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
  // setPositionsKernel<<<gridSize, nParticles / nBlocks>>>(positions, time,
  // maxPosition);
  updatePositionsKernel<<<gridSize, blockSize>>>(positions, velocityCudaPtr,
                                                 time - timeOld, maxPosition);
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

  __shared__ float4 sharedPositions[blockSize];

  unsigned int globalIdx{blockIdx.x * blockDim.x + threadIdx.x};
  float3 acceleration{0.0f};

  float4 currentPosition = positions[globalIdx];
  float4 currentVelocity = ((float4 *)velocityPtr)[globalIdx];

  for (size_t i{0}; i < gridSize; i++) {
    size_t idx = blockDim.x * i + threadIdx.x;
    sharedPositions[threadIdx.x] = positions[idx];

    __syncthreads();

    computeBodyBodyInteractions(&acceleration, &currentPosition,
                                sharedPositions, blockSize, maxPosition);

    __syncthreads();
  }

  // stepIntegration(&(positions[globalIdx]), &(velocities[globalIdx]),
  // timeDiff);
  stepIntegrationLeapfrog(&currentPosition, &currentVelocity, &acceleration,
                          timeDiff);

  // applyReflectiveBoundaryConditions(&(positions[globalIdx]),
  // &(velocities[globalIdx]), maxPosition);
  applyPeriodicBoundaryCondition(&currentPosition, &currentVelocity,
                                 maxPosition);

  positions[globalIdx] = currentPosition;
  ((float4 *)velocityPtr)[globalIdx] = currentVelocity;
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
                                            size_t nBodies, float maxPosition) {
  for (int i = 0; i < nBodies; i++) {
    float4 otherPosition = positions[i];
    // addBodyBodyGravitation(acceleration, position, &otherPosition,
    // maxPosition);
    addVanDerWaalsForces(acceleration, position, &otherPosition, maxPosition);
  }
}

__device__ void addBodyBodyGravitation(float3 *acceleration,
                                       float4 const *position,
                                       float4 const *otherPosition,
                                       float maxPosition) {
  float3 direction = {
      otherPosition->x - position->x,
      otherPosition->y - position->y,
      otherPosition->z - position->z,
  };

  auto halfMax = maxPosition * 0.5f;

  direction.x = direction.x > halfMax    ? direction.x - maxPosition
                : direction.x < -halfMax ? direction.x + maxPosition
                                         : direction.x;
  direction.y = direction.y > halfMax    ? direction.y - maxPosition
                : direction.y < -halfMax ? direction.y + maxPosition
                                         : direction.y;
  direction.z = direction.z > halfMax    ? direction.z - maxPosition
                : direction.z < -halfMax ? direction.z + maxPosition
                                         : direction.z;

  float distSqrWEps = direction.x * direction.x + direction.y * direction.y +
                      direction.z * direction.z + EPSILON_SQR;
  float factor =
      G * otherPosition->w / sqrtf(distSqrWEps * distSqrWEps * distSqrWEps);
  acceleration->x += factor * direction.x;
  acceleration->y += factor * direction.y;
  acceleration->z += factor * direction.z;
}

__device__ void addVanDerWaalsForces(float3 *acceleration,
                                     float4 const *position,
                                     float4 const *otherPosition,
                                     float maxPosition) {

  float3 direction = {
      otherPosition->x - position->x,
      otherPosition->y - position->y,
      otherPosition->z - position->z,
  };

  auto halfMax = maxPosition * 0.5f;

  direction.x = direction.x > halfMax    ? direction.x - maxPosition
                : direction.x < -halfMax ? direction.x + maxPosition
                                         : direction.x;
  direction.y = direction.y > halfMax    ? direction.y - maxPosition
                : direction.y < -halfMax ? direction.y + maxPosition
                                         : direction.y;
  direction.z = direction.z > halfMax    ? direction.z - maxPosition
                : direction.z < -halfMax ? direction.z + maxPosition
                                         : direction.z;

  float distSqr = direction.x * direction.x + direction.y * direction.y +
                  direction.z * direction.z + SOFTENING_SQR;

  auto sigmaDistPow6 = SIGMA_POW_SIX / (distSqr * distSqr * distSqr);
  auto factor = SIGMA_POW_SIX * (1.0f - 2.0f * SIGMA_POW_SIX) * 24.0f *
                EPSILON / (distSqr * position->w);

  acceleration->x += factor * direction.x;
  acceleration->y += factor * direction.y;
  acceleration->z += factor * direction.z;
}
