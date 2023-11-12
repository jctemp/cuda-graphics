#ifndef GLOBAL_H
#define GLOBAL_H

#include <GL/freeglut.h> // includes gl.h, glext.h, glu.h

constexpr float EPSILON{1e-5};
constexpr float EPSILON_SQR{EPSILON * EPSILON};
constexpr float G{6.6743e-11};

constexpr float SOFTENING{1e-5};
constexpr float SOFTENING_SQR{SOFTENING * SOFTENING};
constexpr float SIGMA{1e-5};
constexpr float SIGMA_POW_SIX{SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA};

constexpr size_t gridSize = 4;
constexpr size_t blockSize = 1024;

// model parameters
constexpr GLsizei nParticles{gridSize * blockSize};
constexpr float maxPosition{10.0f}; // bounding box size in each dimension
constexpr float mass{1.0f};         // mass of each particle

#endif // GLOBAL_H
