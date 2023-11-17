#ifndef GLOBAL_H
#define GLOBAL_H

#include <GL/freeglut.h> // includes gl.h, glext.h, glu.h

constexpr float EPSILON{1e-5f};
constexpr float EPSILON_SQR{EPSILON * EPSILON};
constexpr float G{6.6743e-11f};
constexpr float KB{1.0f};
// constexpr float KB{1.38e-23f};

constexpr float ENERGY{0.5};
constexpr float SIGMA{0.45f};
constexpr float SOFTENING_SQR{0.1f};
constexpr float SIGMA_POW_SIX{SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA};

constexpr size_t gridSize = 2;
constexpr size_t blockSize = 1024;

// model parameters
constexpr GLsizei nParticles{gridSize * blockSize};
constexpr float maxPosition{10.0f}; // bounding box size in each dimension
constexpr float mass{1.0f};         // mass of each particle

#endif // GLOBAL_H
