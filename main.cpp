/**
 * OpenGL 3 application using opengl_3_toolkit.
 *
 * @author Volker Ahlers<br>
 *         volker.ahlers@fh-hannover.de
 * @version 2010-03
 */

#include <GL/glew.h> // has to be included before freeglut.h

#include <GL/freeglut.h> // includes gl.h, glext.h, glu.h
#include <cstdlib>
#include <ctime>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#include "graphics.h"
#include "kernel.h"

using namespace std;

/*
 * global definitions
 */

// buffer offset type conversion (cf. OpenGL Programming Guide, 5th ed., p. 87)
#define BUFFER_OFFSET(bytes) ((GLubyte *)NULL + (bytes))

/*
 * global variables
 */

// model parameters
const GLsizei nParticles = 4096;
const float maxPosition = 10.0f; // bounding box size in each dimension
const float mass = 1.0f;         // mass of each particle

// model variables
GLuint positionVBO = 0;
cudaGraphicsRes_pt positionCudaVBO = 0;
float *velocityCudaPtr = 0;

// rendering parameters
const GLfloat pointSize = 3.0f;

/*
 * function prototypes
 */

// initialize positions as OpenGL VBO and velocities as CUDA device meory array
void initModel(GLuint &vbo, cudaGraphicsRes_pt &positionCudaVBO,
               float_pt &velocityCudaPtr, GLsizei nParticles,
               float maxPosition);

// GLUT display callback function
void display();

/*
 * function definitions
 */

// main function
int main(int argc, char **argv) {

  // initialize everything
  initGLUT(&argc, argv);
  initGLEW();
  initCUDA();
  initModel(positionVBO, positionCudaVBO, velocityCudaPtr, nParticles,
            maxPosition);

  // register GLUT display callback function
  glutDisplayFunc(display);

  // start GLUT main loop
  glutMainLoop();

  return 0;
}

void initModel(GLuint &vbo, cudaGraphicsRes_pt &positionCudaVBO,
               float_pt &velocityCudaPtr, GLsizei nParticles,
               float maxPosition) {

  // initialize particle positions and velocities
  size_t memSize = nParticles * 4 * sizeof(GLfloat);
  float *positions = new float[memSize];
  float *velocities = new float[memSize];

  srand(time(NULL));
  for (int i = 0; i < nParticles; i++) {
    for (int j = 0; j < 3; j++) {
      positions[4 * i + j] =
          maxPosition * rand() / static_cast<float>(RAND_MAX);
      velocities[4 * i + j] =
          0.5 * maxPosition * (rand() / static_cast<float>(RAND_MAX) - 0.5f);
    }
    positions[4 * i + 3] = mass;  // use w component for particle mass
    velocities[4 * i + 3] = 0.0f; // unused
  }

  // create VBO for particle positions
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, memSize, positions, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register VBO for particle positions with CUDA
  cudaGraphicsGLRegisterBuffer(&positionCudaVBO, vbo, cudaGraphicsMapFlagsNone);

  // copy particle velocities to CUDA device memory
  cudaMalloc(&velocityCudaPtr, memSize);
  cudaMemcpy(velocityCudaPtr, velocities, memSize, cudaMemcpyHostToDevice);

  delete[] velocities;
  velocities = 0;
  delete[] positions;
  positions = 0;

  checkGLError("initModel()");
}

void display() {

  // update simulation using CUDA
  launchCudaKernel(positionCudaVBO, velocityCudaPtr, nParticles, maxPosition);

  // render particles using OpenGL
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();

  // apply camera transformation
  glTranslatef(0.0f, 0.0f, translateZ);
  glRotatef(rotateX, 1.0, 0.0, 0.0);
  glRotatef(rotateY, 0.0, 1.0, 0.0);

  // set light
  const GLfloat lightPosition[] = {1.0f, 1.0f, 1.0f, 0.0f};
  glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

  // render teapot (test only)
  // glutSolidTeapot(1);

  // center particle box
  GLfloat centerTrans = -0.5f * maxPosition;
  glTranslatef(centerTrans, centerTrans, centerTrans);

  // render particles as VBO
  glColor3f(0.7f, 0.2f, 0.2f);
  glPointSize(pointSize);
  glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
  glVertexPointer(3, GL_FLOAT, 4 * sizeof(float), BUFFER_OFFSET(0));
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, nParticles);
  glDisableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // render particle box (quick and dirty)
  glColor3f(0.5f, 0.5f, 1.0f);
  glLineWidth(2.0f);

  glBegin(GL_LINE_STRIP);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, maxPosition, 0.0f);
  glVertex3f(0.0f, maxPosition, maxPosition);
  glVertex3f(0.0f, 0.0f, maxPosition);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glEnd();

  glBegin(GL_LINE_STRIP);
  glVertex3f(maxPosition, 0.0f, 0.0f);
  glVertex3f(maxPosition, maxPosition, 0.0f);
  glVertex3f(maxPosition, maxPosition, maxPosition);
  glVertex3f(maxPosition, 0.0f, maxPosition);
  glVertex3f(maxPosition, 0.0f, 0.0f);
  glEnd();

  glBegin(GL_LINES);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(maxPosition, 0.0f, 0.0f);
  glVertex3f(0.0f, maxPosition, 0.0f);
  glVertex3f(maxPosition, maxPosition, 0.0f);
  glVertex3f(0.0f, maxPosition, maxPosition);
  glVertex3f(maxPosition, maxPosition, maxPosition);
  glVertex3f(0.0f, 0.0f, maxPosition);
  glVertex3f(maxPosition, 0.0f, maxPosition);
  glEnd();

  glPopMatrix();

  // swap front and back buffers
  glutSwapBuffers();

  // determine frame rate
  static int nFrames = 0;
  static float timeOld = clock() / static_cast<float>(CLOCKS_PER_SEC);
  nFrames++;
  float time = clock() / static_cast<float>(CLOCKS_PER_SEC);
  if (time - timeOld > 1.0f) {
    cout << "FPS: " << nFrames / (time - timeOld) << endl;
    nFrames = 0;
    timeOld = time;
  }

  // request immediate redisplay to keep simulation running
  glutPostRedisplay();

  checkGLError("display()");
}
