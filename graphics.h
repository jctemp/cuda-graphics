/*
 * graphics.h
 *
 *  Created on: 24.10.2010
 *      Author: ahlers
 */

#ifndef GRAPHICS_H_
#define GRAPHICS_H_

/*
 * global variable declarations
 */

// camera control
extern float rotateX;
extern float rotateY;
extern float translateZ;

/*
 * function prototypes
 */

// initialize GLUT, create window
// (using OpenGL compatibility profile)
void initGLUT(int* argc, char** argv, int versionMajor = 2, int versionMinor = 1,
    int width = 700, int height = 700, const char* title = "CUDA/OpenGL");

// initialize GLEW
void initGLEW();

// check for GL errors, exit in case of errors
void checkGLError(const char* functionName);

// print GL version information
void printGLVersion();

#endif /* GRAPHICS_H_ */
