/*
 * graphics.cpp
 *
 *  Created on: 24.10.2010
 *      Author: ahlers
 */

#include <cstdlib>
#include <iostream>
#include <GL/glew.h>      // has to be included before freeglut.h
#include <GL/freeglut.h>  // includes gl.h, glext.h, glu.h
#include "graphics.h"

using namespace std;

/*
 * global variables
 */

// camera parameters
const GLdouble fovy = 60.0;
const GLdouble zNear = 0.01;
const GLdouble zFar = 100.0;
const float mouseRotationVel = 0.5f;
const float mouseTranslationVel = 0.05f;

// camera control
float rotateX = 0.0f;
float rotateY = 0.0f;
float translateZ = -15.0f;
int mouseXOld;
int mouseYOld;
bool mouseIsRightButtonPressed = false;

/*
 * local function prototypes
 */

void keyboard(unsigned char key, int x, int y);

void motion(int x, int y);

void mouse(int button, int state, int x, int y);

void reshape(int width, int height);

/*
 * global function definitions
 */

void initGLUT(int* argc, char** argv, int versionMajor, int versionMinor,
    int width, int height, const char* title) {
  glutInit(argc, argv);
  if (versionMajor >= 4 || (versionMajor == 3 && versionMinor >= 2)) {
    glutInitContextVersion(versionMajor, versionMinor);
//    glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
  }

  // create window
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(width, height);
  glutCreateWindow(title);
  printGLVersion();

  // enable depth test
  glEnable(GL_DEPTH_TEST);

  // enable lighting and color
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_COLOR_MATERIAL);
  glClearColor(0.95f, 0.95f, 0.95f, 1.0f);    // background color

  // initialize modelview matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // register callback functions
  glutKeyboardFunc(keyboard);
  glutMotionFunc(motion);
  glutMouseFunc(mouse);
  glutReshapeFunc(reshape);

  checkGLError("initGLUT()");
}

void initGLEW() {
  GLenum error = glewInit();
  if (error != GLEW_OK) {
    cerr << endl << "GLEW error: " << glewGetErrorString(error) << endl;
    exit(1);
  }
  cout << "GLEW version: " << glewGetString(GLEW_VERSION) << endl << endl;

  // ignore GL error (invalid enum) in glewInit
  glGetError();
}

void checkGLError(const char* functionName) {
  GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    cerr << endl << "GL error in " << functionName << ": " << gluErrorString(error) << endl;
    exit(1);
  }
}

void printGLVersion() {
  cout << "GL version: " << reinterpret_cast<const char*>(glGetString(GL_VERSION)) << endl;
  cout << "GL vendor: " << reinterpret_cast<const char*>(glGetString(GL_VENDOR)) << endl;
  cout << "GL renderer: " << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << endl;
  cout << "GLSL version: " << reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION))
       << endl << endl;
}

/*
 * local function definitions
 */

void keyboard(unsigned char key, int x, int y) {
  if (key == 'w') {
    static bool isWireframeMode = false;
    isWireframeMode = !isWireframeMode;
    if (isWireframeMode) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    glutPostRedisplay();
  }
}

void motion(int x, int y) {
  int dx = x - mouseXOld;
  int dy = y - mouseYOld;

  if (mouseIsRightButtonPressed) {
    translateZ += mouseTranslationVel * dy;
  }
  else {
    rotateY += mouseRotationVel * dx;
    rotateX += mouseRotationVel * dy;
  }
  mouseXOld = x;
  mouseYOld = y;

  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
  if (button == GLUT_RIGHT_BUTTON) {
    if (state == GLUT_DOWN) {
      mouseIsRightButtonPressed = true;
    }
    else if (state == GLUT_UP) {
      mouseIsRightButtonPressed = false;
    }
  }
  mouseXOld = x;
  mouseYOld = y;
}

void reshape(int width, int height) {
  // set viewport
  glViewport(0, 0, width, height);

  // set projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fovy, static_cast<GLdouble>(width) / height, zNear, zFar);
}

