// includes, GL
#include <GL/glew.h>

// includes
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

extern float animTime;

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);

// Callbacks

int drawMode=GL_TRIANGLE_FAN; // the default draw mode
extern int sleepTime;
int sleepInc=100;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

//! Display callback for GLUT
//! Keyboard events handler for GLUT
//! Display callback for GLUT
void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, translate_z);
  glRotatef(rotate_x, 1.0, 0.0, 0.0);
  glRotatef(rotate_y, 0.0, 1.0, 0.0);
  
  // run CUDA kernel to generate vertex positions
  runCuda();
  
  // slow the rendering when the GPU is too fast.
  if(sleepTime) usleep(sleepTime);

  // render the data
  renderCuda(drawMode);
  
  glutSwapBuffers();
  glutPostRedisplay();
  
  animTime += 0.01;
}

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
  case 'q': case(27) :
    exit(0);
    break;
  case 'S': sleepTime += 100; break;
  case 's':
    sleepTime = (sleepTime > 0)?sleepTime -= sleepInc:0;
    break;
  case 'd': case 'D':
    switch(drawMode) {
    case GL_POINTS: drawMode = GL_LINE_STRIP; break;
    case GL_LINE_STRIP: drawMode = GL_TRIANGLE_FAN; break;
    default: drawMode=GL_POINTS;
    } break;
  }
  glutPostRedisplay();
}

// Mouse event handlers for GLUT
void mouse(int button, int state, int x, int y)
{
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1<<button;
  } else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }
  
  mouse_old_x = x; mouse_old_y = y;
  glutPostRedisplay();
}

void motion(int x, int y)
{
  float dx, dy;
  dx = x - mouse_old_x;
  dy = y - mouse_old_y;
  
  if (mouse_buttons & 1) {
    rotate_x += dy * 0.2;
    rotate_y += dx * 0.2;
  } else if (mouse_buttons & 4) {
    translate_z += dy * 0.01;
  }
  
  mouse_old_x = x;
  mouse_old_y = y;
}

