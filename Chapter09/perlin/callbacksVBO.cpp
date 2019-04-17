//callbacks (Rob Farber)
#include <GL/glew.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);

// Callback variables
extern float animTime;
extern int sleepTime, sleepInc;
int drawMode=GL_TRIANGLE_FAN; // the default draw mode
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// break the file modulatity so both perlin and demo kernels build
// some initial values for Perlin
float gain=0.75f, xStart=2.f, yStart=1.f;
float zOffset = 0.0f, octaves = 2.f, lacunarity = 2.0f;

// GLUT callbacks display, keyboard, mouse
void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, translate_z);
  glRotatef(rotate_x, 1.0, 0.0, 0.0);
  glRotatef(rotate_y, 0.0, 1.0, 0.0);

  runCuda(); // run CUDA kernel to generate vertex positions

  renderCuda(drawMode); // render the data
  
  glutSwapBuffers();
  glutPostRedisplay();
  
  animTime += 0.01;
}

void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
  case('q') : case(27) : // exit
    exit(0);
    break;
  case 'd': case 'D': // Drawmode
    switch(drawMode) {
    case GL_POINTS: drawMode = GL_LINE_STRIP; break;
    case GL_LINE_STRIP: drawMode = GL_TRIANGLE_FAN; break;
    default: drawMode=GL_POINTS;
    } break;
  case '+': // Perlin: lower the ocean level
    zOffset += 0.01;
    zOffset = (zOffset > 1.0)? 1.0:zOffset; // guard input
    break;
  case '-': // Perlin: raise the ocean level
    zOffset -= 0.01;
    zOffset = (zOffset < -1.0)? -1.0:zOffset; // guard input
    break;
  case 'k': // move within the Perlin function
    yStart -= 0.1;
    break;
  case 'j': // move within the Perlin function
    yStart += 0.1;
    break;
  case 'l': // move within the Perlin function
    xStart += 0.1;
    break;
  case 'h': // move within the Perlin function
    xStart -= 0.1;
    break;
  case 'I': // Perlin: change gain
    gain += 0.25;
    break;
  case 'i': // Perlin: change gain
    gain -= 0.25;
    gain = (gain < 0.25)?0.25:gain; // guard input
    break;
  case 'O': // Perlin: change octaves
    octaves += 1.0f;
    octaves = (octaves > 8)?8:octaves; // guard input
    break;
  case 'o': // Perlin: change octaves
    octaves -= 1.0f;
    octaves = (octaves<2)?2:octaves; // guard input
    break;
  case 'P': // Perlin: change lacunarity
    lacunarity += 0.25;
    break;
  case 'p': // Perlin: change lacunarity
    lacunarity -= 0.25;
    lacunarity = (lacunarity<0.2)?0.2:lacunarity; // guard input
    break;
  case 'S': // Slow the simulation down
    sleepTime += 100;
    break;
  case 's': // Speed the simulation up
    sleepTime = (sleepTime > 0)?sleepTime -= sleepInc:0;
    break;
  }
  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1<<button;
  } else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }
  
  mouse_old_x = x;
  mouse_old_y = y;
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
 rotate_x = (rotate_x < -60.)?-60.:(rotate_x > 60.)?60:rotate_x;
 rotate_y = (rotate_y < -60.)?-60.:(rotate_y > 60.)?60:rotate_y;
  
  mouse_old_x = x;
  mouse_old_y = y;
}


