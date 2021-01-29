//
// Extended Mars Lander
// Version 1.0
// Graphical functions
// Arthur Tombs, May 2013
//

// Original license:

// Mars lander simulator
// Version 1.6
// Graphics functions
// Gabor Csanyi and Andrew Gee, September 2011

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation, to make use of it
// for non-commercial purposes, provided that (a) its original authorship
// is acknowledged and (b) no modified versions of the source code are
// published. Restriction (b) is designed to protect the integrity of the
// exercise for future generations of students. The authors would be happy
// to receive any suggested modifications by private correspondence to
// ahg@eng.cam.ac.uk and gc121@eng.cam.ac.uk.

// Some functions adapted from freeglut_geometry.c, which is covered by the
// following license:
//
// Copyright (c) 1999-2000 Pawel W. Olszta. All Rights Reserved.
// Written by Pawel W. Olszta, <olszta@sourceforge.net>
// Creation date: Fri Dec 3 1999
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// PAWEL W. OLSZTA BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Some functions adapted from trackball.cpp by Gavin Bell, which is covered by
// the following license:
//
// (c) Copyright 1993, 1994, Silicon Graphics, Inc.
// ALL RIGHTS RESERVED
// Permission to use, copy, modify, and distribute this software for
// any purpose and without fee is hereby granted, provided that the above
// copyright notice appear in all copies and that both the copyright notice
// and this permission notice appear in supporting documentation, and that
// the name of Silicon Graphics, Inc. not be used in advertising
// or publicity pertaining to distribution of the software without specific,
// written prior permission.
//
// THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
// AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
// INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
// FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
// GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
// SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
// KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
// LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
// THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
// ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
// POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
//
// US Government Users Restricted Rights
// Use, duplication, or disclosure by the Government is subject to
// restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
// (c)(1)(ii) of the Rights in Technical Data and Computer Software
// clause at DFARS 252.227-7013 and/or in similar or successor
// clauses in the FAR or the DOD or NASA FAR Supplement.
// Unpublished-- rights reserved under the copyright laws of the
// United States.  Contractor/manufacturer is Silicon Graphics,
// Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
//
// OpenGL(TM) is a trademark of Silicon Graphics, Inc.

#define DECLARE_GLOBAL_VARIABLES
#include "lander.h"

void invert (double m[], double mout[])
  // Inverts a 4x4 OpenGL rotation matrix
{
  double zero_three, one_three, two_three;
  zero_three = -m[12]*m[0] - m[13]*m[1] - m[14]*m[2];
  one_three  = -m[12]*m[4] - m[13]*m[5] - m[14]*m[6];
  two_three  = -m[12]*m[8] - m[13]*m[9] - m[14]*m[10];
  mout[1]  = m[4]; mout[4] = m[1]; mout[2] = m[8]; mout[8] = m[2]; 
  mout[6]  = m[9]; mout[9] = m[6]; mout[12] = zero_three; mout[13] = one_three; 
  mout[14] = two_three; mout[0] = m[0]; mout[5] = m[5]; mout[10] = m[10];
  mout[15] = 1.0; mout[3] = 0.0; mout[7] = 0.0; mout[11] = 0.0;
}

vector3d matrix_mult (double m[], vector3d v)
  // Multiply a vector by a 4x4 rotation matrix (ignoring homogeneous components)
{
  return vector3d (
  m[0]*v.x + m[4]*v.y + m[8]*v.z,
  m[1]*v.x + m[5]*v.y + m[9]*v.z,
  m[2]*v.x + m[6]*v.y + m[10]*v.z);
}

quat_t matrix_to_quat (double m[])
  // Decomposes a 4x4 OpenGL rotation matrix into a unit quaternion
{

  double w, x, y, z;

  w = sqrt(max(0.0, 1.0 + m[0] + m[5] + m[10])) * 0.5;
  x = sqrt(max(0.0, 1.0 + m[0] - m[5] - m[10])) * 0.5;
  y = sqrt(max(0.0, 1.0 - m[0] + m[5] - m[10])) * 0.5;
  z = sqrt(max(0.0, 1.0 - m[0] - m[5] + m[10])) * 0.5;

  if (w >= x && w >= y && w >= z) {
    x = copysign(x, m[6] - m[9]);
    y = copysign(y, m[8] - m[2]);
    z = copysign(z, m[1] - m[4]);
  } else if (x >= w && x >= y && x >= z) {
    w = copysign(w, m[6] - m[9]);
    y = copysign(y, m[1] + m[4]);
    z = copysign(z, m[8] + m[2]);
  } else if (y >= w && y >= x && y >= z) {
    w = copysign(w, m[8] - m[2]);
    x = copysign(x, m[1] + m[4]);
    z = copysign(z, m[6] + m[9]);
  } else if (z >= w && z >= x && z >= y) {
    w = copysign(w, m[1] - m[4]);
    x = copysign(x, m[8] + m[2]);
    y = copysign(y, m[6] + m[9]);
  }

  return quat_t(w, x, y, z).norm();
}

vector3d matrix_to_zxz_eul (double m[])
  // Decomposes a 4x4 OpenGL rotation matrix into ZXZ Euler angles
{
  double phi, theta, psi;

  // Elevation angle in the range 0-pi radians
  theta = acos(m[10]);
  if ((1.0 - abs(m[10])) < 1e-8) {
    // Degenerate case of sin(theta) = 0
    // Arbitrary choice of phi and psi, so set phi = 0
    phi = 0.0;
    if (m[10] > 0) {
      psi = acos((m[0]+m[5])*0.5);
    } else {
      psi = acos((m[1]+m[4])*0.5);
    }
  } else {
    phi = atan2(m[2],-m[6]);
    psi = atan2(m[8], m[9]);
  }

  return vector3d(phi, theta, psi);
}

quat_t axis_to_quat (const vector3d &a, const double phi)
  // Given an axis and angle, compute quaternion
{
  return quat_t(cos(phi/2.0), a.norm() * sin(phi/2.0));
}

double project_to_sphere (const double r, const double x, const double y)
  // Project an x,y pair onto a sphere of radius r or a hyperbolic sheet if
  // we are away from the centre of the sphere
{
  double d, t, z;

  d = sqrt(x*x + y*y);
  if (d < (r * 0.70710678118654752440)) z = sqrt(r*r - d*d);
  else { // on hyperbola
    t = r / 1.41421356237309504880;
    z = t*t / d;
  }
  return z;
}

void quat_to_matrix (double m[], const quat_t &q)
  // Convert quaternion into a rotation matrix
{

  m[0] = q.s * q.s + q.v.x * q.v.x - q.v.y * q.v.y - q.v.z * q.v.z;
  m[1] = 2.0 * (q.v.x * q.v.y - q.s * q.v.z);
  m[2] = 2.0 * (q.s * q.v.y + q.v.x * q.v.z);
  m[3] = 0.0;

  m[4] = 2.0 * (q.s * q.v.z + q.v.x * q.v.y);
  m[5] = q.s * q.s - q.v.x * q.v.x + q.v.y * q.v.y - q.v.z * q.v.z;
  m[6] = 2.0 * (q.v.y * q.v.z - q.s * q.v.x);
  m[7] = 0.0;

  m[8] = 2.0 * (q.v.x * q.v.z - q.s * q.v.y);
  m[9] = 2.0 * (q.s * q.v.x + q.v.y * q.v.z);
  m[10] = q.s * q.s - q.v.x * q.v.x - q.v.y * q.v.y + q.v.z * q.v.z;
  m[11] = 0.0;

  m[12] = 0.0; m[13] = 0.0; m[14] = 0.0; m[15] = 1.0;
}

quat_t track_quats (const double p1x, const double p1y, const double p2x, const double p2y)
  // Derive quaternion from x and y mouse displacements
{
  double t, phi;
  vector3d p1, p2;

  if ((p1x == p2x) && (p1y == p2y)) {
    return quat_t();
  }

  p1.x = p1x; p1.y = p1y;
  p1.z = project_to_sphere(0.5, p1x, p1y);
  p2.x = p2x; p2.y = p2y;
  p2.z = project_to_sphere(0.5, p2x, p2y);
  t = (p1-p2).abs();
  if (t > 1.0) t = 1.0;
  if (t < -1.0) t = -1.0;
  phi = 2.0 * asin(t);
  return axis_to_quat(p2^p1, phi);
}

void microsecond_time (unsigned long long &t)
  // Returns system time in microseconds, used for introducing delays at low simulation speeds
{
#ifdef WIN32
  LARGE_INTEGER counter, frequency;
  QueryPerformanceFrequency(&frequency);
  frequency.QuadPart /= 1000000;
  QueryPerformanceCounter(&counter);
  t = (unsigned long long)(counter.QuadPart/frequency.QuadPart);
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  t = (unsigned long long)tv.tv_usec + 1000000 * (unsigned long long)tv.tv_sec;
#endif
}

void fghCircleTable (double **sint, double **cost, const int n)
  // Borrowed from freeglut source code, used to draw hemispheres and open cones
{
  int i;
  const int size = abs(n);
  const double angle = 2*M_PI/(double)( ( n == 0 ) ? 1 : n );
  
  *sint = (double*) malloc(sizeof(double) * (size+1));
  *cost = (double*) malloc(sizeof(double) * (size+1));
  if (!(*sint) || !(*cost)) exit(1);
  
  (*sint)[0] = 0.0;
  (*cost)[0] = 1.0;
  
  for (i=1; i<size; i++) {
    (*sint)[i] = sin(angle*i);
    (*cost)[i] = cos(angle*i);
  }
  
  (*sint)[size] = (*sint)[0];
  (*cost)[size] = (*cost)[0];
}

void glutOpenHemisphere (GLdouble radius, GLint slices, GLint stacks)
  // Modified from freeglut's glutSolidSphere
{
  int i, j;
  double z0, z1, r0, r1, *sint1, *cost1, *sint2, *cost2;
  
  fghCircleTable(&sint1, &cost1, -slices);
  fghCircleTable(&sint2, &cost2, stacks*2);
  z1 = cost2[(stacks>0)?1:0];
  r1 = sint2[(stacks>0)?1:0];
  
  // Middle stacks
  for (i=1; i<stacks-1; i++) {
    z0 = z1; z1 = cost2[i+1];
    r0 = r1; r1 = sint2[i+1];
    if ((z1 > 0) || (z0 > 0)) continue; // hemisphere
    glBegin(GL_QUAD_STRIP);
    for (j=0; j<=slices; j++) {
      glNormal3d(cost1[j]*r1, sint1[j]*r1, z1);
      glVertex3d(cost1[j]*r1*radius, sint1[j]*r1*radius, z1*radius);
      glNormal3d(cost1[j]*r0, sint1[j]*r0, z0);
      glVertex3d(cost1[j]*r0*radius, sint1[j]*r0*radius, z0*radius);
    }
    glEnd();
  }

  // Bottom cap
  z0 = z1; r0 = r1;
  glBegin(GL_TRIANGLE_FAN);
  glNormal3d(0,0,-1);
  glVertex3d(0,0,-radius);
  for (j=0; j<=slices; j++) {
    glNormal3d(cost1[j]*r0, sint1[j]*r0, z0);
    glVertex3d(cost1[j]*r0*radius, sint1[j]*r0*radius, z0*radius);
  }
  glEnd();
  
  free(sint1); free(cost1);
  free(sint2); free(cost2);
}

void glutMottledSphere (GLdouble radius, GLint slices, GLint stacks)
  // Modified from freeglut's glutSolidSphere, we use this to draw a mottled sphere by modulating
  // the vertex colours.
{
    int i, j;
    unsigned short rtmp = 0;
    double z0, z1, r0, r1, *sint1, *cost1, *sint2, *cost2;
    double *rnd1, *rnd2, *new_r, *old_r, *tmp;
    double mottle = 0.2;

    fghCircleTable(&sint1, &cost1, -slices);
    fghCircleTable(&sint2, &cost2, stacks*2);
    rnd1 = (double*) calloc(sizeof(double), slices+1);
    rnd2 = (double*) calloc(sizeof(double), slices+1);
    z0 = 1.0; z1 = cost2[(stacks>0)?1:0];
    r0 = 0.0; r1 = sint2[(stacks>0)?1:0];

    // Top cap
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0,0,1);
    glColor3f(0.63, 0.33, 0.22);
    glVertex3d(0,0,radius);
    new_r = rnd1;
    for (j=slices; j>=0; j--) {
      glNormal3d(cost1[j]*r1, sint1[j]*r1, z1);
      if (j) {
        new_r[j] = (1.0-mottle) + mottle*randtab[rtmp];
        rtmp = (rtmp+1)%N_RAND;
      } else new_r[j] = new_r[slices];
      glColor3f(new_r[j]*0.63, new_r[j]*0.33, new_r[j]*0.22);
      glVertex3d(cost1[j]*r1*radius, sint1[j]*r1*radius, z1*radius);
    }
    glEnd();

    // Middle stacks
    old_r = rnd1; new_r = rnd2;
    for (i=1; i<stacks-1; i++) {
      z0 = z1; z1 = cost2[i+1];
      r0 = r1; r1 = sint2[i+1];
      glBegin(GL_QUAD_STRIP);
      for (j=0; j<=slices; j++) {
        glNormal3d(cost1[j]*r1, sint1[j]*r1, z1);
        if (j != slices) {
          new_r[j] = (1.0-mottle) + mottle*randtab[rtmp];
          rtmp = (rtmp+1)%N_RAND;
        } else new_r[j] = new_r[0];
        glColor3f(new_r[j]*0.63, new_r[j]*0.33, new_r[j]*0.22);
        glVertex3d(cost1[j]*r1*radius, sint1[j]*r1*radius, z1*radius);
        glNormal3d(cost1[j]*r0, sint1[j]*r0, z0);
        glColor3f(old_r[j]*0.63, old_r[j]*0.33, old_r[j]*0.22);
        glVertex3d(cost1[j]*r0*radius, sint1[j]*r0*radius, z0*radius);
      }
      tmp = old_r; old_r = new_r; new_r = tmp;
      glEnd();
    }

    // Bottom cap
    z0 = z1; r0 = r1;
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0,0,-1);
    glColor3f(0.63, 0.33, 0.22);
    glVertex3d(0,0,-radius);
    for (j=0; j<=slices; j++) {
      glNormal3d(cost1[j]*r0, sint1[j]*r0, z0);
      glColor3f(old_r[j]*0.63, old_r[j]*0.33, old_r[j]*0.22);
      glVertex3d(cost1[j]*r0*radius, sint1[j]*r0*radius, z0*radius);
    }
    glEnd();

    free(rnd1); free(rnd2);
    free(sint1); free(cost1);
    free(sint2); free(cost2);
}


void drawTexturedSphere (GLdouble radius, GLint slices, GLint stacks)
  // Modified from freeglut's glutSolidSphere, we use this to draw the textured planet
{
  int i, j;
  double z0, z1, r0, r1, *sint1, *cost1, *sint2, *cost2;

  fghCircleTable(&sint1, &cost1, -slices);
  fghCircleTable(&sint2, &cost2, stacks*2);
  r1 = 0.0; z1 = 1.0;

  for (i=0; i<stacks; i++) {
    z0 = z1; z1 = cost2[i+1];
    r0 = r1; r1 = sint2[i+1];
    glBegin(GL_QUAD_STRIP);
    for (j=0; j<=slices; j++) {
      glNormal3d(cost1[j]*r1, sint1[j]*r1, z1);
      glTexCoord2d((double)j/slices, 0.5-asin(z1)/M_PI); glVertex3d(cost1[j]*r1*radius, sint1[j]*r1*radius, z1*radius);
      glNormal3d(cost1[j]*r0, sint1[j]*r0, z0);
      glTexCoord2d((double)j/slices, 0.5-asin(z0)/M_PI); glVertex3d(cost1[j]*r0*radius, sint1[j]*r0*radius, z0*radius);
    }
    glEnd();
  }

  free(sint1); free(cost1);
  free(sint2); free(cost2);
}

void glutCone (GLdouble base, GLdouble height, GLint slices, GLint stacks, bool closed)
  // Modified from freeglut's glutSolidCone, we need this (a) to draw cones without bases and
  // (b) to draw cones with bases, which glutSolidCone does not do correctly under Windows,
  // for some reason.
{
  int i, j;
  double z0, z1, r0, r1, *sint, *cost;
  const double zStep = height / ( ( stacks > 0 ) ? stacks : 1 );
  const double rStep = base / ( ( stacks > 0 ) ? stacks : 1 );
  const double cosn = ( height / sqrt ( height * height + base * base ));
  const double sinn = ( base   / sqrt ( height * height + base * base ));
  
  fghCircleTable(&sint, &cost, -slices);
  z0 = 0.0; z1 = zStep;
  r0 = base; r1 = r0 - rStep;
  
  if (closed) {
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0.0, 0.0, -1.0);
    glVertex3d(0.0, 0.0, z0);
    for (j=0; j<=slices; j++) glVertex3d(cost[j]*r0, sint[j]*r0, z0);
    glEnd();
  }

  for (i=0; i<stacks-1; i++) {
    glBegin(GL_QUAD_STRIP);
    for (j=0; j<=slices; j++) {
      glNormal3d(cost[j]*sinn, sint[j]*sinn, cosn);
      glVertex3d(cost[j]*r0, sint[j]*r0, z0);
      glVertex3d(cost[j]*r1, sint[j]*r1, z1);
    }
    z0 = z1; z1 += zStep;
    r0 = r1; r1 -= rStep;
    glEnd();
  }
  
  glBegin(GL_TRIANGLES);
  glNormal3d(cost[0]*sinn, sint[0]*sinn, cosn);
  for (j=0; j<slices; j++) {
    glVertex3d(cost[j+0]*r0, sint[j+0]*r0, z0);
    glVertex3d(0.0, 0.0, height);
    glNormal3d(cost[j+1]*sinn, sint[j+1]*sinn, cosn);
    glVertex3d(cost[j+1]*r0, sint[j+1]*r0, z0);
  }
  glEnd();
  
  free(sint); free(cost);
}

void enable_lights (void) 
  // Enable the appropriate subset of lights
{
  if (static_lighting) {
    glDisable(GL_LIGHT0); glDisable(GL_LIGHT1);
    glEnable(GL_LIGHT2); glEnable(GL_LIGHT3);
    glDisable(GL_LIGHT4); glDisable(GL_LIGHT5);
  } else {
    glEnable(GL_LIGHT0); glEnable(GL_LIGHT1);
    glDisable(GL_LIGHT2); glDisable(GL_LIGHT3);
    glDisable(GL_LIGHT4); glDisable(GL_LIGHT5);
  }
}

void setup_lights (void)
  // Specifies attributes of all lights, enables a subset of lights according to the lighting model
{
  GLfloat none[] = { 0.0, 0.0, 0.0, 1.0 };
  GLfloat low[] = { 0.15, 0.15, 0.15, 1.0 };
  GLfloat medium[] = { 0.5, 0.5, 0.5, 1.0 };
  GLfloat high[] = { 0.75, 0.75, 0.75, 1.0 };

  // Lights 0 and 1 are for the dynamic lighting model, with the lights fixed in the viewer's reference frame
  glLightfv(GL_LIGHT0, GL_AMBIENT, none);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, high);
  glLightfv(GL_LIGHT0, GL_SPECULAR, none);
  glLightfv(GL_LIGHT0, GL_POSITION, top_right);
  glLightfv(GL_LIGHT1, GL_AMBIENT, none);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, medium);
  glLightfv(GL_LIGHT1, GL_SPECULAR, none);
  glLightfv(GL_LIGHT1, GL_POSITION, straight_on);

  // Lights 2 and 3 are for the static lighting model, with the lights fixed in the planetary reference frame
  glLightfv(GL_LIGHT2, GL_AMBIENT, none);
  glLightfv(GL_LIGHT2, GL_DIFFUSE, high);
  glLightfv(GL_LIGHT2, GL_SPECULAR, none);
  glLightfv(GL_LIGHT3, GL_AMBIENT, low);
  glLightfv(GL_LIGHT3, GL_DIFFUSE, none);
  glLightfv(GL_LIGHT3, GL_SPECULAR, none);

  // Lights 4 and 5 are for highlighting the lander with static lights, to avoid flat views with ambient illumination only
  glLightfv(GL_LIGHT4, GL_AMBIENT, none);
  glLightfv(GL_LIGHT4, GL_DIFFUSE, low);
  glLightfv(GL_LIGHT4, GL_SPECULAR, none);
  glLightfv(GL_LIGHT5, GL_AMBIENT, none);
  glLightfv(GL_LIGHT5, GL_DIFFUSE, low);
  glLightfv(GL_LIGHT5, GL_SPECULAR, none);

  enable_lights();
}

void glut_print (float x, float y, const string &s, void *font, Alignment align)
  // Prints string at location (x,y) in a bitmap font
  // NB: glutBitmapLength may not be available on systems without freeglut
  //     Please report the issue on Camtools if your compiler reports an error
{
  if (align == ALIGN_MID) {
    x = glutBitmapLength(font, (const unsigned char*)s.c_str())*-0.5 + x;
  } else if (align == ALIGN_RIGHT) {
    x = glutBitmapLength(font, (const unsigned char*)s.c_str())*-1.0 + x;
  }
  glRasterPos2f(x, y);
  for (unsigned short i = 0; i < s.length(); i++) glutBitmapCharacter(font, s[i]);
}

double atmospheric_density (const vector3d &pos)
  // Simple exponential model between surface and exosphere (around 200km), surface density is approximately 0.017 kg/m^3,
  // scale height is approximately 11km
{
  double alt;

  alt = pos.abs()-MARS_RADIUS;
  if ((alt > EXOSPHERE) || (alt < 0.0)) return 0.0;
  else return (0.017 * exp(-alt/11000.0));
}

void draw_dial (double cx, double cy, double val, string title, string units)
  // Draws a single instrument dial, position (cx, cy), value val, title
{
  int e = 0;
  double a = 0.0;
  ostringstream s;

  // Work out value mantissa and exponent
  if (val > 0.0) {
    e = max(floor(log10(val)), 1.0);
    a = val / pow(10.0, e);
  }

  // Draw dial ticks
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINES);
  for (int i=30; i<=330; i+=30) {
    glVertex2d(cx - OUTER_DIAL_RADIUS * sin(i*M_PI/180.0), cy - OUTER_DIAL_RADIUS * cos(i*M_PI/180.0));
    glVertex2d(cx - INNER_DIAL_RADIUS * sin(i*M_PI/180.0), cy - INNER_DIAL_RADIUS * cos(i*M_PI/180.0));
  }
  glEnd();

  // Draw dial needle
  glColor3f(0.0, 1.0, 1.0);
  glBegin(GL_LINES);
  glVertex2d(cx, cy);
  glVertex2d(cx - INNER_DIAL_RADIUS * sin((a*30+30)*M_PI/180.0), cy - INNER_DIAL_RADIUS * cos((a*30+30)*M_PI/180.0));
  glEnd();

  // Draw exponent indicator, value and title
  s.precision(val>=1.0?1:2);
  glColor3f(1.0, 1.0, 1.0);
  s.str(""); s << "x 10 ^ " << e << " " << units;
  glut_print(cx+2, cy+10, s.str(), GLUT_BITMAP_HELVETICA_10, ALIGN_MID);
  glut_print(cx+2, cy-OUTER_DIAL_RADIUS-15, title, GLUT_BITMAP_HELVETICA_10, ALIGN_MID);
  s.str(""); s << fixed << val << " " << units;
  glut_print(cx+2, cy-OUTER_DIAL_RADIUS-30, s.str(), GLUT_BITMAP_HELVETICA_10, ALIGN_MID);

  // Draw tick labels
  for (int i=0; i<=10; i++) {
    switch(i) {
    case 0:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) - 8, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 9, "0");
      break;
    case 1:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) - 7, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 6, "1");
      break;
    case 2:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) - 9, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 4, "2");
      break;
    case 3:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) - 9, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 1, "3");
      break;
    case 4:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) - 8, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) + 3, "4");
      break;
    case 5:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) - 3, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) + 4, "5");
      break;
    case 6:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) + 2, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) + 3, "6");
      break;
    case 7:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) + 4, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0), "7");
      break;
    case 8:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) + 3, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 4, "8");
      break;
    case 9:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) + 3, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 6, "9");
      break;
    case 10:
      glut_print(cx - OUTER_DIAL_RADIUS * sin((i*30+30)*M_PI/180.0) + 3, cy - OUTER_DIAL_RADIUS * cos((i*30+30)*M_PI/180.0) - 8, "10");
      break;
    }
  }
}

void draw_control_bar (double tlx, double tly, double val, double red, double green, double blue, string title)
  // Draws control bar, top left (tlx, tly), val (fraction, range 0-1), colour (red, green, blue), title
{
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINE_LOOP);
  glVertex2d(tlx, tly-20.0);
  glVertex2d(tlx+240.0, tly-20.0);
  glVertex2d(tlx+240.0, tly);
  glVertex2d(tlx, tly);
  glEnd();
  glut_print(tlx, tly-40, title);
  glColor3f(red, green, blue);
  glBegin(GL_QUADS);
  glVertex2d(tlx+1.5, tly-18.5);
  glVertex2d(tlx+1.5+236.5*val, tly-18.5);
  glVertex2d(tlx+1.5+236.5*val, tly-1.5);
  glVertex2d(tlx+1.5, tly-1.5);
  glEnd();
}

void draw_indicator_lamp (double tcx, double tcy, string off_text, string on_text, bool on)
  // Draws indicator lamp, top centre (tcx, tcy), appropriate text and background colour depending on on/off
{
  if (on) glColor3f(0.5, 0.0, 0.0);
  else glColor3f(0.0, 0.5, 0.0);
  glBegin(GL_QUADS);
  glVertex2d(tcx-73.5, tcy-18.5);
  glVertex2d(tcx+73.5, tcy-18.5);
  glVertex2d(tcx+73.5, tcy-1.5);
  glVertex2d(tcx-73.5, tcy-1.5);
  glEnd();
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINE_LOOP);
  glVertex2d(tcx-75.0, tcy-20.0);
  glVertex2d(tcx+75.0, tcy-20.0);
  glVertex2d(tcx+75.0, tcy);
  glVertex2d(tcx-75.0, tcy);
  glEnd();
  if (on) glut_print(tcx-70.0, tcy-14.0, on_text);
  else glut_print(tcx-70.0, tcy-14.0, off_text);
}

void draw_instrument_window (void)
  // Draws the instruments
{
  ostringstream s;

  s.precision(1);
  glutSetWindow(instrument_window);
  glClear(GL_COLOR_BUFFER_BIT);

  // Draw altimeter
  draw_dial (view_width+GAP-400, INSTRUMENT_HEIGHT/2, altitude, "Altitude", "m");

  // Draw auto-pilot lamp
  instrument_auto->render(gui_focus == instrument_auto);

  // Draw climb rate meter
  if (climb_speed >= 0.0) draw_dial (view_width+GAP-150, INSTRUMENT_HEIGHT/2, landed ? 0.0 : climb_speed, "Climb rate", "m/s");
  else draw_dial (view_width+GAP-150, INSTRUMENT_HEIGHT/2, landed ? 0.0 : -climb_speed, "Descent rate", "m/s");

  // Draw attitude stabilizer lamp
  instrument_att->render(gui_focus == instrument_att);

  // Draw ground speed meter
  draw_dial (view_width+GAP+100, INSTRUMENT_HEIGHT/2, landed ? 0.0 : ground_speed, "Ground speed", "m/s");

  // Draw parachute lamp
  switch (parachute_status) {
  case NOT_DEPLOYED:
    instrument_para->col1 = vector3d(0.2,0.2,0.2); instrument_para->text   = "Parachute not deployed";
    break;
  case DEPLOYED:
    instrument_para->col1 = vector3d(0.0,0.5,0.0); instrument_para->text   = "Parachute deployed";
    break;
  case LOST:
    instrument_para->col1 = vector3d(0.5,0.0,0.0); instrument_para->text   = "Parachute lost";
    break;
  }
  instrument_para->render(gui_focus == instrument_para);

  // Draw speed bar
  draw_control_bar(view_width+GAP+240, INSTRUMENT_HEIGHT-18, simulation_speed/10.0, 0.0, 0.0, 1.0, "Simulation speed");
  
  // Draw digital clock
  glColor3f(1.0, 1.0, 1.0);
  s.str(""); s << "Time " << fixed << simulation_time << " s";
  glut_print(view_width+GAP+400, INSTRUMENT_HEIGHT-58, s.str());
  if (paused) {
    glColor3f(1.0, 0.0, 0.0);
    glut_print(view_width+GAP+338, INSTRUMENT_HEIGHT-32, "PAUSED");
  }

  // Display coordinates
  glColor3f(1.0, 1.0, 1.0);
  s.str(""); s << "x position " << fixed << position.x << " m";
  glut_print(view_width+GAP+240, INSTRUMENT_HEIGHT-97, s.str());
  s.str(""); s << "velocity " << fixed << velocity_from_positions.x << " m/s";
  glut_print(view_width+GAP+380, INSTRUMENT_HEIGHT-97, s.str());
  s.str(""); s << "y position " << fixed << position.y << " m";
  glut_print(view_width+GAP+240, INSTRUMENT_HEIGHT-117, s.str());
  s.str(""); s << "velocity " << fixed << velocity_from_positions.y << " m/s";
  glut_print(view_width+GAP+380, INSTRUMENT_HEIGHT-117, s.str());
  s.str(""); s << "z position " << fixed << position.z << " m";
  glut_print(view_width+GAP+240, INSTRUMENT_HEIGHT-137, s.str());
  s.str(""); s << "velocity " << fixed << velocity_from_positions.z << " m/s";
  glut_print(view_width+GAP+380, INSTRUMENT_HEIGHT-137, s.str());

  // Draw thrust bar
  //   Bar width set by average throttle
  //   Displayed number is net force from thrusters
  double ta = 0.0;
  for (int i=0; i<N_THRUSTERS; i++) ta += throttle[i];
  s.str(""); s << "Thrust " << fixed << thrust_wrt_world().abs() << " N";
  draw_control_bar(view_width+GAP+240, INSTRUMENT_HEIGHT-170, ta/N_THRUSTERS, 1.0, 0.0, 0.0, s.str());

  // Draw fuel bar
  s.str(""); s << "Fuel " << fixed << fuel*FUEL_CAPACITY << " litres";
  if (fuel > 0.5) draw_control_bar(view_width+GAP+240, INSTRUMENT_HEIGHT-242, fuel, 0.0, 1.0, 0.0, s.str());
  else if (fuel > 0.2) draw_control_bar(view_width+GAP+240, INSTRUMENT_HEIGHT-242, fuel, 1.0, 0.5, 0.0, s.str());
  else draw_control_bar(view_width+GAP+240, INSTRUMENT_HEIGHT-242, fuel, 1.0, 0.0, 0.0, s.str());

  // Display simulation status
  if (landed) glColor3f(1.0, 1.0, 0.0);
  else glColor3f(1.0, 1.0, 1.0);
  s.str(""); s << "Scenario " << scenario;
  if (!landed) s << ": " << scenario_description[scenario];
  glut_print(view_width+GAP-488, 17, s.str());
  if (landed) {
    if (altitude < LANDER_SIZE/2.0) glut_print(80, 17, "Lander is below the surface!");
    else {
      s.str(""); s << "Fuel consumed " << fixed << FUEL_CAPACITY*(1.0-fuel) << " litres";
      glut_print(view_width+GAP-427, 17, s.str());
      s.str(""); s << "Descent rate at touchdown " << fixed << -climb_speed << " m/s";
      glut_print(view_width+GAP-232, 17, s.str());
      s.str(""); s << "Ground speed at touchdown " << fixed << ground_speed << " m/s";
      glut_print(view_width+GAP+16, 17, s.str());
    }
  }

  glutSwapBuffers();
}

void display_help_arrows (void)
  // Displays help arrow in close-up view window
{

  double s = -closeup_offset/50.0;

  glDisable(GL_LIGHTING);
  glColor3f(1.0, 1.0, 1.0);
  glLineWidth(1.0);

  // Surface arrow
  glPushMatrix();
  look_in_direction(position);
  glBegin(GL_LINES);
    glVertex3d(0.0, 0.0, 2.0*s);
    glVertex3d(0.0, 0.0, 6.0*s);
  glEnd();
  glTranslated(0.0, 0.0, 6.0*s);
  glutCone(0.2*s, 0.5*s, 5, 5, true);
  glTranslated(0.0, 0.0, 1.0*s);
  glut_print(0.0, 0.0, "surface");
  glPopMatrix();

  // Ground speed arrow
  if ((ground_speed > MAX_IMPACT_GROUND_SPEED) && !landed) {
    vector3d np = position.norm();
    vector3d tv = velocity_from_positions - (velocity_from_positions * np) * np;
    glPushMatrix();
    look_in_direction(-tv);
    glBegin(GL_LINES);
      glVertex3d(0.0, 0.0, 2.0*s);
      glVertex3d(0.0, 0.0, 6.0*s);
    glEnd();
    glTranslated(0.0, 0.0, 6.0*s);
    glutCone(0.2*s, 0.5*s, 5, 5, true);
    glTranslated(0.0, 0.0, 1.0*s);
    glut_print(0.0, 0.0, "ground speed");
    glPopMatrix();
  }

  glEnable(GL_LIGHTING);
}

void display_help_text (void)
  // Displays help information in orbital view window
{
  ostringstream s;

  int x = 20, y = view_height - 20;

  glColor3f(1.0, 1.0, 1.0);

  glut_print(x, y, "Right/Left arrow keys - increase/decrease simulation speed");y -= 15;
  glut_print(x, y, "Space - single step through simulation");y -= 20;

  glut_print(x, y, "Up/Down arrow keys - increase/decrease thrust");y -= 20;

  glut_print(x, y, "Left mouse - rotate 3D views");y -= 15;
  glut_print(x, y, "Middle/shift mouse/up wheel - zoom in 3D views");y -= 15;
  glut_print(x, y, "Right mouse/down wheel - zoom out 3D views");y -= 20;

  glut_print(x, y, "a - toggle autopilot");y -= 15;
  glut_print(x, y, "s - toggle attitude stabilizer");y -= 15;
  glut_print(x, y, "p - deploy parachute");y -= 20;

  glut_print(x, y, "l - toggle lighting mode");y -= 15;
  glut_print(x, y, "t - toggle terrain texture");y -= 15;
  glut_print(x, y, "Esc/q - quit");y -= 20;

  glut_print(x, y, "Keys 0-9 - start simulation in scenario n");y -= 15;
  glut_print(x, y, "r - restart current scenario");y -= 20;

  // If the view is high enough, leave a gap
  y = min(y, 10*15 + 20);

  for (unsigned short i=0; i<10; i++) {
    s.str("");
    s << "Scenario " << i << ": " << scenario_description[i];
    glut_print(x, y, s.str());
    y -= 15;
  }
}

void display_help_prompt (void)
  // Displays help prompt in the close-up view window
{
  string ss = "press 'h' for help";
  unsigned short i;
  unsigned long long t;
  float c;

  microsecond_time(t);
  c = 1.0 - (t - time_program_started) / 3000000.0;
  if (c < 0.0) return;
  glColor4f(1.0, 1.0, 1.0, c);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, view_width, 0, view_height, -1.0, 1.0); 
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);

  glRasterPos2f(view_width/2 - 87, view_height-130);
  for (i = 0; i < ss.length(); i++) glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ss[i]);

  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void draw_orbital_window (void)
  // Draws the orbital view
{
  unsigned short i, j;
  double m[16], sf;
  GLint slices, stacks;

  // Viewing transformation
  quat_to_matrix(m, orbital_quat);
  glMultMatrixd(m);
  if (orbital_zoom > 2.0) { // gradual pan towards the lander when zoomed in
    sf = 1.0 - exp((2.0-orbital_zoom)/2.0);
    glTranslated(XYZ(-sf*position));
  }

  if (static_lighting) {
    // Specify light positions here, to fix them in the world coordinate system
    glLightfv(GL_LIGHT2, GL_POSITION, minus_y);
    glLightfv(GL_LIGHT3, GL_POSITION, plus_y);
  }

  // Draw planet
  glPushMatrix();
  glRotated(360.0*simulation_time/MARS_DAY, 0.0, 0.0, 1.0); // to make the planet spin
  if (orbital_zoom > 0.5) {
    slices = (int)(24*orbital_zoom); if (slices > 160) slices = 160;
    stacks = (int)(16*orbital_zoom); if (stacks > 100) stacks = 100;
  } else {
    slices = 12; stacks = 8;
  }
  if (do_texture) {
    glColor3f(0.70, 0.60, 0.60);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, planet_texture2);
    drawTexturedSphere(MARS_RADIUS, slices, stacks);
    glDisable(GL_TEXTURE_2D);
  } else {
    glColor3f(0.63, 0.33, 0.22);
    glLineWidth(1.0);
    gluQuadricDrawStyle(quadObj, GLU_FILL);
    gluSphere(quadObj, MARS_RADIUS, slices, stacks);

    glColor3f(0.31, 0.16, 0.11);
    gluQuadricDrawStyle(quadObj, GLU_LINE);
    gluSphere(quadObj, (1.0 + 0.01/orbital_zoom)*MARS_RADIUS, slices, stacks);
  }

  glPopMatrix();

  // Draw previous lander positions in cyan that fades with time
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glLineWidth(1.0);
  glBegin(GL_LINE_STRIP);
  glColor3f(0.0, 1.0, 1.0);
  glVertex3d(XYZ(position));
  j = (track.p+N_TRACK-1)%N_TRACK;
  for (i=0; i<track.n; i++) {
    glColor4f(0.0, 0.75*(N_TRACK-i)/N_TRACK, 0.75*(N_TRACK-i)/N_TRACK, 1.0*(N_TRACK-i)/N_TRACK);
    glVertex3d(XYZ(track.pos[j])); 
    j = (j+N_TRACK-1)%N_TRACK;
  }
  glEnd();
  glDisable(GL_BLEND);

  // Draw lander as a cyan dot
  glColor3f(0.0, 1.0, 1.0);
  glPointSize(3.0);
  glBegin(GL_POINTS);
  glVertex3d(XYZ(position));
  glEnd();
  glEnable(GL_LIGHTING);

}

void draw_multi_window (void)
{
  glutSetWindow(multi_window);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  switch (multi_view_state) {
    case VIEW_MENU:
    case VIEW_PROP:
    case VIEW_AUTO:
      break;
    case VIEW_ORBITAL:
      draw_orbital_window();
      break;
#ifdef GRAPHING
    case VIEW_GRAPH:
      draw_graph_window();
      break;
#endif
    case VIEW_INSTRUMENT:
    case VIEW_INVALID:
      cerr << "Error: Can not render invalid multi-view window" << endl;
      exit(1);
  }

  if (multi_view_state == VIEW_ORBITAL) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, view_width, 0, view_height, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
  }

  // Help information
  if (help) display_help_text();

  draw_gui();

  if (multi_view_state == VIEW_ORBITAL) {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
  }

  glutSwapBuffers();
}

#ifdef GRAPHING

void log_value (float value, unsigned int series)
{
  // Store the value in the appropriate vector
  if (n_points < N_POINTS) {
    // Add the new value to the end of the vector (extending it by one)
    //graph_series[series].data.push_back(value);
    graph_series[series].data[n_points] = value;
  } else {
    // Add the new value to the vector (over-writing the oldest)
    //graph_series[series].data[n_start%N_POINTS] = value;
    graph_series[series].data[n_start%N_POINTS] = value;
    
    // Update the min/max statistics
    
    // Remove the oldest value if it falls outside of the time window
    if (!graph_series[series].maxima.empty()) {
      if (graph_series[series].maxima.front() == n_start%N_POINTS) {
        graph_series[series].maxima.pop_front();
      }
    }
    if (!graph_series[series].minima.empty()) {
      if (graph_series[series].minima.front() == n_start%N_POINTS) {
        graph_series[series].minima.pop_front();
      }
    }
  }
  
  // Remove maxima less than the new value
  while (!graph_series[series].maxima.empty()) {
    if (graph_series[series].data[graph_series[series].maxima.back()] <= value) {
      graph_series[series].maxima.pop_back();
    } else break;
  }
  // Remove minima greater than the new value
  while (!graph_series[series].minima.empty()) {
    if (graph_series[series].data[graph_series[series].minima.back()] >= value) {
      graph_series[series].minima.pop_back();
    } else break;
  }
  
  // Add the new value to the end of the queue
  graph_series[series].maxima.push_back((n_points+n_start)%N_POINTS);
  graph_series[series].minima.push_back((n_points+n_start)%N_POINTS);
}

vector3d graph_colour (unsigned int series, bool set)
 // Set OpenGL colour for plotting a series in the graph view
{
  // Plot colours
  const float cols[] = { 0.9, 0.0, 0.0,  // Red
                         0.8, 0.4, 0.0,  // Orange
                         0.8, 0.8, 0.0,  // Yellow
                         0.0, 0.8, 0.0,  // Green
                         0.0, 0.3, 0.0,  // Dark Green
                         0.0, 0.8, 0.8,  // Cyan
                         0.0, 0.0, 0.9,  // Blue
                         0.8, 0.0, 0.8,  // Purple
                         0.4, 0.0, 0.8   // Lilac
                       };

  vector3d col(cols[series*3+0], cols[series*3+1], cols[series*3+2]);
  if (set) glColor3f(XYZ(col));
  return col;
}

void draw_graph_window (void)
{
  // Padding around edge for axis labels
  const double PAD_L = 25.0, PAD_T = 20.0, PAD_R = 20.0, PAD_B = 70.0;

  float data_min_y = 1e50, data_max_y = -1e50;

  double tick_step_x, tick_step_y;

  // Are any traces set to be displayed
  bool visible = false;

  // Find the minimum and maximum data values of the visible traces
  for (unsigned int i=0; i<N_SERIES; i++) {
    if (graph_series[i].visible) {
      visible = true;
      float tmin = graph_series[i].data[graph_series[i].minima.front()];
      if (tmin < data_min_y) data_min_y = tmin;
      float tmax = graph_series[i].data[graph_series[i].maxima.front()];
      if (tmax > data_max_y) data_max_y = tmax;
    }
  }

  // If no series are visible, or there is only one point to draw, do not plot a graph
  if (!visible || n_points < 2) {
    glColor3f(1.0, 1.0, 1.0);
    glut_print(view_width*0.5, view_height*0.5-5, "No data to plot", GLUT_BITMAP_HELVETICA_12, ALIGN_MID);
    return;
  }

  double plot_height = view_height - (PAD_B+PAD_T), plot_width = view_width - (PAD_L+PAD_R);
  double plot_min_y, plot_max_y;
  double exponent_y = 0.0, exponent_x = 0.0;

  // Default to x-axis at half of view height
  double axis_height = 0.0;

  // If data is either all positive or all negative, shift axes fully
  // Otherwise, shift proportionally by min/max amplitudes
  // If data is all zeros, display the range as -1 to +1
  if (data_min_y == 0.0 && data_max_y == 0.0) {
    data_min_y = -1.0;
    data_max_y = 1.0;
  } else if (data_min_y >= 0.0) {
    data_min_y = 0.0;
  } else if (data_max_y <= 0.0) {
    data_max_y = 0.0;
  }

  double range_y = data_max_y - data_min_y;
  exponent_y = floor(log10(range_y));
  tick_step_y = 1.0 * pow(10.0, exponent_y);
  int Nt_y = floor(range_y / tick_step_y) + 1;
  if (Nt_y < 5) {
    tick_step_y *= 0.5;
    Nt_y = floor(range_y / tick_step_y) + 1;
    if (Nt_y < 5) {
      tick_step_y *= 0.4;
    }
  }
  plot_min_y = -ceil(-data_min_y / tick_step_y) * tick_step_y;
  plot_max_y = ceil(data_max_y / tick_step_y) * tick_step_y;
  axis_height = plot_height * abs(-plot_min_y/(plot_max_y-plot_min_y));

  // Determine plot limits in x axis (time)
  double plot_min_x = n_start * delta_t;
  double plot_max_x = (n_start + n_points - 1) * delta_t;
  exponent_x = max(floor(log10(n_points * delta_t)), 0.0);
  tick_step_x = 1.0 * pow(10.0, exponent_x);
  int Nt_x = floor((n_points * delta_t) / tick_step_x) + 1;
  if (Nt_x < 5) {
    tick_step_x *= 0.5;
    Nt_x = floor((n_points * delta_t) / tick_step_x) + 1;
    if (Nt_x < 5) {
      tick_step_x *= 0.4;
    }
  }

  double tick_start_x = tick_step_x;

  // If time axis starts at 0, ensure it extends beyond 1s
  if (plot_min_x == 0) {
    plot_max_x = max(plot_max_x, 1.0);
  } else {
    // Shift the minimum axis value by integer multiples of the tick spacing
    // to give the sliding-window effect
    tick_start_x = ceil(plot_min_x / tick_step_x) * tick_step_x;
  }

  // Calculate scaling factors for each axis
  double xfac = (plot_width * delta_t) / (plot_max_x - plot_min_x);
  double yfac = plot_height / abs(plot_max_y - plot_min_y);

  // Draw faint grid lines in the background
  glColor3f(0.15, 0.15, 0.15);
  glBegin(GL_LINES);
  // Vertical lines
  for (double x = (tick_start_x-plot_min_x)*xfac/delta_t; x<=plot_width+1.0; x += tick_step_x*xfac/delta_t) {
    glVertex2d(PAD_L+x, PAD_B);
    glVertex2d(PAD_L+x, PAD_B+plot_height);
  }
  // Horizontal lines
  for (double y=0.0; y<=plot_height+1.0; y+=tick_step_y*yfac) {
    glVertex2d(PAD_L,            PAD_B+y);
    glVertex2d(PAD_L+plot_width, PAD_B+y);
  }
  glEnd();

  // Draw major axes
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINES);
    // Y-axis
    glVertex2d(PAD_L, PAD_B);
    glVertex2d(PAD_L, PAD_B+plot_height);
    // X-axis
    glVertex2d(PAD_L,            PAD_B+axis_height);
    glVertex2d(PAD_L+plot_width, PAD_B+axis_height);
  glEnd();

  // For converting numbers into strings
  ostringstream ss;
  ss.str("");

  if (exponent_y != 0.0) {
    ss << "x10^" << exponent_y;
    glut_print(PAD_L+10, (axis_height>plot_height/2)?PAD_B-5:view_height-12, ss.str());
    ss.str("");
  }

  exponent_x = floor(log10(plot_max_x));
  ss << "Time (s)";
  if (exponent_x >= 2.0) {
    ss << " x10^" << exponent_x;
  } else {
    exponent_x = 0;
  }
  glut_print(PAD_L+plot_width*0.5, PAD_B+axis_height-26, ss.str(), GLUT_BITMAP_HELVETICA_10, ALIGN_MID);
  ss.str("");

  // Draw numbers along the axes
  for (double y=plot_min_y; y <= plot_max_y; y += tick_step_y) {
    ss << y / pow(10.0, exponent_y);
    glut_print(PAD_L-2, PAD_B+axis_height+y*yfac-3, ss.str(), GLUT_BITMAP_HELVETICA_10, ALIGN_RIGHT);
    ss.str("");
  }

  // Increase number of digits shown for large numbers on x axis
  if (exponent_x > 4.0) ss.precision(3);
  else ss.precision(2);
  for (double x = tick_start_x; x <= plot_max_x; x += tick_step_x) {
    ss << x / pow(10.0, exponent_x);
    const double y = PAD_B+axis_height+((axis_height>plot_height/2)?4:-12);
    glut_print(PAD_L + ((x-plot_min_x)*xfac)/delta_t, y, ss.str(), GLUT_BITMAP_HELVETICA_10, ALIGN_MID);
    ss.str("");
  }

  // Transform data using OpenGL to fit axes
  glTranslated(PAD_L, PAD_B+axis_height, 0.0);
  glScaled(xfac, yfac, 1.0);
  glTranslated(-plot_min_x/delta_t, 0.0, 0.0);

  // When data is too dense for the view, do not plot every point (nearest-neighbour sampling)
  unsigned int stride = floor(n_points/plot_width) + 1.0;

  // --- Plot the data ---

  for (unsigned int series=0; series<N_SERIES; series++) {

    if (!graph_series[series].visible) continue;

    // Set a distinct colour for this series
    graph_colour(series);

    // Plot the data as a continuous line
    glBegin(GL_LINE_STRIP);
    float x = (float)n_start;
    float *d = &(graph_series[series].data[n_start%N_POINTS]);
    float *e = &(graph_series[series].data[N_POINTS]);
    for (unsigned long p=0; p<n_points; p+=stride) {
      glVertex2f(x, *d);
      x += (float)stride;
      d += stride;
      if (d >= e) d = &(graph_series[series].data[d-e]);
    }
    glEnd();
  }

  // Reset the transformation matrix
  glLoadIdentity();

}
#endif

void draw_gui () {
  for (vector<GuiControl*>::iterator it=controls.begin(); it<controls.end(); it++) {
    if ((*it)->view == multi_view_state) (*it)->render((*it == gui_focus));
  }
}

void draw_parachute (double d)
  // OpenGL hemisphere to draw a simple parachute, distance d behind the lander
{
  glLineWidth(0.8);
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);

  double x0, x1, y0, y1, r0, z0, *sint1, *cost1, *sint2, *cost2;
  double radius = CHUTE_SIZE;

  fghCircleTable(&sint1, &cost1, 50);
  fghCircleTable(&sint2, &cost2, 100);

  glColor3f(0.4, 0.4, 0.4);
  glBegin(GL_LINES);
  glVertex3d(0.0, 0.0, 0.0);
  glVertex3d(0.0, 0.0, d/3.0);
  for (int i=0; i<49; i+=5) {
    glVertex3d(0.0, 0.0, d/3.0);
    glVertex3d(radius*sint1[i], radius*cost1[i], d);
  }
  glEnd();

  glEnable(GL_LIGHTING);

  x1 = cost1[0]; y1 = sint1[0];

  for (int i=0; i<50; i++) {
    x0 = x1; x1 = cost1[i+1];
    y0 = y1; y1 = sint1[i+1];

    if (i&1) glColor3f(0.9, 0.9, 0.95);
    else glColor3f(0.9, 0.7, 0.3);

    glBegin(GL_QUAD_STRIP);
    for (int j=25; j<49; j++) {
      r0 = sint2[j]; z0 = cost2[j];
      glNormal3d(x0*r0, y0*r0, -z0);
      glVertex3d(x0*r0*radius, y0*r0*radius, d-z0*radius);
      glNormal3d(x1*r0, y1*r0, -z0);
      glVertex3d(x1*r0*radius, y1*r0*radius, d-z0*radius);
    }
    glEnd();
  }

  free(sint1); free(cost1);
  free(sint2); free(cost2);

  glEnable(GL_CULL_FACE);
}

void update_closeup_coords (void)
  // Updates the close-up view's coordinate frame, based on the lander's current position and velocity.
  // This needs to be called every time step, even if the view is not being rendered, since any-angle
  // attitude stabilizers reference closeup_coords.right
{

  vector3d t;

  // Vector from planet center to lander - maps to the view y-axis
  vector3d s = position.norm();

  // Calculate component of velocity perpendicular to s
  vector3d tv = velocity_from_positions - (velocity_from_positions * s) * s;

  // At simulation start, construct a coordinate system
  // based on lander position and velocity
  if (!closeup_coords.initialized) {

    // If tangential velocity is reasonable, use it to give the x-axis
    if (tv.abs() > SMALL_NUM) {
      t = tv;
    } else {
      // Otherwise, choose an arbitrary vector perpendicular to s
      t = vector3d(-s.y, s.x, 0.0);
      if (t.abs() < SMALL_NUM) t = vector3d(-s.z, 0.0, s.x);
    }

    closeup_coords.initialized = true;

  } else {

    // Remove any component not orthogonal to s, and renormalize
    // This is safe, as we do not expect the lander position to change dramatically between timesteps
    t = closeup_coords.right - (closeup_coords.right * s) * s;

  }

  closeup_coords.up    = s;
  closeup_coords.right = t.norm();

  // Adjust the terrain texture angle so that it matches the lander velocity
  {
    double tmp = closeup_coords.right * tv.norm();
    if (tmp > 1.0) tmp = 1.0; else if (tmp < -1.0) tmp = -1.0;

    terrain_angle = (180.0/M_PI)*acos(tmp);
    if ((-closeup_coords.right^tv.norm())*s < 0.0) terrain_angle *= -1.0;

    while (terrain_angle < 0.0) terrain_angle += 360.0;
    while (terrain_angle >= 360.0) terrain_angle -= 360.0;
  }

}

void draw_closeup_window (void)
  // Draws the close-up view of the lander
{
  static double terrain_offset_x = 0.0;
  static double terrain_offset_y = 0.0;
  static double ground_line_offset = 0.0;
  static double last_redraw_time = 0.0;
  static unsigned short rn = 0;
  vector3d s, t, n;
  double lander_drag, chute_drag, glow_factor, aspect_ratio, view_depth, f, tmp;
  double horizon, fog_density, cx, cy, m[16], m2[16], transition_altitude, ground_plane_size;
  unsigned short i, j, rtmp;
  GLfloat fogcolour[4];
  bool dark_side;
  float rand_tri[8];

  glutSetWindow(closeup_window);
  aspect_ratio = (double)view_width/view_height;
  if (do_texture) transition_altitude = TRANSITION_ALTITUDE;
  else transition_altitude = TRANSITION_ALTITUDE_NO_TEXTURE;
  ground_plane_size = 5.0*transition_altitude;

  // Work out an atmospheric haze colour based on prevailing atmospheric density. The power law in the
  // expression below couples with the fog calculation further down, to ensure that the fog doesn't dim
  // the scene on the way down.
  tmp = pow(atmospheric_density(position)/atmospheric_density(vector3d(MARS_RADIUS, 0.0, 0.0)), 0.5);
  if (static_lighting) tmp *= 0.5 * (1.0 + position.norm()*vector3d(0.0, -1.0, 0.0)); // set sky colour
  fogcolour[0] = tmp*0.98; fogcolour[1] = tmp*0.67; fogcolour[2] = tmp*0.52; fogcolour[3] = 0.0;
  glClearColor(tmp*0.98, tmp*0.67, tmp*0.52, 0.0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (altitude < 0.0) { // just blank the screen if the lander is below the surface
    glutSwapBuffers();
    return;
  }
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Set projection matrix and fog values based on altitude.
  // Above the exosphere we see a long way and there is no fog.
  // Between the exosphere and transition_altitude, we see up to the horizon with increasing fog.
  // At transition_altitude we have a totally opaque haze, to disguise the transition from spherical surface to flat surface.
  // Below transition_altitude, we can see as far as the horizon (or transition_altitude with no terrain texture), 
  // with the fog decreasing towards touchdown.
  if (altitude > EXOSPHERE) gluPerspective(CLOSEUP_VIEW_ANGLE, aspect_ratio, 1.0, closeup_offset + 2.0*MARS_RADIUS);
  else {
    horizon = sqrt(position.abs2() - MARS_RADIUS*MARS_RADIUS);
    if (altitude > transition_altitude) {
      f = (altitude-transition_altitude) / (EXOSPHERE-transition_altitude);
      if (f < SMALL_NUM) fog_density = 1000.0; else fog_density = (1.0-f) / (f*horizon);
      view_depth = closeup_offset + horizon;
    } else {
      f = 1.0 - (altitude / transition_altitude);
      if (f < SMALL_NUM) fog_density = 1000.0; else fog_density = (1.0-f) / (f*transition_altitude);
      if (do_texture) {
        fog_density = 0.00005 + 0.5*fog_density;
        view_depth = closeup_offset + horizon;
      } else view_depth = closeup_offset + transition_altitude;
    }
    gluPerspective(CLOSEUP_VIEW_ANGLE, aspect_ratio, 1.0, view_depth);
    glFogf(GL_FOG_DENSITY, fog_density);
    glFogfv(GL_FOG_COLOR, fogcolour);
    if (do_texture) glHint(GL_FOG_HINT, GL_NICEST);
    else glHint(GL_FOG_HINT, GL_FASTEST);
    glEnable(GL_FOG);
  }

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // The world coordinate system in this view is centered on the lander, with y-axis vertical
  // (from planet centre to lander) and x-axis parallel to the lander's tangential velocity.
  // We now need a modelling rotation transformation to map from this system to the planetary
  // coordinate system.

  // Direction from surface to lander (radial) - this must map to the world y-axis
  s = closeup_coords.up;

  // Direction of tangential velocity - this must map to the world x-axis
  t = closeup_coords.right; 

  // Mutual perpendicular to these two vectors - this must map to the world z-axis
  n = t^s;

  // Construct modelling matrix (rotation only) from these three vectors
  m[0] = t.x; m[1] = t.y; m[2] = t.z; m[3] = 0.0;
  m[4] = s.x; m[5] = s.y; m[6] = s.z; m[7] = 0.0;
  m[8] = n.x; m[9] = n.y; m[10] = n.z; m[11] = 0.0;
  m[12] = 0.0; m[13] = 0.0; m[14] = 0.0; m[15] = 1.0;
  invert(m, m2); 

  // Update terrain texture/line offsets
  if (simulation_time != last_redraw_time) {
    terrain_offset_x += cos(terrain_angle*M_PI/180.0) * ground_speed * (simulation_time-last_redraw_time) / (2.0*ground_plane_size);
    terrain_offset_y += sin(terrain_angle*M_PI/180.0) * ground_speed * (simulation_time-last_redraw_time) / (2.0*ground_plane_size);
    while (terrain_offset_x < 0.0) terrain_offset_x += 1.0;
    while (terrain_offset_x >= 1.0) terrain_offset_x -= 1.0;
    while (terrain_offset_y < 0.0) terrain_offset_y += 1.0;
    while (terrain_offset_y >= 1.0) terrain_offset_y -= 1.0;
    ground_line_offset -= ground_speed * (simulation_time-last_redraw_time);
    ground_line_offset -= GROUND_LINE_SPACING*((int)ground_line_offset/(int)(GROUND_LINE_SPACING));
    last_redraw_time = simulation_time;
  }

  // Viewing transformation
  glTranslated(0.0, 0.0, -closeup_offset);
  glRotated(closeup_xr, 1.0, 0.0, 0.0);
  glRotated(closeup_yr, 0.0, 1.0, 0.0);

  if (static_lighting) {
    // Specify light positions here, to fix them in the planetary coordinate system
    glPushMatrix();
    glMultMatrixd(m2); // now in the planetary coordinate system
    glLightfv(GL_LIGHT2, GL_POSITION, minus_y);
    glLightfv(GL_LIGHT3, GL_POSITION, plus_y);
    glLightfv(GL_LIGHT4, GL_POSITION, plus_y);
    glLightfv(GL_LIGHT5, GL_POSITION, plus_z);
    glPopMatrix(); // back to the view's world coordinate system
  }

  // Surface colour
  glColor3f(0.63, 0.33, 0.22);

  if (altitude < transition_altitude) {

    // Draw ground plane below the lander's current position - we need to do this in quarters, with a vertex
    // nearby, to get the fog calculations correct in all OpenGL implementations.
    glBindTexture(GL_TEXTURE_2D, terrain_texture);
    if (do_texture) glEnable(GL_TEXTURE_2D);
    glNormal3d(0.0, 1.0, 0.0);
    glBegin(GL_QUADS);
    glTexCoord2f(1.0 + terrain_offset_x, 1.0 + terrain_offset_y); glVertex3d(ground_plane_size, -altitude, ground_plane_size);      
    glTexCoord2f(1.0 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(ground_plane_size, -altitude, 0.0);
    glTexCoord2f(0.5 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(0.0, -altitude, 0.0);      
    glTexCoord2f(0.5 + terrain_offset_x, 1.0 + terrain_offset_y); glVertex3d(0.0, -altitude, ground_plane_size);
    glTexCoord2f(0.5 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(0.0, -altitude, 0.0);      
    glTexCoord2f(1.0 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(ground_plane_size, -altitude, 0.0);
    glTexCoord2f(1.0 + terrain_offset_x, 0.0 + terrain_offset_y); glVertex3d(ground_plane_size, -altitude, -ground_plane_size);
    glTexCoord2f(0.5 + terrain_offset_x, 0.0 + terrain_offset_y); glVertex3d(0.0, -altitude, -ground_plane_size);
    glTexCoord2f(0.5 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(0.0, -altitude, 0.0);      
    glTexCoord2f(0.5 + terrain_offset_x, 0.0 + terrain_offset_y); glVertex3d(0.0, -altitude, -ground_plane_size);
    glTexCoord2f(0.0 + terrain_offset_x, 0.0 + terrain_offset_y); glVertex3d(-ground_plane_size, -altitude, -ground_plane_size);
    glTexCoord2f(0.0 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(-ground_plane_size, -altitude, 0.0);
    glTexCoord2f(0.5 + terrain_offset_x, 1.0 + terrain_offset_y); glVertex3d(0.0, -altitude, ground_plane_size);
    glTexCoord2f(0.5 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(0.0, -altitude, 0.0);      
    glTexCoord2f(0.0 + terrain_offset_x, 0.5 + terrain_offset_y); glVertex3d(-ground_plane_size, -altitude, 0.0);
    glTexCoord2f(0.0 + terrain_offset_x, 1.0 + terrain_offset_y); glVertex3d(-ground_plane_size, -altitude, ground_plane_size);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    if (!do_texture) { // draw lines on the ground plane at constant x (to show ground speed)
      glEnable(GL_BLEND);
      glLineWidth(2.0);
      glPushMatrix();
      glRotated(-terrain_angle, 0.0, 1.0, 0.0);
      glBegin(GL_LINES);
      tmp = ground_line_offset + transition_altitude;
      while (tmp > -transition_altitude) {
        // Fade the lines out towards the horizon, to avoid aliasing artefacts. The fade is a function of distance from the
        // centre (tmp) and altitude: the lower the lander gets, the more pronounced the fade.
        // We need to do draw each line in two parts, with a vertex nearby, to get the fog calculations correct in all OpenGL implementations.
        // To make the lines fade more strongly when landed, decrease the second number.
        // To make the lines less apparent at high altitude, decrease the first number. 
        f = exp( -fabs( pow((transition_altitude-altitude) / transition_altitude, 10.0) * tmp / (10.0*GROUND_LINE_SPACING)) );
        glColor4f(0.32, 0.17, 0.11, f);
        glVertex3d(tmp, -altitude, -transition_altitude);
        glVertex3d(tmp, -altitude, 0.0);
        glVertex3d(tmp, -altitude, 0.0);
        glVertex3d(tmp, -altitude, transition_altitude);
        tmp -= GROUND_LINE_SPACING;
      }
      glEnd();
      glPopMatrix();
      glDisable(GL_BLEND);
    }

    if (!crashed) { // draw a circular shadow below the lander
      glColor3f(0.32, 0.17, 0.11);
      glBegin(GL_TRIANGLES);
      for (i=0; i<360; i+=10) {
        glVertex3d(0.0, -altitude, 0.0);
        glVertex3d(LANDER_SIZE*cos(M_PI*(i+10)/180.0), -altitude, LANDER_SIZE*sin(M_PI*(i+10)/180.0));
        glVertex3d(LANDER_SIZE*cos(M_PI*i/180.0), -altitude, LANDER_SIZE*sin(M_PI*i/180.0));
      }
      glEnd();
    } else {
      rtmp = 0;
      glColor3f(1.0, 1.0, 1.0);
      glBegin(GL_TRIANGLES); // draw some shards of metal
      for (i=0; i<60; i++) {
        for (j=0; j<8; j++) { rand_tri[j] = randtab[rtmp]; rtmp = (rtmp+1)%N_RAND; }
        cx = 40.0 * (rand_tri[0] - 0.5);
        cy = 40.0 * (rand_tri[1] - 0.5);
        glNormal3d(0.0, 1.0, 0.0);
        glVertex3d(cx + 2.0*LANDER_SIZE*rand_tri[2], -altitude, cy + 2.0*LANDER_SIZE*rand_tri[3]);
        glVertex3d(cx + 2.0*LANDER_SIZE*rand_tri[4], -altitude, cy + 2.0*LANDER_SIZE*rand_tri[5]);
        glVertex3d(cx + 2.0*LANDER_SIZE*rand_tri[6], -altitude, cy + 2.0*LANDER_SIZE*rand_tri[7]);
      }
      glEnd();
      if (parachute_status != LOST) {
        glColor3f(1.0, 1.0, 0.0);
        glBegin(GL_TRIANGLES);  // draw some shreds of yellow canvas
        for (i=0; i<30; i++) {
          for (j=0; j<8; j++) { rand_tri[j] = randtab[rtmp]; rtmp = (rtmp+1)%N_RAND; }
          cx = 40.0 * (rand_tri[0] - 0.5);
          cy = 40.0 * (rand_tri[1] - 0.5);
          glNormal3d(0.0, 1.0, 0.0);
          glVertex3d(cx + 2.0*LANDER_SIZE*rand_tri[2], -altitude, cy + 2.0*LANDER_SIZE*rand_tri[3]);
          glVertex3d(cx + 2.0*LANDER_SIZE*rand_tri[4], -altitude, cy + 2.0*LANDER_SIZE*rand_tri[5]);
          glVertex3d(cx + 2.0*LANDER_SIZE*rand_tri[6], -altitude, cy + 2.0*LANDER_SIZE*rand_tri[7]);
        }
        glEnd();
      }
    }
    glEnable(GL_DEPTH_TEST);
  
  } else {

    // Draw spherical planet - can disable depth test (for speed)
    glDisable(GL_DEPTH_TEST);
    glPushMatrix();

    if (altitude > EXOSPHERE) {

      // Draw the planet reduced size at a reduced displacement, to avoid numerical OpenGL problems with huge viewing distances.
      glTranslated(0.0, -MARS_RADIUS, 0.0);
      glMultMatrixd(m2); // now in the planetary coordinate system
      glRotated(360.0*simulation_time/MARS_DAY, 0.0, 0.0, 1.0); // to make the planet spin
      double r = MARS_RADIUS * (MARS_RADIUS / (altitude + MARS_RADIUS));
      if (do_texture) {
        glColor3f(0.70, 0.60, 0.60);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, planet_texture1);
        drawTexturedSphere(r, 160, 100);
        glDisable(GL_TEXTURE_2D);
      } else {
        glutMottledSphere(r, 160, 100);
      }

    } else {

      // Draw the planet actual size at the correct displacement
      glTranslated(0.0, -(MARS_RADIUS + altitude), 0.0);
      glMultMatrixd(m2); // now in the planetary coordinate system
      glRotated(360.0*simulation_time/MARS_DAY, 0.0, 0.0, 1.0); // to make the planet spin
      if (do_texture) {
        glColor3f(0.70, 0.60, 0.60);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, planet_texture1);
        drawTexturedSphere(MARS_RADIUS, 160, 100);
        glDisable(GL_TEXTURE_2D);
      } else {
        glutMottledSphere(MARS_RADIUS, 160, 100);
      }

    }

    glPopMatrix(); // back to the view's world coordinate system
    glEnable(GL_DEPTH_TEST);

  }

  glDisable(GL_FOG); // fog only applies to the ground
  dark_side = (static_lighting && (position.y > 0.0) && (sqrt(position.x*position.x + position.z*position.z) < MARS_RADIUS));
  if (dark_side) { // in the shadow of the planet, we need some diffuse lighting to highlight the lander
    glDisable(GL_LIGHT2); glDisable(GL_LIGHT3); 
    glEnable(GL_LIGHT4); glEnable(GL_LIGHT5);
  }

  // Work out drag on lander - if it's high, we will surround the lander with an incandescent glow. Also
  // work out drag on parachute: if it's zero, we will not draw the parachute fully open behind the lander.
  // Assume high Reynolds number, quadratic drag = -0.5 * rho * v^2 * A * C_d
  lander_drag = 0.5*DRAG_COEF_LANDER*atmospheric_density(position)*M_PI*LANDER_SIZE*LANDER_SIZE*velocity_from_positions.abs2();
  chute_drag = 0.5*DRAG_COEF_CHUTE*atmospheric_density(position)*M_PI*CHUTE_SIZE*CHUTE_SIZE*velocity_from_positions.abs2();

  // Switch to the planetary coordinate system
  glPushMatrix();
  glMultMatrixd(m2);

  // Display help arrows to show surface direction and tangential velocity
  if (help && !(landed || crashed)) display_help_arrows();
  else display_help_prompt();

  // Lander orientation relative to planetary coordinate system - unit quaternion
  quat_to_matrix(m, orientation);
  invert(m, m2);
  glMultMatrixd(m2);

  // Draw lander
  if (!crashed) {

    // Draw the lander's parachute - behind the lander in the direction of travel
    if (parachute_status == DEPLOYED) {
    
      glPushMatrix();
      glTranslated(XYZ(parachute_attachment));
      
      if (velocity_from_positions.abs() < SMALL_NUM || altitude > EXOSPHERE) {
        // Lander is apparently stationary - so draw the parachute above and near to the lander
        tmp = 1.5*2.0;
      } else {
        if (chute_drag) tmp = 1.5*5.0; // parachute fully open
        else tmp = 1.5*2.0; // parachute not fully open
        look_in_direction(matrix_mult(m, -velocity_from_positions));
      }

      draw_parachute(tmp);
      glPopMatrix();
    }

    // Draw the lander body
    draw_lander();

    for (int i=0; i<N_THRUSTERS; i++) {

      glPushMatrix();
      glTranslated(XYZ(thruster_position[i]));
      look_in_direction(thruster_direction[i]);

      glColor3f(0.2, 0.2, 0.2);
      draw_frustrum(THRUSTER_SIZE, THRUSTER_SIZE*0.5, THRUSTER_SIZE, 10, 10, true);

      double scale = 1.6*LANDER_SIZE*throttle[i];
      scale *= 0.7 + 0.3*randtab[rn]; rn = (rn+1)%N_RAND; // a little random variation for added realism

      if (scale) {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glTranslated(0.0, 0.0, THRUSTER_SIZE);
        draw_flare(THRUSTER_SIZE*0.4, scale+0.01, 6, 10);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
      }

      glPopMatrix();

    }
  }

  if (dark_side) { // back to standard lighting model
    glEnable(GL_LIGHT2); glEnable(GL_LIGHT3);
    glDisable(GL_LIGHT4); glDisable(GL_LIGHT5);
  }

  glPopMatrix(); // back to the world coordinate system

  // Draw incandescent glow surrounding lander
  if (lander_drag*velocity_from_positions.abs() > HEAT_FLUX_GLOW_THRESHOLD) {
    // Calculate an heuristic "glow factor", in the range 0 to 1, for graphics effects
    glow_factor = (lander_drag*velocity_from_positions.abs()-HEAT_FLUX_GLOW_THRESHOLD) / (4.0*HEAT_FLUX_GLOW_THRESHOLD);
    if (glow_factor > 1.0) glow_factor = 1.0;
    glow_factor *= 0.7 + 0.3*randtab[rn]; rn = (rn+1)%N_RAND; // a little random variation for added realism
    glRotated((180.0/M_PI)*atan2(climb_speed, ground_speed), 0.0, 0.0, 1.0);
    glRotated(-90.0, 0.0, 1.0, 0.0);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glColor4f(1.0, glow_factor, 0.0, 0.8*glow_factor);
    glutCone(1.25*LANDER_SIZE, (2.0 + 10.0*glow_factor)*LANDER_SIZE, 50, 50+(int)(250*glow_factor), false);
    glutOpenHemisphere(1.25*LANDER_SIZE, 50, 50);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
  }

  glutSwapBuffers(); 
}

void draw_main_window (void)
  // Draw grey lines to partition the display into three sub-windows
{
  glutSetWindow(main_window);
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glColor3f(0.5, 0.5, 0.5);
  glBegin(GL_LINES);
  glVertex2i(view_width + 2*GAP, INSTRUMENT_HEIGHT + 2*GAP);
  glVertex2i(view_width + 2*GAP, win_height);
  glVertex2i(0, INSTRUMENT_HEIGHT + 2*GAP);
  glVertex2i(win_width, INSTRUMENT_HEIGHT + 2*GAP);
  glEnd();
  glutSwapBuffers();
}

void refresh_all_subwindows (void)
  // Marks all subwindows as needing a redraw every n times called, where n depends on the simulation speed
{
  static unsigned short n = 0;

  if (simulation_speed > 5) {
    switch (simulation_speed) {
    case 6:
      n += 500; // update graphics every 2 iterations
      break;
    case 7:
      n += 100; // update graphics every 10 iterations
      break;
    case 8:
      n += 20; // update graphics every 50 iterations
      break;
    case 9:
      n += 10;  // update graphics every 100 iterations
      break;
    case 10:
      n += 1;  // update graphics every 1000 iterations
      break;
    }
    if (n>=1000) n=0;
    if (!paused && !landed && n) return;
  }

  glutPostWindowRedisplay(closeup_window);
  glutPostWindowRedisplay(multi_window);
  glutPostWindowRedisplay(instrument_window);
}

bool safe_to_deploy_parachute (void)
  // Checks whether the parachute is safe to deploy at the current position and velocity
{
  double drag;

  // Assume high Reynolds number, quadratic drag = -0.5 * rho * v^2 * A * C_d
  drag = 0.5*DRAG_COEF_CHUTE*atmospheric_density(position)*M_PI*CHUTE_SIZE*CHUTE_SIZE*velocity_from_positions.abs2();
  // Do not use the global variable "altitude" here, in case this function is called from within the
  // numerical_dynamics function, before altitude is updated in the update_visualization function
  if ((drag > MAX_PARACHUTE_DRAG) || ((velocity_from_positions.abs() > MAX_PARACHUTE_SPEED) && ((position.abs() - MARS_RADIUS) < EXOSPHERE))) return false;
  else return true;
}

void update_visualization (void)
  // The visualization part of the idle function. Re-estimates altitude, velocity, climb speed and ground
  // speed from current and previous positions. Updates throttle and fuel levels, then redraws all subwindows.
{
  static vector3d last_track_position;
  vector3d av_p, d;
  double a, b, c, mu;

  simulation_time += delta_t;
  altitude = position.abs() - MARS_RADIUS;

  // Use average of current and previous positions when calculating climb and ground speeds
  av_p = (position + last_position).norm();
  if (delta_t != 0.0) velocity_from_positions = (position - last_position)/delta_t;
  else velocity_from_positions = vector3d(0.0, 0.0, 0.0);
  climb_speed = velocity_from_positions*av_p;
  ground_speed = (velocity_from_positions - climb_speed*av_p).abs();

  double M[16];
  quat_to_matrix(M, orientation);
  euler_angles = matrix_to_zxz_eul(M);

  // Check to see whether the lander has landed
  if (altitude < LANDER_SIZE/2.0) {
    glutIdleFunc(NULL);
    // Estimate position and time of impact
    d = position - last_position;
    a = d.abs2();
    b = 2.0*last_position*d;
    c = last_position.abs2() - (MARS_RADIUS + LANDER_SIZE/2.0) * (MARS_RADIUS + LANDER_SIZE/2.0);
    mu = (-b - sqrt(b*b-4.0*a*c))/(2.0*a);
    position = last_position + mu*d;
    simulation_time -= (1.0-mu)*delta_t; 
    altitude = LANDER_SIZE/2.0;
    landed = true;
    if ((fabs(climb_speed) > MAX_IMPACT_DESCENT_RATE) || (fabs(ground_speed) > MAX_IMPACT_GROUND_SPEED)) crashed = true;
    velocity_from_positions = vector3d(0.0, 0.0, 0.0);
  }

  // Update throttle and fuel (throttle might have been adjusted by the autopilot)
  for (int i=0; i<N_THRUSTERS; i++) {
    if (throttle[i] < 0.0) throttle[i] = 0.0;
    if (throttle[i] > 1.0) throttle[i] = 1.0;
    fuel -= delta_t * (FUEL_RATE_AT_MAX_THRUST*throttle[i]) / FUEL_CAPACITY;
  }
  if (fuel <= 0.0) fuel = 0.0;
  if (landed || (fuel == 0.0)) {
    for (int i=0; i<N_THRUSTERS; i++) throttle[i] = 0.0;
  }
  throttle_control = (short)(throttle[0]*THROTTLE_GRANULARITY + 0.5);

  // Check to see whether the parachute has vaporized or the tethers have snapped
  if (parachute_status == DEPLOYED) {
    if (!safe_to_deploy_parachute() || parachute_lost) {
      parachute_lost = true; // to guard against the autopilot reinstating the parachute!
      parachute_status = LOST;
    }
  }

  // Update record of lander's previous positions, but only if the position or the velocity has 
  // changed significantly since the last update
  if ( !track.n || (position-last_track_position).norm() * velocity_from_positions.norm() < TRACK_ANGLE_DELTA
      || (position-last_track_position).abs() > TRACK_DISTANCE_DELTA ) {
    track.pos[track.p] = position;
    if (track.n < N_TRACK) track.n++;
    if ((++track.p) == N_TRACK) track.p = 0;
    last_track_position = position;
  }

#ifdef GRAPHING
  log_value(altitude, 0);
  log_value(climb_speed, 1);
  log_value(ground_speed, 2);
  log_value(euler_angles.x, 3);
  log_value(euler_angles.y, 4);
  log_value(euler_angles.z, 5);
  log_value(angular_velocity.x, 6);
  log_value(angular_velocity.y, 7);
  log_value(angular_velocity.z, 8);

  if (n_points < N_POINTS) n_points++;
  else n_start++;
#endif

  // Redraw everything
  refresh_all_subwindows();
}

void attitude_stabilization (void)
  // Three-axis stabilization to ensure the lander's base is always pointing downwards
{
  vector3d up, left, out;
  static vector3d previous_left;
  double m[16];

  up = position.norm(); // this is the direction we want the lander's nose to point in

  // !!!!!!!!!!!!! HINT TO STUDENTS ATTEMPTING THE EXTENSION EXERCISES !!!!!!!!!!!!!!
  // For any-angle attitude control, we just need to set "up" to something different,
  // and leave the remainder of this function unchanged. For example, suppose we want
  // the attitude to be stabilized at stabilized_attitude_angle to the vertical in the
  // close-up view. So we need to rotate "up" by stabilized_attitude_angle degrees around
  // an axis perpendicular to the plane of the close-up view. This axis is given by the
  // vector product of "up"and "closeup_coords.right". To calculate the result of the
  // rotation, search the internet for information on the axis-angle rotation formula.

  // Set left to something perpendicular to up
  left.x = -up.y; left.y = up.x; left.z = 0.0;
  if (left.abs() < SMALL_NUM) {left.x = -up.z; left.y = 0.0; left.z = up.x;}
  left = left.norm();

  // Prevent instantaneous 180 degree spin when x passes through 0
  if (left * previous_left < 0.0 && simulation_time > 0.0) left *= -1;
  previous_left = left;

  out = left^up;

  // Construct modelling matrix (rotation only) from these three vectors
  m[0] = out.x; m[1] = out.y; m[2] = out.z; m[3] = 0.0;
  m[4] = left.x; m[5] = left.y; m[6] = left.z; m[7] = 0.0;
  m[8] = up.x; m[9] = up.y; m[10] = up.z; m[11] = 0.0;
  m[12] = 0.0; m[13] = 0.0; m[14] = 0.0; m[15] = 1.0;

  // Decomponse into unit quaternion
  orientation = matrix_to_quat(m);

  // Set angular velocity to zero, otherwise it will continue to change
  angular_velocity = vector3d();
}

vector3d thrust_wrt_world (const quat_t &ori)
  // Works out thrust vector in the world reference frame, given the lander's orientation
{
  double k;
  vector3d a;

  // Clip all throttle values to the range [0,1]
  for (int i=0; i<N_THRUSTERS; i++) {
    if (throttle[i] < 0.0) throttle[i] = 0.0;
    if (throttle[i] > 1.0) throttle[i] = 1.0;
    if (landed || (fuel == 0.0)) throttle[i] = 0.0;
  }

  for (int i=0; i<N_THRUSTERS; i++) {
    a += thruster_direction[i] * -throttle[i];
  }

  if (a.abs() > SMALL_NUM) return (-ori).transform(a) * MAX_THRUST;

  return vector3d();
}

void update_lander_state (void)
  // The GLUT idle function, called every time round the event loop
{
  unsigned long delay;

  // User-controlled delay
  if ((simulation_speed > 0) && (simulation_speed < 5)) {
    delay = (5-simulation_speed)*MAX_DELAY/4;
#ifdef WIN32
    Sleep(delay/1000); // milliseconds
#else
    usleep( (useconds_t)delay ); // microseconds
#endif
  }

  // This needs to be called every time step, even if the close-up view is not being rendered,
  // since any-angle attitude stabilizers reference closeup_coords.right
  update_closeup_coords();

  // Update historical record
  last_position = position;

  // Here we can apply an autopilot to adjust the thrust, parachute and attitude
  if (autopilot_enabled) autopilot();

  // Mechanical dynamics
  numerical_dynamics();

  // Here we can apply 3-axis stabilization to ensure the base is always pointing downwards
  if (stabilized_attitude) attitude_stabilization();

  // Refresh the visualization
  update_visualization();
}

void reset_simulation (void)
  // Resets the simulation to the initial state
{
  vector3d p, tv;
  unsigned long i;

  // Reset these three lander parameters here, so they can be overwritten in initialize_simulation() if so desired
  stabilized_attitude_angle = 0;
  for (int i=0; i<N_THRUSTERS; i++) throttle[i] = 0.0;
  fuel = 1.0;

  // Restore initial lander state
  initialize_simulation();

  // Correct initial attitude, if set by initial conditions
  if (stabilized_attitude) attitude_stabilization();

  // Calculate euler angles from orientation
  double M[16];
  quat_to_matrix(M, orientation);
  euler_angles = matrix_to_zxz_eul(M);

  // Check whether the lander is underground - if so, make sure it doesn't move anywhere
  landed = false;
  crashed = false;
  altitude = position.abs() - MARS_RADIUS;
  if (altitude < LANDER_SIZE/2.0) {
    glutIdleFunc(NULL);
    landed = true;
    velocity = vector3d(0.0, 0.0, 0.0);
  }

  // Visualisation routine's record of various speeds and velocities
  velocity_from_positions = velocity;
  last_position = position - delta_t*velocity_from_positions;
  p = position.norm();
  climb_speed = velocity_from_positions*p;
  tv = velocity_from_positions - climb_speed*p;
  ground_speed = tv.abs();

  // Miscellaneous state variables
  throttle_control = (short)(throttle[0]*THROTTLE_GRANULARITY + 0.5);
  simulation_time = 0.0;
  track.n = 0;
  parachute_lost = false;
  closeup_coords.initialized = false;
  closeup_coords.up    = vector3d(0.0, 1.0, 0.0);
  closeup_coords.right = vector3d(1.0, 0.0, 0.0);
  update_closeup_coords();

#ifdef GRAPHING
  // Clear all existing data
  n_points = 0; n_start = 0;
  for (int i=0; i<N_SERIES; i++) {
    graph_series[i].maxima.clear();
    graph_series[i].minima.clear();
  }

  log_value(altitude, 0);
  log_value(climb_speed, 1);
  log_value(ground_speed, 2);
  log_value(euler_angles.x, 3);
  log_value(euler_angles.y, 4);
  log_value(euler_angles.z, 5);
  log_value(angular_velocity.x, 6);
  log_value(angular_velocity.y, 7);
  log_value(angular_velocity.z, 8);

  n_points++;
#endif

  // Reset GLUT state
  if (paused || landed) refresh_all_subwindows();
  else glutIdleFunc(update_lander_state);
}

void set_multi_projection_matrix (void)
  // Called from reshape and zoom functions
{
  glutSetWindow(multi_window);
  if (multi_view_state == VIEW_ORBITAL) {
    double aspect_ratio;

    aspect_ratio = (double)view_width/(double)view_height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2.0*MARS_RADIUS*aspect_ratio/orbital_zoom, 2.0*MARS_RADIUS*aspect_ratio/orbital_zoom, 
      -2.0*MARS_RADIUS/orbital_zoom, 2.0*MARS_RADIUS/orbital_zoom, -100.0*MARS_RADIUS, 100.0*MARS_RADIUS);
  } else {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, view_width, 0, view_height, -1.0, 1.0);
  }
}

void reshape_main_window (int width, int height)
  // Called when the main window is created or resized
{
  // Resize the main window
  glutSetWindow(main_window);
  win_width = glutGet(GLUT_WINDOW_WIDTH);
  win_height = glutGet(GLUT_WINDOW_HEIGHT);

  // Work out subwindow dimensions and set projection matrix for main window
  view_width = (win_width - 4*GAP)/2;
  if (view_width < 1) view_width = 1; // GLUT warns about non-positive window dimensions
  view_height = (win_height - INSTRUMENT_HEIGHT - 4*GAP);
  if (view_height < 1) view_height = 1; // GLUT warns about non-positive window dimensions
  glViewport(0, 0, win_width, win_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, win_width, 0, win_height, -1.0, 1.0);

  // Resize and initialize the close-up view window
  glutSetWindow(closeup_window);
  glutPositionWindow(GAP, GAP);
  glutReshapeWindow(view_width, view_height);
  glViewport(0, 0, view_width, view_height);
  glDrawBuffer(GL_BACK);
  glutPostWindowRedisplay(closeup_window);
  
  // Resize and initialize the multi-view window
  glutSetWindow(multi_window);
  glutPositionWindow(view_width + 3*GAP, GAP);
  glutReshapeWindow(view_width, view_height);
  glViewport(0, 0, view_width, view_height);
  set_multi_projection_matrix();
  glDrawBuffer(GL_BACK);
  glutPostWindowRedisplay(multi_window);

  // Resize and initialize the instrument window
  glutSetWindow(instrument_window);
  glutPositionWindow(GAP, view_height + 3*GAP);
  glutReshapeWindow(2*(view_width+GAP), INSTRUMENT_HEIGHT);
  glViewport(0, 0, 2*(view_width+GAP), INSTRUMENT_HEIGHT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 2*(view_width+GAP), 0, INSTRUMENT_HEIGHT, -1.0, 1.0);
  glDrawBuffer(GL_BACK);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glutPostWindowRedisplay(instrument_window);

  // Layout GUI controls
  int mx = view_width*0.5 - 80.0;
  int cy = view_height - 60;
  gui_menu_prop->setPosition(mx,cy);  gui_menu_prop->setSize(160,20);  cy -= 35;
  gui_menu_orbit->setPosition(mx,cy); gui_menu_orbit->setSize(160,20); cy -= 35;
#ifdef GRAPHING
  gui_menu_graph->setPosition(mx,cy); gui_menu_graph->setSize(160,20); cy -= 35;
#endif
  gui_menu_auto->setPosition(mx,cy); gui_menu_auto->setSize(160,20); cy -= 35;
  gui_menu_help->setPosition(mx,cy); gui_menu_help->setSize(160,20);
  gui_menu_quit->setPosition(view_width*0.5 - 60.0,20); gui_menu_quit->setSize(120,20);

  cy = view_height - 30;
  gui_prop_load->setPosition(5, cy);
  gui_prop_file->setPosition(130, cy); gui_prop_file->setSize(160,20);
  gui_prop_save->setPosition(295, cy); cy -= 45;
  gui_prop_txt1->setPosition(5, cy+5);
  gui_prop_mass->setPosition(200, cy); cy -= 60;
  gui_prop_txt2->setPosition(5, cy+5);
  gui_prop_ixx->setPosition(200, cy);  cy -= 35;
  gui_prop_iyy->setPosition(200, cy);  cy -= 35;
  gui_prop_izz->setPosition(200, cy);

#ifdef GRAPHING
  mx = view_width - 92 * 2;
  cy = 1;
  gui_graph_csv->setPosition(125, cy);  gui_graph_csv->setSize(65, 20);
  gui_graph_csvp->setPosition(195, cy); gui_graph_csvp->setSize(100, 20);
  gui_graph_show3->setPosition(mx, cy); gui_graph_show1->setSize(90, 15); cy += 18;
  gui_graph_show2->setPosition(mx, cy); gui_graph_show2->setSize(90, 15); cy += 18;
  gui_graph_show1->setPosition(mx, cy); gui_graph_show3->setSize(90, 15);
  mx += 92;
  cy = 1; cy += 18;
  gui_graph_show5->setPosition(mx, cy); gui_graph_show5->setSize(90, 15); cy += 18;
  gui_graph_show4->setPosition(mx, cy); gui_graph_show4->setSize(90, 15);
#endif

  cy = view_height - 30;
  for (vector<GuiControl*>::iterator it=auto_controls.begin(); it<auto_controls.end(); it++) {
    (*it)->setPosition(5, cy+5); it++;
    (*it)->setPosition(200, cy); cy -= 35;
  }

  instrument_auto->setPosition(view_width+GAP-475, INSTRUMENT_HEIGHT-38); instrument_auto->setSize(150, 20);
  instrument_att->setPosition(view_width+GAP-225, INSTRUMENT_HEIGHT-38); instrument_att->setSize(150, 20);
  instrument_para->setPosition(view_width+GAP+25, INSTRUMENT_HEIGHT-38); instrument_para->setSize(150, 20);
}

bool gui_mouse_button (int button, int state, int x, int y, ViewState filter)
  // Test mouse button events against the gui controls
  // x and y are measured from the lower left corner of the relevant view
{
  bool stop = false;

  // Only consider the left mouse button for gui events
  if (button == GLUT_LEFT_BUTTON) {
    // If a control is in focus, send it the event
    if (gui_focus != NULL) {
      gui_focus->event(state==GLUT_DOWN?GUI_MOUSE_DOWN:GUI_MOUSE_UP, x, y);

      // If the control is still in focus, stop event processing
      if (gui_focus != NULL) stop = true;
    }
    if (!stop && state == GLUT_DOWN) {
      // Search all visible controls for one which is under the mouse
      for (vector<GuiControl*>::iterator it=controls.begin(); it<controls.end(); it++) {
        if ((*it)->view == filter) {
          if ((*it)->hitTest(x, y)) {
            // A control is hit, so send it the event and stop looking further
            (*it)->event(GUI_MOUSE_DOWN, x, y);
            stop = true;
            break;
          }
        }
      }
    }
  }
  return stop;
}

void multi_mouse_button (int button, int state, int x, int y)
  // Callback for mouse button presses in the multi-view window
  // x and y are measured from the top left corner of the current view
{
  bool stop = false;

  stop = gui_mouse_button(button, state, x, view_height-y, multi_view_state);

  if (!stop && multi_view_state == VIEW_ORBITAL) {
    if ((button == GLUT_WHEEL_UP) || (((button == GLUT_MIDDLE_BUTTON) || glutGetModifiers()) && (state == GLUT_DOWN))) {
      if (orbital_zoom < 200.0) { // don't let them get too close!
        orbital_zoom /= 0.9;
        if ((button == GLUT_MIDDLE_BUTTON) || glutGetModifiers()) orbital_zoom /= 0.9; // to match wheel events
      }
      save_orbital_zoom = -1.0;
      set_multi_projection_matrix();
    } else if ((button == GLUT_WHEEL_DOWN) || ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN))) {
      if (orbital_zoom > 0.1) {
        orbital_zoom *= 0.9;
        if (button == GLUT_RIGHT_BUTTON) orbital_zoom *= 0.9; // to match wheel events
      }
      save_orbital_zoom = -1.0;
      set_multi_projection_matrix();
    }
    if (button == GLUT_LEFT_BUTTON) {
      if (state == GLUT_UP) {
        last_click_x = -1;
        last_click_y = -1;
      }
      if (state == GLUT_DOWN) {
        last_click_x = x;
        last_click_y = y;
      }
    }
  }
  if (paused || landed) glutPostWindowRedisplay(multi_window);
}

void multi_mouse_motion (int x, int y)
  // Callback for mouse drags in the multi-view window
{
  if (multi_view_state == VIEW_ORBITAL) {
    quat_t spin_quat;
    
    if (last_click_x < 0) return; // not the left mouse button

    spin_quat = track_quats((2.0*last_click_x - view_width) / view_width,
          (view_height - 2.0*last_click_y) / view_height,
          (2.0*x - view_width) / view_width,
          (view_height - 2.0*y) / view_height);
    orbital_quat = orbital_quat * spin_quat;
    
    last_click_x = x;
    last_click_y = y;
    
    if (paused || landed) glutPostWindowRedisplay(multi_window);
  }
}

void instrument_mouse_button (int button, int state, int x, int y)
  // Callback for mouse button presses in the instrument window
  // x and y are measured from the top left corner of the current view
{
  gui_mouse_button(button, state, x, INSTRUMENT_HEIGHT-y, VIEW_INSTRUMENT);
  if (paused || landed) refresh_all_subwindows();
}

void closeup_mouse_button (int button, int state, int x, int y)
  // Callback for mouse button presses in the close-up view window
{
  if ((button == GLUT_WHEEL_UP) || (((button == GLUT_MIDDLE_BUTTON) || glutGetModifiers()) && (state == GLUT_DOWN))) {
    if (closeup_offset > 2.5*LANDER_SIZE) {
      closeup_offset *= 0.9;
      if ((button == GLUT_MIDDLE_BUTTON) || glutGetModifiers()) closeup_offset *= 0.9; // to match wheel events
    }
    if (paused || landed) refresh_all_subwindows();
  } else if ((button == GLUT_WHEEL_DOWN) || ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN))) {
    if (closeup_offset < 200.0*LANDER_SIZE) {
      closeup_offset /= 0.9;
      if (button == GLUT_RIGHT_BUTTON) closeup_offset /= 0.9; // to match wheel events
    }
    if (paused || landed) refresh_all_subwindows();
  }
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_UP) {
      last_click_y = -1;
      last_click_x = -1;
    }
    if (state == GLUT_DOWN) {
      last_click_y = y;
      last_click_x = x;
    }
  }
}

void closeup_mouse_motion (int x, int y)
  // Callback for mouse drags in the close-up view window
{
  if (last_click_x < 0) return; // not the left mouse button

  closeup_xr += y - last_click_y;
  closeup_yr += x - last_click_x;
  if (closeup_xr < 0.0)  closeup_xr = 0.0;
  if (closeup_xr > 90.0) closeup_xr = 90.0;
  while (closeup_yr <= -180.0) closeup_yr += 360.0;
  while (closeup_yr >   180.0) closeup_yr -= 360.0;
  last_click_y = y;
  last_click_x = x;
  if (paused || landed) glutPostWindowRedisplay(closeup_window);
}

void glut_special (int key, int x, int y)
  // Callback for special key presses in all windows
{

  // If a gui control has focus, intercept the key event
  if (gui_focus != NULL) {
    gui_focus->event(GUI_SPECIAL, key);
    refresh_all_subwindows();
    return;
  }

  switch(key) {
  case GLUT_KEY_UP: // throttle up
    if (!autopilot_enabled && !landed && (fuel>0.0)) {
      throttle_control++;
      if (throttle_control>THROTTLE_GRANULARITY) throttle_control = THROTTLE_GRANULARITY;
      for (int i=0; i<N_THRUSTERS; i++) throttle[i] = (double)throttle_control/THROTTLE_GRANULARITY;
    }
    break;
  case GLUT_KEY_DOWN: // throttle down
    if (!autopilot_enabled && !landed) {
      throttle_control--;
      if (throttle_control<0) throttle_control = 0;
      for (int i=0; i<N_THRUSTERS; i++) throttle[i] = (double)throttle_control/THROTTLE_GRANULARITY;
    }
    break;
  case GLUT_KEY_RIGHT: // faster simulation
    simulation_speed++;
    if (simulation_speed>10) simulation_speed = 10;
    if (paused) {
      if (!landed) glutIdleFunc(update_lander_state);
      paused = false;
    }
    break;
  case GLUT_KEY_LEFT: // slower simulation
    simulation_speed--;
    if (simulation_speed<0) simulation_speed = 0;
    if (!simulation_speed) {
      glutIdleFunc(NULL);
      paused = true;
    }
    break;
  }

  if (paused || landed) refresh_all_subwindows();
}

void glut_key (unsigned char k, int x, int y)
  // Callback for key presses in all windows
{

  // If a gui control has focus, intercept the key event
  if (gui_focus != NULL) {
    gui_focus->event(GUI_KEY, (int)k);
    refresh_all_subwindows();
    return;
  }

  switch(k) {

  case 27: case 'q': case 'Q':
    // Escape or q or Q  - exit
    exit(0);
    break;

  case '0':
    // switch to scenario 0
    scenario = 0;
    reset_simulation();
    break;

  case '1':
    // switch to scenario 1
    scenario = 1;
    reset_simulation();
    break;

  case '2':
    // switch to scenario 2
    scenario = 2;
    reset_simulation();
    break;

  case '3':
    // switch to scenario 3
    scenario = 3;
    reset_simulation();
    break;

  case '4':
    // switch to scenario 4
    scenario = 4;
    reset_simulation();
    break;

  case '5':
    // switch to scenario 5
    scenario = 5;
    reset_simulation();
    break;

  case '6':
    // switch to scenario 6
    scenario = 6;
    reset_simulation();
    break;

  case '7':
    // switch to scenario 7
    scenario = 7;
    reset_simulation();
    break;

  case '8':
    // switch to scenario 8
    scenario = 8;
    reset_simulation();
    break;

  case '9':
    // switch to scenario 9
    scenario = 9;
    reset_simulation();
    break;

  case 'r': case 'R':
    // r or R - restart scenario
    reset_simulation();
    break;

  case 'a': case 'A':
    // a or A - autopilot
    if (!landed) autopilot_enabled = !autopilot_enabled;
    if (paused) refresh_all_subwindows();
    break;

  case 'h': case 'H':
    // h or H - help
    if (help) {
      help = false;
      if (save_orbital_zoom > 0.0) orbital_zoom = save_orbital_zoom;
      set_multi_projection_matrix();
    } else {
      help = true;
      save_orbital_zoom = orbital_zoom;
      orbital_zoom = 0.4;
      if (multi_view_state != VIEW_ORBITAL) set_multi_view(VIEW_ORBITAL);
      else set_multi_projection_matrix();
    }
    if (paused || landed) refresh_all_subwindows();
    break;

  case 'l': case 'L':
    // l or L - toggle lighting model
    static_lighting = !static_lighting;
    glutSetWindow(multi_window); enable_lights();
    glutSetWindow(closeup_window); enable_lights();
    if (paused || landed) refresh_all_subwindows();
    break;

  case 't': case 'T':
    // t or T - terrain texture
    do_texture = !do_texture;
    if (!texture_available) do_texture = false;
    if (paused || landed) refresh_all_subwindows();
    break;

  case 'p': case 'P':
    // p or P - deploy parachute
    if (!autopilot_enabled && !landed && (parachute_status == NOT_DEPLOYED)) parachute_status = DEPLOYED;
    if (paused) refresh_all_subwindows();
    break;

  case 's': case 'S':
    // s or S - attitude stabilizer
    if (!landed) stabilized_attitude = !stabilized_attitude;
    if (paused) refresh_all_subwindows();
    break;

  case 32:
    // space bar
    simulation_speed = 0;
    glutIdleFunc(NULL);
    if (paused && !landed) update_lander_state();
    else refresh_all_subwindows();
    paused = true;
    break;
  
#ifdef GRAPHING
  case 'g': case 'G':
    // g or G - show graph view
    set_multi_view(VIEW_GRAPH);
    break;
#endif

  case 'm': case 'M':
    // m or M - show menu view
    set_multi_view(VIEW_MENU);
    break;

  case 'o': case 'O':
    // o or O - show orbital view
    set_multi_view(VIEW_ORBITAL);
    break;

  }
}

int main (int argc, char* argv[])
  // Initializes GLUT windows and lander state, then enters GLUT main loop
{
  // Extract path from argument 0
  string path = string(argv[0]);
  size_t sp = path.rfind(PATH_SEP);
  if (sp != string::npos) path.erase(sp+1);
  else path.erase();
  exe_path = path;

  // Main GLUT window
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowPosition(0, 0);
  glutInitWindowSize(PREFERRED_WIDTH, PREFERRED_HEIGHT);
  view_width = (PREFERRED_WIDTH - 4*GAP)/2;
  view_height = (PREFERRED_HEIGHT - INSTRUMENT_HEIGHT - 4*GAP);
  main_window = glutCreateWindow("Mars Lander (Gabor Csanyi and Andrew Gee, September 2011)");
  glDrawBuffer(GL_BACK);
  glLineWidth(2.0);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glutDisplayFunc(draw_main_window);
  glutReshapeFunc(reshape_main_window);  
  glutIdleFunc(update_lander_state);
  glutKeyboardFunc(glut_key);
  glutSpecialFunc(glut_special);

  // The close-up view subwindow
  closeup_window = glutCreateSubWindow(main_window, GAP, GAP, view_width, view_height);
  glDrawBuffer(GL_BACK);
  setup_lights();
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_CULL_FACE); // we only need back faces for the parachute
  glDisable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_NORMALIZE);
  glDepthFunc(GL_LEQUAL);
  glShadeModel(GL_SMOOTH);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE); // we need two-sided lighting for the parachute
  glEnable(GL_COLOR_MATERIAL);
  glFogi(GL_FOG_MODE, GL_EXP);
  glutDisplayFunc(draw_closeup_window);
  glutMouseFunc(closeup_mouse_button);
  glutMotionFunc(closeup_mouse_motion);
  glutKeyboardFunc(glut_key);
  glutSpecialFunc(glut_special);
  texture_available = setup_terrain_texture() && setup_lander_texture() && setup_planet_texture(planet_texture1);
  closeup_offset = 50.0;
  closeup_xr = 10.0;
  closeup_yr = 0.0;
  terrain_angle = 0.0;

  // The multi-view subwindow, initially orbital view
  multi_window = glutCreateSubWindow(main_window, view_width + 3*GAP, GAP, view_width, view_height);
  glDrawBuffer(GL_BACK);
  setup_lights();
  multi_view_state = VIEW_INVALID;
  glutDisplayFunc(draw_multi_window);
  glutMouseFunc(multi_mouse_button);
  glutMotionFunc(multi_mouse_motion);
  glutKeyboardFunc(glut_key);
  glutSpecialFunc(glut_special);
  if (texture_available) texture_available = setup_planet_texture(planet_texture2);
  quadObj = gluNewQuadric();
  orbital_quat = quat_t(-0.82, -0.53, 0.21, -0.047).norm();
  save_orbital_zoom = 1.0;
  orbital_zoom = 1.0;
  set_multi_view(VIEW_MENU);

  // The instrument subwindow
  instrument_window = glutCreateSubWindow(main_window, GAP, view_height + 3*GAP, 2*(view_width+GAP), INSTRUMENT_HEIGHT);
  glutDisplayFunc(draw_instrument_window);
  glutMouseFunc(instrument_mouse_button);
  glutKeyboardFunc(glut_key);
  glutSpecialFunc(glut_special);

  if (!texture_available) {
    cout << "Textures not available. See error message(s) above for details." << endl;
    do_texture = false;
  }

  // Generate the random number table
  srand(0);
  for (int i=0; i<N_RAND; i++) randtab[i] = (float)rand()/RAND_MAX;

#ifdef GRAPHING
  // Set initial graph series visibility (altitude only)
  graph_series[0].visible = true;
  for (int i=1; i<N_SERIES; i++) graph_series[i].visible = false;
#endif

  // Create the gui controls for all subwindows
  setup_gui();

  // Process the lander mesh data to product a set of elemental areas
  // for drag calculations in lander.cpp
  setup_aero_mesh();

  // Initialize the simulation state
  reset_simulation();
  microsecond_time(time_program_started);

  // Start the main event loop
  // This function does not return
  glutMainLoop();
}

void set_multi_view (ViewState new_state)
 // Sets necessary GL state and variables for 
{

  // Nothing to do if already in the required state
  if (new_state == multi_view_state) return;

  // Turn help off when leaving orbital view
  if (new_state != VIEW_ORBITAL && help) {
    help = false;
    if (save_orbital_zoom > 0.0) orbital_zoom = save_orbital_zoom;
  }

  glutSetWindow(multi_window);

  // Set OpenGL state for either 3D or 2D rendering
  if (new_state == VIEW_ORBITAL) {
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_LIGHTING);
      glEnable(GL_CULL_FACE); // since the only polygons in this view define a solid sphere
      glDisable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glEnable(GL_NORMALIZE);
      glDepthFunc(GL_LEQUAL);
      glShadeModel(GL_SMOOTH);
      glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
      glEnable(GL_COLOR_MATERIAL);
  } else {
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);
      glDisable(GL_CULL_FACE);
      glDisable(GL_BLEND);
      glDisable(GL_NORMALIZE);
      glDisable(GL_COLOR_MATERIAL);
  }

  multi_view_state = new_state;

  // Unfocus all controls in the gui
  gui_focus = NULL;

  // Set the correct orthographic projection for the view
  set_multi_projection_matrix();

  // Redraw the window
  glutPostWindowRedisplay(multi_window);

}

unsigned char *load_texture_file (const string &path, GLsizei w, GLsizei h, GLint channels)
{
  unsigned char *tex = NULL;
  bool texture_ok = false;

  ifstream file(path.c_str(), ios::in | ios::binary);
  if (file) {
    tex = (unsigned char*)malloc(sizeof(unsigned char) * (w * h * channels));
    if (tex == NULL) {
      cerr << "Error: malloc failed in load_texture_file()" << endl;
      exit(1);
    }

    file.read((char *)tex, w * h * channels);

    if (file.fail()) {
      cerr << "Error: Failed to read texture from file '" << path << "'" << endl;
    } else {
      texture_ok = true;
    }

    file.close();
  } else {
    // Failed to open file (possibly file not found)
  }

  if (!texture_ok && tex != NULL) {
    free(tex);
    tex = NULL;
  }

  return tex;
}

bool prepare_texture (GLuint id, unsigned char *data, GLsizei w, GLsizei h, GLint channels)
{
  bool texture_ok = false;
  GLsizei tw = w, th = h;
  GLenum form;

  switch (channels) {
  case 1:
    form = GL_LUMINANCE;
    break;
  case 3:
    form = GL_RGB;
    break;
  default:
    cerr << "Error: Unsupported colour channel count (" << channels << ") in prepare_texture()" << endl;
    return false;
  }

  glBindTexture(GL_TEXTURE_2D, id);
  if (glGetError() != GL_NO_ERROR) {
    cerr << "Error: Could not bind texture id " << id << endl;
    return false;
  }

  while (!texture_ok && (min(tw, th) >= 64)) { // try progressively smaller texture maps, give up below 64x64
    glGetError();  // Clear any OpenGL error
    if (!gluBuild2DMipmaps(GL_TEXTURE_2D, channels, tw, th, form, GL_UNSIGNED_BYTE, data) && (glGetError() == GL_NO_ERROR)) {
      texture_ok = true;
    } else {
      tw /= 2; th /= 2;
    }
  }

  if (texture_ok) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  }

  return texture_ok;
}

bool setup_terrain_texture ()
{
  bool texture_ok = false;

  // Generate a random texture for the ground
  unsigned char *tex1 = (unsigned char*)malloc(sizeof(unsigned char) * (TERRAIN_TEXTURE_SIZE*TERRAIN_TEXTURE_SIZE));
  if (tex1 != NULL) {
    for (unsigned long x=0; x<TERRAIN_TEXTURE_SIZE*TERRAIN_TEXTURE_SIZE; x++) tex1[x] = 192 + (unsigned char) (63.0*rand()/RAND_MAX);
    glGenTextures(1, &terrain_texture);
    texture_ok = prepare_texture(terrain_texture, tex1, TERRAIN_TEXTURE_SIZE, TERRAIN_TEXTURE_SIZE, 1);
    free(tex1);
  } else {
    cerr << "Error: malloc failed in setup_textures()" << endl;
    exit(1);
  }

  return texture_ok;
}

bool setup_lander_texture ()
{
  return setup_texture("lander.dat", LANDER_TEXTURE_SIZE, 1, lander_texture);
}

bool setup_planet_texture (GLuint &id)
{
  return setup_texture("mars.dat", PLANET_TEXTURE_SIZE, 3, id);
}

bool setup_texture (const string &name, GLsizei size, GLint channels, GLuint &id)
{
  bool texture_ok = false;

  // Attempt to load the texture file from the executable directory
  unsigned char *tex = load_texture_file(exe_path + name, size, size, channels);

  // If the file was not found, try looking in the two parent directories
  // (common when using an IDE to compile)
  if (tex == NULL) tex = load_texture_file(exe_path + "../" + name, size, size, channels);
  if (tex == NULL) tex = load_texture_file(exe_path + "../../" + name, size, size, channels);

  // If texture data was loaded, make it available as an OpenGL texture
  // in the current subwindow context
  if (tex != NULL) {
    glGenTextures(1, &id);
    texture_ok = prepare_texture(id, tex, size, size, channels);
    free(tex);
  } else {
    cerr << "Error: Failed to locate texture file '" << name << "'" << endl;
  }

  return texture_ok;
}

void draw_lander ()
{
  if (do_texture) {
    glBindTexture(GL_TEXTURE_2D, lander_texture);
    glEnable(GL_TEXTURE_2D);
  }
  glColor3f(0.8, 0.8, 0.8);
  draw_frustrum(LANDER_SIZE*0.9, LANDER_SIZE*0.5, LANDER_SIZE*0.5, 40, 25, true);
  if (do_texture) glDisable(GL_TEXTURE_2D);
  
  glColor3f(0.5, 0.4, 0.3);
  draw_frustrum(LANDER_SIZE, LANDER_SIZE*0.25, -LANDER_SIZE*0.5, 40, 25, true);
}

void draw_frustrum (GLdouble base, GLdouble top, GLdouble height, GLint slices, GLint stacks, bool closed)
  // Modified from freeglut's glutSolidCone, used to draw thrusters
{
  double z0, z1, r0, r1, *sint, *cost;

  if (stacks < 1) stacks = 1;

  const double dr = base - top;

  double zStep = height / stacks;
  double rStep = dr / stacks;
  double cosn = ( height  / sqrt ( height * height + dr * dr ));
  double sinn = ( dr / sqrt ( height * height + dr * dr ));

  fghCircleTable(&sint, &cost, -slices);

  if (height > 0) {
    z0 = 0.0; r0 = base;
  } else {
    z0 = height; r0 = top;
    zStep = -zStep; rStep = -rStep;
  }
  z1 = z0 + zStep; r1 = r0 - rStep;

  if (closed) {
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0.0, 0.0, -1.0);
    glTexCoord2f(0.25, 0.75); glVertex3d(0.0, 0.0, z0);
    for (int j=0; j<=slices; j++) {
      glTexCoord2f(0.25+0.25*cost[j], 0.75+0.25*sint[j]);
      glVertex3d(cost[j]*r0, sint[j]*r0, z0);
    }
    glEnd();
  }

  for (int i=0; i<stacks; i++) {
    glBegin(GL_QUAD_STRIP);
    for (int j=0; j<=slices; j++) {
      glNormal3d(cost[j]*sinn, sint[j]*sinn, cosn);
      glTexCoord2f((double)j / slices, 0.5 * z0 / height); glVertex3d(cost[j]*r0, sint[j]*r0, z0);
      glTexCoord2f((double)j / slices, 0.5 * z1 / height); glVertex3d(cost[j]*r1, sint[j]*r1, z1);
    }
    z0 = z1; z1 += zStep;
    r0 = r1; r1 -= rStep;
    glEnd();
  }

  if (closed) {
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0.0, 0.0, 1.0);
    glTexCoord2f(0.75, 0.75); glVertex3d(0.0, 0.0, z0);
    for (int j=slices; j>=0; j--) {
      glTexCoord2f(0.75-0.25*cost[j], 0.75+0.25*sint[j]);
      glVertex3d(cost[j]*r0, sint[j]*r0, z0);
    }
    glEnd();
  }

  free(sint); free(cost);
}

void draw_flare (GLdouble base, GLdouble height, GLint slices, GLint stacks)
  // Modified from freeglut's glutSolidCone, we need this to draw thruster flares with changing colour
{
  double z0, z1, r0, r1, *sint, *cost;

  if (stacks < 1) stacks = 1;

  const double zStep = height / stacks;
  const double rStep = base / stacks;
  fghCircleTable(&sint, &cost, -slices);
  z0 = 0.0; z1 = zStep;
  r0 = base; r1 = r0 - rStep;

  for (int i=0; i<stacks-1; i++) {
    glBegin(GL_QUAD_STRIP);
    for (int j=0; j<=slices; j++) {
      glColor4f(1.0-(z0/height)*0.2, 1.0-(z0/height)*0.8, 1.0-(z0/height), 1.0-(z0/height)*0.9);
      glVertex3d(cost[j]*r0, sint[j]*r0, z0);
      glColor4f(1.0-(z1/height)*0.2, 1.0-(z1/height)*0.8, 1.0-(z1/height), 1.0-(z1/height)*0.9);
      glVertex3d(cost[j]*r1, sint[j]*r1, z1);
    }
    z0 = z1; z1 += zStep;
    r0 = r1; r1 -= rStep;
    glEnd();
  }

  glBegin(GL_TRIANGLES);
  for (int j=0; j<slices; j++) {
    glColor4f(1.0-(z0/height)*0.2, 1.0-(z0/height)*0.8, 1.0-(z0/height), 1.0-(z0/height)*0.9);
    glVertex3d(cost[j+1]*r0, sint[j+1]*r0, z0);
    glVertex3d(cost[j+0]*r0, sint[j+0]*r0, z0);
    glColor4f(0.8, 0.2, 0.0, 0.0);
    glVertex3d(0.0, 0.0, height);
  }
  glEnd();

  free(sint); free(cost);
}

void look_in_direction (const vector3d &v)
  // Similar to gluLookAt, transforms current OpenGL matrix to look in the direction of a vector
{
  vector3d z = v.norm();
  vector3d y = vector3d(-z.y, z.x, 0.0);
  if (y.abs() < SMALL_NUM) y = vector3d(z.z, 0.0, -z.x);
  y = y.norm();
  vector3d x = y ^ z;
  
  double m[16];
  m[0] = x.x; m[1] = x.y; m[2] = x.z; m[3] = 0.0;
  m[4] = y.x; m[5] = y.y; m[6] = y.z; m[7] = 0.0;
  m[8] = z.x; m[9] = z.y; m[10] = z.z; m[11] = 0.0;
  m[12] = 0.0; m[13] = 0.0; m[14] = 0.0; m[15] = 1.0;

  glMultMatrixd(m);
}

void setup_aero_mesh ()
{
  // Bottom face
  aero.push_back(aero_element_t(vector3d(0.0, 0.0, -M_PI*0.25*LANDER_SIZE*0.25*LANDER_SIZE), vector3d(0.0, 0.0, -0.5*LANDER_SIZE)));
  
  // Heat shield
  for (int i=0; i<8; i++) {
    double a = 2.0*M_PI*(i/8.0);
    vector3d n = vector3d(cos(a)*0.555, sin(a)*0.555, -0.832) * (1.127*M_PI*LANDER_SIZE*LANDER_SIZE)/8.0;
    aero.push_back(aero_element_t(n, vector3d(cos(a)*0.625*LANDER_SIZE, sin(a)*0.625*LANDER_SIZE, -0.25*LANDER_SIZE)));
  }
  
  // Flat section
  aero.push_back(aero_element_t(vector3d(0.0, 0.0, M_PI*0.1*LANDER_SIZE*0.1*LANDER_SIZE), vector3d()));
  
  // Upper body
  for (int i=0; i<8; i++) {
    double a = 2.0*M_PI*(i/8.0);
    vector3d n = vector3d(cos(a)*0.781, sin(a)*0.781, 0.625) * (0.8964*M_PI*LANDER_SIZE*LANDER_SIZE)/8.0;
    aero.push_back(aero_element_t(n, vector3d(cos(a)*0.7*LANDER_SIZE, sin(a)*0.7*LANDER_SIZE, 0.25*LANDER_SIZE)));
  }
  
  // Top face
  aero.push_back(aero_element_t(vector3d(0.0, 0.0, M_PI*0.5*LANDER_SIZE*0.5*LANDER_SIZE), vector3d(0.0, 0.0, 0.5*LANDER_SIZE)));
}

void GuiText::render (bool focus) const {
  glColor3f(1.0, 1.0, 1.0);
  glut_print(x, y, text, font, align);
}

void GuiButton::event (GuiEvent type, int mx, int my) {
  // Buttons focus on mouse down, and generate an event on mouse up
  if (type == GUI_MOUSE_DOWN) {
    gui_focus = this;
    refresh_all_subwindows();
  } else if (type == GUI_MOUSE_UP) {
    if (hitTest(mx, my)) do_callback();
    gui_focus = NULL;
    refresh_all_subwindows();
  }
}

void GuiButton::render (bool focus) const {
  if (focus) glColor3f(0.5, 0.5, 0.5);
  else glColor3f(XYZ(col1));
  glBegin(GL_QUADS);
    glVertex2d(x, y);
    glVertex2d(x+width, y);
    glVertex2d(x+width, y+height);
    glVertex2d(x, y+height);
  glEnd();
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINE_LOOP);
    glVertex2d(x, y);
    glVertex2d(x+width, y);
    glVertex2d(x+width, y+height);
    glVertex2d(x, y+height);
  glEnd();

  glut_print(width*0.5+x, y+6, text, GLUT_BITMAP_HELVETICA_12, ALIGN_MID);
}

void GuiToggle::event (GuiEvent type, int mx, int my) {
  // Buttons focus on mouse down, and generate an event on mouse up
  if (type == GUI_MOUSE_DOWN) {
    gui_focus = this;
    refresh_all_subwindows();
  } else if (type == GUI_MOUSE_UP) {
    if (hitTest(mx, my)) {
      if (val != NULL) (*val) = !(*val);
      do_callback();
    }
    gui_focus = NULL;
    refresh_all_subwindows();
  }
}

void GuiToggle::render (bool focus) const {
  // Toggle buttons focus on mouse down, and toggle their state on mouse up
  if (focus) glColor3f(0.5, 0.5, 0.5);
  else if (val != NULL) {
    if (*val) glColor3f(XYZ(col1));
    else glColor3f(XYZ(col2));
  } else glColor3f(0.0, 0.0, 0.0);
  glBegin(GL_QUADS);
    glVertex2d(x, y);
    glVertex2d(x+width, y);
    glVertex2d(x+width, y+height);
    glVertex2d(x, y+height);
  glEnd();
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINE_LOOP);
    glVertex2d(x, y);
    glVertex2d(x+width, y);
    glVertex2d(x+width, y+height);
    glVertex2d(x, y+height);
  glEnd();

  glut_print(width*0.5+x, height*0.5+y-4, text, GLUT_BITMAP_HELVETICA_12, ALIGN_MID);
}

void GuiInput::event (GuiEvent type, int i1, int i2) {
  // Input boxes focus on mouse down, and generate an event on key press
  if (type == GUI_MOUSE_DOWN) {
    if (hitTest(i1, i2)) {
      tmptext = text;
      pos = text.length();
      gui_focus = this;
    } else {
      update();
      do_callback();
      gui_focus = NULL;
    }
  } else if (type == GUI_SPECIAL) {
    switch (i1) {
    case GLUT_KEY_LEFT:
      if (pos) pos -= 1;
      break;
    case GLUT_KEY_RIGHT:
      if (pos < text.length()) pos += 1;
      break;
    case GLUT_KEY_HOME:
      pos = 0;
      break;
    case GLUT_KEY_END:
      pos = text.length();
      break;
    }
  } else if (type == GUI_KEY) {
    // Check for printable ascii characters
    if (i1 >= 32 && i1 <= 126) {
      text.insert(text.begin()+(pos++), (char)i1);
    } else {
      // Check for known ascii control characters
      switch (i1) {
      case ASCII_BACKSPACE:
        // Backspace
        if (pos) text.erase((--pos), 1);
        break;
      case 27:
        // Escape
        if (val == NULL) text = tmptext;
        else refresh();
        gui_focus = NULL;
        break;
      case ASCII_DELETE:
        // Delete
        if (pos < text.length()) text.erase(pos, 1);
        break;
      case 13:
        // Return
        update();
        do_callback();
        gui_focus = NULL;
        break;
      default:
        break;
      }
    }
  }
}

void GuiInput::update ()
{
  if (val != NULL) {
    stringstream ss(text);
    double tmp;
    ss >> tmp;
    if (!ss.fail()) *val = tmp;
    refresh();
  }
}

void GuiInput::refresh ()
{
  if (val != NULL) {
    stringstream ss;
    ss << *val;
    text = ss.str();
  }
}

void GuiInput::render (bool focus) const {
  glColor3f(0.1, 0.1, 0.1);
  glBegin(GL_QUADS);
    glVertex2d(x, y);
    glVertex2d(x+width, y);
    glVertex2d(x+width, y+height);
    glVertex2d(x, y+height);
  glEnd();
  if (focus) glColor3f(0.2, 0.8, 0.2);
  else {
    glColor3f(1.0, 1.0, 1.0);
  }
  glBegin(GL_LINE_LOOP);
    glVertex2d(x, y);
    glVertex2d(x+width, y);
    glVertex2d(x+width, y+height);
    glVertex2d(x, y+height);
  glEnd();
  
  glut_print(x+5, y+5, text);
  
  if (focus) {
    glColor3f(1.0, 1.0, 1.0);
    int ox = x+5+glutBitmapLength(GLUT_BITMAP_HELVETICA_10, (const unsigned char*)text.substr(0, pos).c_str());
    glBegin(GL_LINES);
    glVertex2f(ox, y+2);
    glVertex2f(ox, y+15);
    glEnd();
  }
  
}

void setup_gui () {
  // Create the control objects
  controls.push_back(gui_menu_prop  = new GuiButton(VIEW_MENU, "Lander properties", &gui_event));
  controls.push_back(gui_menu_orbit = new GuiButton(VIEW_MENU, "Orbital view", &gui_event));
#ifdef GRAPHING
  controls.push_back(gui_menu_graph = new GuiButton(VIEW_MENU, "Graphing view", &gui_event));
#endif
  controls.push_back(gui_menu_auto  = new GuiButton(VIEW_MENU, "Autopilot settings", &gui_event));
  controls.push_back(gui_menu_help  = new GuiButton(VIEW_MENU, "Help", &gui_event));
  controls.push_back(gui_menu_quit  = new GuiButton(VIEW_MENU, "Quit", &gui_event));

  controls.push_back(gui_prop_load  = new GuiButton(VIEW_PROP, "Load file", &gui_event));
  controls.push_back(gui_prop_file  = new GuiInput(VIEW_PROP, "lander_properties.dat"));
  controls.push_back(gui_prop_save  = new GuiButton(VIEW_PROP, "Save file", &gui_event));
  controls.push_back(gui_prop_txt1  = new GuiText(VIEW_PROP, "Unloaded lander mass (kg)"));
  controls.push_back(gui_prop_mass  = new GuiInput(VIEW_PROP, "", &Unloaded_Lander_Mass));
  controls.push_back(gui_prop_txt2  = new GuiText(VIEW_PROP, "Principle radii of gyration (m^2)"));
  controls.push_back(gui_prop_ixx   = new GuiInput(VIEW_PROP, "", &Ixx));
  controls.push_back(gui_prop_iyy   = new GuiInput(VIEW_PROP, "", &Iyy));
  controls.push_back(gui_prop_izz   = new GuiInput(VIEW_PROP, "", &Izz));
  controls.push_back(gui_prop_menu  = new GuiButton(VIEW_PROP, "Main menu", &gui_event));

  controls.push_back(gui_orbit_menu = new GuiButton(VIEW_ORBITAL, "Main menu", &gui_event));

#ifdef GRAPHING
  controls.push_back(gui_graph_show1 = new GuiToggle(VIEW_GRAPH, "Altitude", &(graph_series[0].visible), NULL, graph_colour(0, false), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(gui_graph_show2 = new GuiToggle(VIEW_GRAPH, "Climb speed", &(graph_series[1].visible), NULL, graph_colour(1, false), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(gui_graph_show3 = new GuiToggle(VIEW_GRAPH, "Ground speed", &(graph_series[2].visible), NULL, graph_colour(2, false), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(gui_graph_show4 = new GuiToggle(VIEW_GRAPH, "Euler angles", &(graph_series[3].visible), &gui_event, graph_colour(3, false), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(gui_graph_show5 = new GuiToggle(VIEW_GRAPH, "Ang velocity", &(graph_series[6].visible), &gui_event, graph_colour(6, false), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(gui_graph_csv  = new GuiButton(VIEW_GRAPH, "Save as", &gui_event));
  controls.push_back(gui_graph_csvp = new GuiInput(VIEW_GRAPH, "data.csv"));
  controls.push_back(gui_graph_menu = new GuiButton(VIEW_GRAPH, "Main menu", &gui_event));
#endif

  setup_autopilot();  // Allow the autopilot code to add controls to this view
  controls.push_back(gui_auto_menu  = new GuiButton(VIEW_AUTO, "Main menu", &gui_event));

  controls.push_back(instrument_auto = new GuiToggle(VIEW_INSTRUMENT, "Autopilot", &autopilot_enabled, NULL, vector3d(0.0, 0.5, 0.0), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(instrument_att  = new GuiToggle(VIEW_INSTRUMENT, "Attitude stabilizer", &stabilized_attitude, NULL, vector3d(0.0, 0.5, 0.0), vector3d(0.2, 0.2, 0.2)));
  controls.push_back(instrument_para = new GuiButton(VIEW_INSTRUMENT, "Parachute", &gui_event));
}

void gui_event (GuiControl *control) {
  if (control == gui_menu_prop) {
    set_multi_view(VIEW_PROP);
  } else if (control == gui_menu_orbit) {
    set_multi_view(VIEW_ORBITAL);
  } else if (control == gui_menu_graph) {
    set_multi_view(VIEW_GRAPH);
  } else if (control == gui_menu_auto) {
    set_multi_view(VIEW_AUTO);
  } else if (control == gui_menu_help) {
    help = true;
    save_orbital_zoom = orbital_zoom;
    orbital_zoom = 0.4;
    set_multi_view(VIEW_ORBITAL);
    if (paused || landed) refresh_all_subwindows();
  } else if (control == gui_menu_quit) {
    exit(0);
  } else if (control == gui_prop_menu ||
             control == gui_orbit_menu ||
             control == gui_graph_menu ||
             control == gui_auto_menu) {
    set_multi_view(VIEW_MENU);
  } else if (control == gui_prop_load) {
    if (!gui_prop_file->text.length()) {
      gui_prop_file->text = "lander_properties.dat";
    }
    bool res = load_lander_properties(exe_path + gui_prop_file->text);
    if (res) {
      gui_prop_mass->refresh();
      gui_prop_ixx->refresh();
      gui_prop_iyy->refresh();
      gui_prop_izz->refresh();
    } else {
      gui_prop_mass->update();
      gui_prop_ixx->update();
      gui_prop_iyy->update();
      gui_prop_izz->update();
    }
  } else if (control == gui_prop_save) {
    if (!gui_prop_file->text.length()) {
      gui_prop_file->text = "lander_properties.dat";
    }
    save_lander_properties(exe_path + gui_prop_file->text);
  }
#ifdef GRAPHING
  else if (control == gui_graph_show4) {
    graph_series[4].visible = graph_series[3].visible;
    graph_series[5].visible = graph_series[3].visible;
  } else if (control == gui_graph_show5) {
    graph_series[7].visible = graph_series[6].visible;
    graph_series[8].visible = graph_series[6].visible;
  } else if (control == gui_graph_csv) {
    if (!gui_graph_csvp->text.length()) {
      gui_graph_csvp->text = "data.csv";
    }
    save_graph_csv(exe_path + gui_graph_csvp->text);
  }
#endif
  else if (control == instrument_para) {
    if (!autopilot_enabled && !landed && (parachute_status == NOT_DEPLOYED)) parachute_status = DEPLOYED;
    if (!(paused || landed)) refresh_all_subwindows();
  }
}

#ifdef GRAPHING
bool save_graph_csv (const string &path)
 // Save the data currently displayed in graph view to a csv file
{
  ofstream file;
  bool good = false;
  
  cout << "Saving graph data to '" << path << "'" << endl;
  
  file.open(path.c_str());
  if (file.is_open()) {
    // Write csv header
    file << "\"Time\",\"Altitude\",\"Climb rate\",\"Ground speed\",\"Phi\",\"Theta\",\"Psi\",\"OmegaX\",\"OmegaY\",\"OmegaZ\"" << endl;
    
    // Full precision for float output
    file.precision(15);
    
    // Write data
    for (unsigned long i=0; i<n_points; i++) {
      file << (i + n_start) * delta_t;
      for (unsigned short s=0; s<N_SERIES; s++) {
        file << ',' << graph_series[s].data[i];
      }
      file << endl;
    }
    
    // Close file
    file.close();
    
    good = true;
  } else {
    cerr << "Error opening file '" << path << "' for writing" << endl;
  }
  return good;
}
#endif

bool load_lander_properties (const string & path)
{
  ifstream file;
  bool good = false;

  cout << "Loading lander properties from '" << path << "'" << endl;
  
  file.open(path.c_str());
  if (file.is_open()) {
    file >> Unloaded_Lander_Mass;
    file >> Ixx;
    file >> Iyy;
    file >> Izz;
    good = true;
  } else {
    cerr << "Error opening file '" << path << "' for reading" << endl;
  }
  return good;
}

bool save_lander_properties (const string & path)
{
  ofstream file;
  bool good = false;

  cout << "Saving lander properties to '" << path << "'" << endl;
  
  file.open(path.c_str());
  if (file.is_open()) {
    file << Unloaded_Lander_Mass << endl;
    file << Ixx << endl;
    file << Iyy << endl;
    file << Izz << endl;
    good = true;
  } else {
    cerr << "Error opening file '" << path << "' for writing" << endl;
  }
  return good;
}

void gui_autopilot_constant (double &d, const string &s)
{
  GuiText *new_text = new GuiText(VIEW_AUTO, s);
  GuiInput *new_input = new GuiInput(VIEW_AUTO, "", &d);
  
  auto_controls.push_back(new_text);
  auto_controls.push_back(new_input);
  
  controls.push_back(new_text);
  controls.push_back(new_input);
}
