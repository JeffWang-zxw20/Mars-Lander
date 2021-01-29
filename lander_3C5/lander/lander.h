//
// Extended Mars Lander
// Version 1.0
// Header file
// Arthur Tombs, May 2013
//

// Original license:

// Mars lander simulator
// Version 1.6
// Header file
// Gabor Csanyi and Andrew Gee, September 2011

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation, to make use of it
// for non-commercial purposes, provided that (a) its original authorship
// is acknowledged and (b) no modified versions of the source code are
// published. Restriction (b) is designed to protect the integrity of the
// exercise for future generations of students. The authors would be happy
// to receive any suggested modifications by private correspondence to
// ahg@eng.cam.ac.uk and gc121@eng.cam.ac.uk.

// Some reports suggest that Dev-C++/MinGW does not define WIN32
#if defined (__MINGW32__) && !defined (WIN32)
#define WIN32
#endif

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <windows.h>
#define copysign _copysign
#define PATH_SEP '\\'
#else
#include <sys/time.h>
#include <unistd.h>
#define PATH_SEP '/'
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#define ASCII_BACKSPACE 127
#define ASCII_DELETE 8
#else
#include <GL/glut.h>
#define ASCII_BACKSPACE 8
#define ASCII_DELETE 127
#endif
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <deque>

// GLUT mouse wheel operations work under Linux only
#if !defined (GLUT_WHEEL_UP)
#define GLUT_WHEEL_UP 3
#define GLUT_WHEEL_DOWN 4
#endif

// Comment out this line to disable graph drawing
#define GRAPHING

// Graphics constants
#define GAP 5
#define SMALL_NUM 0.0000001
#define N_RAND 20000
#define PREFERRED_WIDTH 1024
#define PREFERRED_HEIGHT 768
#define MIN_INSTRUMENT_WIDTH 1024
#define INSTRUMENT_HEIGHT 300
#define GROUND_LINE_SPACING 20.0
#define CLOSEUP_VIEW_ANGLE 30.0
#define TRANSITION_ALTITUDE 10000.0
#define TRANSITION_ALTITUDE_NO_TEXTURE 4000.0
#define TERRAIN_TEXTURE_SIZE 1024
#define LANDER_TEXTURE_SIZE 128
#define PLANET_TEXTURE_SIZE 512
#define INNER_DIAL_RADIUS 65.0
#define OUTER_DIAL_RADIUS 75.0
#define MAX_DELAY 160000
#define N_TRACK 1000
#define TRACK_DISTANCE_DELTA 100000.0
#define TRACK_ANGLE_DELTA 0.999
#define HEAT_FLUX_GLOW_THRESHOLD 1000000.0

// Mars constants
#define MARS_RADIUS 3386000.0 // (m)
#define MARS_MASS 6.42E23 // (kg)
#define GRAVITY 6.673E-11 // (m^3/kg/s^2)
#define MARS_DAY 88642.65 // (s)
#define EXOSPHERE 200000.0 // (m)

// Lander constants
#define LANDER_SIZE 1.0 // (m)
#define CHUTE_SIZE 3.0 // (m)
#define UNLOADED_LANDER_MASS 100.0 // (kg)
#define IXX ((3 * ((LANDER_SIZE * LANDER_SIZE) / 4 + (LANDER_SIZE * LANDER_SIZE) / 2)) / 5)
#define IYY IXX
#define IZZ ((3 * LANDER_SIZE * LANDER_SIZE) / 10)
#define FUEL_CAPACITY 100.0 // (l)
#define FUEL_RATE_AT_MAX_THRUST 0.5 // (l/s)
#define FUEL_DENSITY 1.0 // (kg/l)
#define N_THRUSTERS 4
#define THRUSTER_SIZE (LANDER_SIZE*0.1)
// MAX_THRUST, as defined below, is 0.5 * weight of fully loaded lander at surface
#define MAX_THRUST (0.5 * (FUEL_DENSITY*FUEL_CAPACITY+UNLOADED_LANDER_MASS) * (GRAVITY*MARS_MASS/(MARS_RADIUS*MARS_RADIUS))) // (N)
#define ENGINE_LAG 0.0 // (s)
#define ENGINE_DELAY 0.0 // (s)
#define DRAG_COEF_CHUTE 2.0
#define DRAG_COEF_LANDER 1.0
#define MAX_PARACHUTE_DRAG 20000.0 // (N)
#define MAX_PARACHUTE_SPEED 500.0 // (m/s)
#define THROTTLE_GRANULARITY 20 // for manual control
#define MAX_IMPACT_GROUND_SPEED 1.0 // (m/s)
#define MAX_IMPACT_DESCENT_RATE 1.0 // (m/s)

using namespace std;

class vector3d {
  // Utility class for three-dimensional vector operations
public:
  vector3d (void) : x(0.0), y(0.0), z(0.0) {}
  vector3d (double a, double b, double c=0.0) : x(a), y(b), z(c) {}
  vector3d (int a, int b, int c=0) : x(a), y(b), z(c) {}
  
  bool operator== (const vector3d &v) const { return ((x==v.x)&&(y==v.y)&&(z==v.z)); }
  bool operator!= (const vector3d &v) const { return ((x!=v.x)||(y!=v.y)||(z!=v.z)); }
  vector3d operator+ (const vector3d &v) const { return vector3d(x+v.x, y+v.y, z+v.z); }
  vector3d operator- (const vector3d &v) const { return vector3d(x-v.x, y-v.y, z-v.z); }
  friend vector3d operator- (const vector3d &v) { return vector3d(-v.x, -v.y, -v.z); }
  vector3d& operator+= (const vector3d &v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
  vector3d& operator-= (const vector3d &v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
  vector3d operator^ (const vector3d &v) const { return vector3d(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x); }
  double operator* (const vector3d &v) const { return (x*v.x + y*v.y +z*v.z); }
  friend vector3d operator* (const vector3d &v, const double &a) { return vector3d(v.x*a, v.y*a, v.z*a); }
  friend vector3d operator* (const double &a, const vector3d &v) { return vector3d(v.x*a, v.y*a, v.z*a); }
  vector3d& operator*= (const double &a) { x*=a; y*=a; z*=a; return *this; }
  vector3d operator/ (const double &a) const { return vector3d(x/a, y/a, z/a); }
  vector3d& operator/= (const double &a) { x/=a; y/=a; z/=a; return *this; }
  inline double abs2() const { return (x*x + y*y + z*z); }
  inline double abs() const { return sqrt(this->abs2()); }
  vector3d norm() const { double s(this->abs()); if (s==0) return *this; else return vector3d(x/s, y/s, z/s); }
  friend ostream& operator << (ostream &out, const vector3d &v) { out << v.x << ' ' << v.y << ' ' << v.z; return out; }
  
  double x, y, z;
private:
};

// Macro for putting vectors into OpenGL functions
#define XYZ(X) X.x,X.y,X.z

// Data type for recording lander's previous positions
struct track_t {
  unsigned short n;
  unsigned short p;
  vector3d pos[N_TRACK];
};

// Quaternions for lander pose and orbital view transformation
struct quat_t {
  double s;
  vector3d v;
  
  quat_t (void) : s(1.0), v() {}
  quat_t (double d0, double d1, double d2, double d3) : s(d0), v(d1, d2, d3) {}
  quat_t (double a, const vector3d &b) : s(a), v(b) {}
  
  inline double abs2 (void) const { return (s*s + v.abs2()); }
  inline double abs (void) const { return sqrt(this->abs2()); }
  quat_t norm (void) const { double s(this->abs()); if (s==0.0) return *this; return (*this)/s; }
  
  vector3d transform (const vector3d &x) const { vector3d c=x^v; return ((v*x)*v) + ((s*s)*x) + ((s+s)*c) + (c^v); }
  
  bool operator== (const quat_t &q) const { return ((s==q.s)&&(v==q.v)); }
  bool operator!= (const quat_t &q) const { return ((s!=q.s)||(v!=q.v)); }
  quat_t operator+ (const quat_t &q) const { return quat_t(s+q.s, v+q.v); }
  quat_t operator- (const quat_t &q) const { return quat_t(s-q.s, v-q.v); }
  quat_t operator* (const quat_t &q) const { return quat_t((s*q.s)-(v*q.v), (v*q.s)+(s*q.v)+(v^q.v)); }
  friend quat_t operator- (const quat_t &q) { return quat_t(q.s, -q.v) / q.abs2(); }
  friend quat_t operator* (const quat_t &q, const double &a) { return quat_t(q.s*a, q.v*a); }
  friend quat_t operator* (const double &a, const quat_t &q) { return quat_t(q.s*a, q.v*a); }
  quat_t operator/ (const double &a) const { return quat_t(s/a, v/a); }
  friend ostream& operator << (ostream &out, const quat_t &q) { out << q.s << ' ' << q.v; return out; }
};

// Data structure for the state of the close-up view's coordinate system
struct closeup_coords_t {
  bool initialized;
  vector3d up, right;
};

// Enumerated data type for parachute status
enum parachute_status_t { NOT_DEPLOYED = 0, DEPLOYED = 1, LOST = 2 };

// Struct for storing aero elements
struct aero_element_t {
  vector3d dA, pos;
  
  aero_element_t (const vector3d &v1, const vector3d &v2) : dA(v1), pos(v2) {}
};

typedef enum{VIEW_INVALID, VIEW_PROP, VIEW_ORBITAL, VIEW_GRAPH, VIEW_MENU, VIEW_AUTO, VIEW_INSTRUMENT} ViewState;

typedef enum{GUI_MOUSE_DOWN, GUI_MOUSE_UP, GUI_MOUSE_DRAG, GUI_KEY, GUI_SPECIAL} GuiEvent;

// Alignment types for text rendering in glut_print
typedef enum{ALIGN_LEFT, ALIGN_MID, ALIGN_RIGHT} Alignment;

class GuiControl {
public:
  int x, y, width, height;
  const ViewState view;
  void (*callback)(GuiControl*);

  void do_callback (void) { if (callback != NULL) (*callback)(this); }

  GuiControl (ViewState v, void (*c)(GuiControl*) = NULL)
             : x(1), y(1), width(100), height(20), view(v), callback(c) { }
  void setPosition (int nx, int ny) { x = max(nx,1); y = max(ny,1); }
  void setSize (int nw, int nh) { width = max(nw,1); height = max(nh,1); }
  bool hitTest (int mx, int my) const { return (mx >= x && mx <= (x+width) && my >= y && my <= (y+height)); }
  virtual void event (GuiEvent, int=0, int=0) { }
  virtual void render (bool = false) const = 0;
};

class GuiText : public GuiControl {
public:
  string text;
  void *font;
  Alignment align;

  GuiText (ViewState v, const string &s, void *f = GLUT_BITMAP_HELVETICA_10, Alignment a = ALIGN_LEFT)
          : GuiControl(v), text(s), font(f), align(a) { }
  void render (bool = false) const;
};

class GuiButton : public GuiControl {
public:
  string text;
  vector3d col1;

  GuiButton (ViewState v, const string &s, void (*c)(GuiControl*) = NULL,
             const vector3d &c1 = vector3d(0.1, 0.1, 0.3))
            : GuiControl(v, c), text(s), col1(c1) { }
  void event (GuiEvent, int=0, int=0);
  void render (bool = false) const;
};

class GuiToggle : public GuiButton {
public:
  bool * val;
  vector3d col2;

  GuiToggle (ViewState v, const string &s, bool *b, void (*c)(GuiControl*) = NULL,
             const vector3d &c1 = vector3d(0.0, 0.5, 0.0), const vector3d &c2 = vector3d(0.0, 0.5, 0.0))
            : GuiButton(v, s, c, c1), val(b), col2(c2) { }
  void event (GuiEvent, int=0, int=0);
  void render (bool = false) const;
};

class GuiInput : public GuiControl {
public:
  string text, tmptext;
  unsigned short pos;
  double *val;

  void refresh (void);
  void update (void);

  GuiInput (ViewState v, const string &s, double *d=NULL, void (*c)(GuiControl*) = NULL)
           : GuiControl(v, c), text(s), pos(s.length()), val(d) { refresh(); }
  void event (GuiEvent, int=0, int=0);
  void render (bool = false) const;
};

#ifdef DECLARE_GLOBAL_VARIABLES // actual declarations of all global variables for lander_graphics.cpp

string exe_path;

// GL windows and objects
int main_window, closeup_window, multi_window, instrument_window, view_width, view_height, win_width, win_height;
GLUquadricObj *quadObj;
GLuint terrain_texture, lander_texture, planet_texture1, planet_texture2;
short throttle_control;
track_t track;
bool texture_available;

ViewState multi_view_state;

// Vector of pointers to GUI objects
vector<GuiControl*> controls;
GuiControl *gui_focus = NULL;

GuiText   *gui_prop_txt1, *gui_prop_txt2,
          *gui_auto_constl[3];
GuiButton *gui_menu_prop, *gui_menu_orbit, *gui_menu_graph, *gui_menu_auto, *gui_menu_help, *gui_menu_quit,
          *gui_prop_load, *gui_prop_save, *gui_prop_menu,
          *gui_orbit_menu,
          *gui_graph_csv, *gui_graph_menu,
          *gui_auto_on, *gui_auto_menu;
GuiInput  *gui_prop_file, *gui_prop_mass, *gui_prop_ixx, *gui_prop_iyy, *gui_prop_izz,
          *gui_graph_csvp;
GuiToggle *gui_graph_show1, *gui_graph_show2, *gui_graph_show3, *gui_graph_show4, *gui_graph_show5;

GuiToggle *instrument_auto, *instrument_att;
GuiButton *instrument_para;

vector<GuiControl*> auto_controls;

#ifdef GRAPHING

#define N_SERIES 9
#define N_POINTS 100000

struct graph_series_t {
  bool visible;
  std::deque<unsigned long> maxima, minima;
  float data[N_POINTS];
};

// Number of points stored, and number of points skipped due to moving window
unsigned long n_points = 0, n_start = 0;

graph_series_t graph_series[N_SERIES];

#endif

// Simulation parameters
bool help = false;
bool paused = false;
bool landed = false;
bool crashed = false;
int last_click_x = -1;
int last_click_y = -1;
short simulation_speed = 5;
double delta_t, simulation_time;
unsigned short scenario = 0;
string scenario_description[10];
bool static_lighting = false;
closeup_coords_t closeup_coords;
float randtab[N_RAND];
bool do_texture = true;
vector3d thruster_position[N_THRUSTERS], thruster_direction[N_THRUSTERS];
vector3d parachute_attachment;
unsigned long long time_program_started;

// Lander state - the visualization routines use velocity_from_positions, so not sensitive to 
// any errors in the velocity update in numerical_dynamics
vector3d position, velocity, angular_velocity, velocity_from_positions, last_position;
quat_t orientation;
vector3d euler_angles;
double climb_speed, ground_speed, altitude, throttle[N_THRUSTERS], fuel;
bool stabilized_attitude, autopilot_enabled, parachute_lost;
parachute_status_t parachute_status;
int stabilized_attitude_angle;

double Ixx = IXX, Iyy = IYY, Izz = IZZ;
double Unloaded_Lander_Mass = UNLOADED_LANDER_MASS;
vector<aero_element_t> aero;

// Orbital and closeup view parameters
double orbital_zoom, save_orbital_zoom, closeup_offset, closeup_xr, closeup_yr, terrain_angle;
quat_t orbital_quat;

// For GL lights
GLfloat plus_y[] = { 0.0, 1.0, 0.0, 0.0 };
GLfloat minus_y[] = { 0.0, -1.0, 0.0, 0.0 };
GLfloat plus_z[] = { 0.0, 0.0, 1.0, 0.0 };
GLfloat top_right[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat straight_on[] = { 0.0, 0.0, 1.0, 0.0 };

#else // extern declarations of those global variables used in lander.cpp

extern vector3d thruster_position[], thruster_direction[];
extern vector3d parachute_attachment;
extern bool stabilized_attitude, autopilot_enabled;
extern double delta_t, simulation_time, throttle[], fuel;
extern unsigned short scenario;
extern string scenario_description[];
extern vector3d position, velocity, angular_velocity;
extern quat_t orientation;
extern vector3d euler_angles;
extern parachute_status_t parachute_status;
extern int stabilized_attitude_angle;

extern double Ixx, Iyy, Izz;
extern double Unloaded_Lander_Mass;
extern vector<aero_element_t> aero;

#endif

// Function prototypes
void invert (double m[], double mout[]);
vector3d matrix_mult (double m[], vector3d v);
quat_t matrix_to_quat (double m[]);
vector3d matrix_to_zxz_eul (double m[]);
quat_t axis_to_quat (const vector3d &a, const double phi);
double project_to_sphere (const double r, const double x, const double y);
void quat_to_matrix (double m[], const quat_t &q);
quat_t track_quats (const double p1x, const double p1y, const double p2x, const double p2y);
void microsecond_time (unsigned long long &t);
void fghCircleTable (double **sint, double **cost, const int n);
void glutOpenHemisphere (GLdouble radius, GLint slices, GLint stacks);
void glutMottledSphere (GLdouble radius, GLint slices, GLint stacks);
void drawTexturedSphere (GLdouble radius, GLint slices, GLint stacks);
void glutCone (GLdouble base, GLdouble height, GLint slices, GLint stacks, bool closed);
void enable_lights (void);
void setup_lights (void);
void glut_print (float x, float y, const string &s, void *font = GLUT_BITMAP_HELVETICA_10, Alignment align = ALIGN_LEFT);
double atmospheric_density (const vector3d &pos);
void draw_dial (double cx, double cy, double val, string title, string units);
void draw_control_bar (double tlx, double tly, double val, double red, double green, double blue, string title);
void draw_indicator_lamp (double tcx, double tcy, string off_text, string on_text, bool on);
void draw_instrument_window (void);
void display_help_arrows (void);
void display_help_prompt (void);
void display_help_text (void);
void draw_multi_window (void);
void draw_orbital_window (void);
void draw_gui (void);
void draw_parachute (double d);
void update_closeup_coords (void);
void draw_closeup_window (void);
void draw_main_window (void);
void refresh_all_subwindows (void);
bool safe_to_deploy_parachute (void);
void update_visualization (void);
void attitude_stabilization (void);
vector3d thrust_wrt_world (const quat_t &ori = orientation);
vector3d torque_from_thrusters (void);
void setup_autopilot (void);
void autopilot (void);
void numerical_dynamics (void);
void calculate_accelerations (const vector3d &r, const vector3d &v, const quat_t &q, const vector3d &omega, vector3d &v_dot, vector3d &omega_dot);
void calculate_drag (const vector3d &r, const vector3d &v, const quat_t &q, const vector3d &omega, vector3d &force, vector3d &torque);
void initialize_simulation (void);
void update_lander_state (void);
void reset_simulation (void);
void set_multi_projection_matrix (void);
void reshape_main_window (int width, int height);
void multi_mouse_button (int button, int state, int x, int y);
void multi_mouse_motion (int x, int y);
void closeup_mouse_button (int button, int state, int x, int y);
void closeup_mouse_motion (int x, int y);
void glut_special (int key, int x, int y);
void glut_key (unsigned char k, int x, int y);

#ifdef GRAPHING
vector3d graph_colour (unsigned int series, bool set = true);
void draw_graph_window (void);
bool save_graph_csv (const string &path);
#endif

void set_multi_view (ViewState new_state);
void setup_gui (void);
void gui_event (GuiControl*);

void gui_autopilot_constant (double&, const string &);

bool load_lander_properties (const string & path);
bool save_lander_properties (const string & path);

void setup_aero_mesh (void);

unsigned char *load_texture_file (const string &path, GLsizei w, GLsizei h, GLint channels);
bool prepare_texture (GLuint id, unsigned char *data, GLsizei w, GLsizei h, GLint channels);
bool setup_terrain_texture (void);
bool setup_lander_texture (void);
bool setup_planet_texture (GLuint &id);
bool setup_texture (const string &name, GLsizei size, GLint channels, GLuint &id);

void draw_lander (void);
void draw_frustrum (GLdouble base, GLdouble top, GLdouble height, GLint slices, GLint stacks, bool closed);
void draw_flare (GLdouble base, GLdouble height, GLint slices, GLint stacks);
void look_in_direction (const vector3d &v);
