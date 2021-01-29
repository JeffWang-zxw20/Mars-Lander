//
// Extended Mars Lander
// Version 1.0
// Dynamical simulation and autopilot functions
// Arthur Tombs, May 2013
//

// Original license:

// Mars lander simulator
// Version 1.6
// Mechanical simulation functions
// Gabor Csanyi and Andrew Gee, September 2011

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation, to make use of it
// for non-commercial purposes, provided that (a) its original authorship
// is acknowledged and (b) no modified versions of the source code are
// published. Restriction (b) is designed to protect the integrity of the
// exercise for future generations of students. The authors would be happy
// to receive any suggested modifications by private correspondence to
// ahg@eng.cam.ac.uk and gc121@eng.cam.ac.uk.

#include "lander.h"

// STUDENTS -- Set default autopilot values here
double k_p = 0.1;
double k_d = 0.0;
double z_e = 500.0;

void setup_autopilot (void)
{
  gui_autopilot_constant(k_p, "Proportional gain k_p");
  gui_autopilot_constant(k_d, "Derivative gain k_d");
  gui_autopilot_constant(z_e, "Demand altitude (m)");
}

void autopilot (void)
  // Autopilot to adjust the engine throttle, parachute and attitude control
{
  const double m = fuel * FUEL_CAPACITY * FUEL_DENSITY + Unloaded_Lander_Mass;

  // Altitude control

  // Force due to gravity at the current altitude
  double Fg = (-GRAVITY * MARS_MASS * m) / position.abs2();

  // Equilibrium thrust to counter gravity
  double t_e = -Fg / (MAX_THRUST*N_THRUSTERS);

  // Current altitude and altitude error
  double z = position.abs() - MARS_RADIUS;
  double delta_z = z_e - z;

  // Rate of change of altitude
  double z_dot = velocity * position.norm();
  double delta_z_dot = 0.0 - z_dot;

  // Equilibrium Euler angles and error
  vector3d E = vector3d(0.0, M_PI*0.5, 0.0);
  vector3d delta_E = E - euler_angles;

  // Separate out the component angles
  double delta_phi = delta_E.x, delta_theta = delta_E.y, delta_psi = delta_E.z;

  // Error in angular velocity
  vector3d delta_omega = -angular_velocity;

  // Use feedback gains to set throttle
  double t = t_e +
             k_p * delta_z +
             k_d * delta_z_dot;

  // Clip the throttle to between 0.0 and 1.0
  if (t > 1.0) t = 1.0;
  else if (t < 0.0) t = 0.0;

  // Set all throttles to the same value
  for (int i=0; i<N_THRUSTERS; i++) {
    throttle[i] = t;
  }

}

void numerical_dynamics (void)
  // This is the function that performs the numerical integration to update the
  // lander's pose. The time step is delta_t (global variable).
{
  static vector3d previous_position;
  static quat_t previous_orientation;
  
  vector3d &omega = angular_velocity;
  
  vector3d acceleration, omega_dot;
  calculate_accelerations(position, velocity, orientation, omega, acceleration, omega_dot);
  
  vector3d v_half = velocity + acceleration * (delta_t * 0.5);
  vector3d omega_half = omega + omega_dot * (delta_t * 0.5);
  
  position += v_half * delta_t;
  //orientation = (orientation + (orientation * quat_t(0.0, omega_half * (delta_t * 0.5)))).norm();
  
  calculate_accelerations(position, velocity, orientation, omega + omega_dot*delta_t, acceleration, omega_dot);
  
  velocity = v_half + acceleration * (delta_t * 0.5);
  //angular_velocity = omega_half + omega_dot * (delta_t * 0.5);
}

void calculate_accelerations (const vector3d &r, const vector3d &v, const quat_t &q, const vector3d &omega, vector3d &v_dot, vector3d &omega_dot)
 // Calculate accelerations of position and orientation
{
  vector3d force, torque;
  
  const double mass = fuel * FUEL_CAPACITY * FUEL_DENSITY + Unloaded_Lander_Mass;
  const double A = Ixx * mass, B = Iyy * mass, C = Izz * mass;
  
  // Calculate aerodynamic drag forces
  calculate_drag(r, v, q, omega, force, torque);
  
  // Add lander thrust
  force  += thrust_wrt_world(q);
  torque += torque_from_thrusters();
  
  // Calculate linear acceleration using Newton's Second Law
  // and adding gravitational component
  v_dot = (force / mass) + ((-GRAVITY * MARS_MASS) / r.abs2()) * r.norm();
  
  //
  // Calculate angular acceleration using Newton-Euler equations
  // STUDENTS -- Here you need to calculate the angular acceleration vector omega_dot
  //
  // NB: C++ does not respect 'natural' ordering for overloaded operators
  //     The expression a^b+c does NOT give the same result as (a^b)+c
  //
}

vector3d torque_from_thrusters (void)
  // Works out the torque vector in the lander reference frame, due to off-axis thrusters
{
  vector3d torque;
  //
  // STUDENTS -- You need to write the contents of this function
  //             Use a for loop to sum the torques from all thrusters
  //
  return torque;
}

void calculate_drag (const vector3d &r, const vector3d &v, const quat_t &q, const vector3d &omega, vector3d &force, vector3d &torque)
 // Calculate drag force and torque acting on the lander
{
  const double drag_coef_skin = 0.2;

  double density = atmospheric_density(r);

  force = vector3d(); torque = vector3d();

  // If outside the atmosphere, return with no force
  if (density == 0.0) return;

  // Transform wind velocity into lander reference frame
  // STUDENTS -- For the extension activities, you may want to set
  //             a world_wind here based on planetary rotation.
  vector3d world_wind = vector3d(0.0, 0.0, 0.0);

  vector3d wind = q.transform(world_wind - v);

  // Add on parachute drag, if deployed
  if (parachute_status == DEPLOYED) {
    // Add on term due to attachment point being rotated
    vector3d rv = v + (-q).transform(omega ^ parachute_attachment);

    force += 0.5*DRAG_COEF_CHUTE*density*(M_PI*CHUTE_SIZE*CHUTE_SIZE)*rv.abs2() * -rv.norm();
    torque = parachute_attachment ^ (force.abs() * wind.norm());
  }

  // Add bluff body drag and skin friction
  vector3d local_force = vector3d();

  for (vector<aero_element_t>::iterator it=aero.begin(); it<aero.end(); it++) {

    // Wind vector relative to surface element
    // taking lander spin into account
    vector3d W = wind - (omega ^ (*it).pos);

    // Only consider faces pointing into the wind
    double f = it->dA * W;
    if (f < 0) {

      vector3d N = it->dA.norm();

      // Velocity component tangential to surface
      vector3d Wt = W - ((W * N) * N);

      vector3d bluff = -0.5*DRAG_COEF_LANDER*density*(it->dA * W.norm())*W.abs2() * W.norm();
      vector3d skin = 0.5*drag_coef_skin*density*it->dA.abs()*Wt.abs2() * Wt.norm();

      local_force += (bluff + skin);
      torque += it->pos ^ (bluff + skin);
    }
  }

  // Add on drag force referred to global frame
  force += (-q).transform(local_force);

}

void initialize_simulation (void)
  // Lander pose initialization - selects one of 10 possible scenarios
{
  // The parameters to set are:
  // position - in Cartesian planetary coordinate system (m)
  // velocity - in Cartesian planetary coordinate system (m/s)
  // orientation - in global coordinate system (unit quaternion)
  // angular_velocity - in lander coordinate sysyem (rad/s)
  // delta_t - the simulation time step
  // boolean state variables - parachute_status, stabilized_attitude, autopilot_enabled
  // scenario_description - a descriptive string for the help screen

  scenario_description[0] = "circular orbit";
  scenario_description[1] = "descent from 10km";
  scenario_description[2] = "elliptical orbit, thrust changes orbital plane";
  scenario_description[3] = "polar launch at escape velocity (but drag prevents escape)";
  scenario_description[4] = "elliptical orbit that clips the atmosphere and decays";
  scenario_description[5] = "descent from 200km";
  scenario_description[6] = "tumbling descent from 10km";
  scenario_description[7] = "an initially perturbed hover at 500m altitude";
  scenario_description[8] = "a long distance from Mars";
  scenario_description[9] = "";

  // Default thruster positions and directions
  thruster_position[0] = LANDER_SIZE*vector3d(0.45, 0.45, -0.2);
  thruster_direction[0] = vector3d(0.0, 0.0, -1.0);

  thruster_position[1] = LANDER_SIZE*vector3d(0.45, -0.45, -0.2);
  thruster_direction[1] = vector3d(0.0, 0.0, -1.0);

  thruster_position[2] = LANDER_SIZE*vector3d(-0.45, 0.45, -0.2);
  thruster_direction[2] = vector3d(0.0, 0.0, -1.0);

  thruster_position[3] = LANDER_SIZE*vector3d(-0.45, -0.45, -0.2);
  thruster_direction[3] = vector3d(0.0, 0.0, -1.0);

  // Parachute attachment point
  parachute_attachment = vector3d(0.0, 0.0, LANDER_SIZE*0.5);

  switch (scenario) {

  case 0:
    // a circular equatorial orbit
    position = vector3d(1.2*MARS_RADIUS, 0.0, 0.0);
    velocity = vector3d(0.0, -3247.087385863725, 0.0);
    orientation = quat_t(0.7071067811865476, 0.0, 0.7071067811865476, 0.0);
    angular_velocity = vector3d();
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 1:
    // a descent from rest at 10km altitude
    position = vector3d(0.0, -(MARS_RADIUS + 10000.0), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = quat_t(0.7071067811865476, 0.7071067811865476, 0.0, 0.0);
    angular_velocity = vector3d();
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = false;
    break;

  case 2:
    // an elliptical polar orbit
    position = vector3d(0.0, 0.0, 1.2*MARS_RADIUS);
    velocity = vector3d(3500.0, 0.0, 0.0);
    orientation = quat_t(0.7071067811865476, 0.7071067811865476, 0.0, 0.0);
    angular_velocity = vector3d();
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 3:
    // polar surface launch at escape velocity (but drag prevents escape)
    position = vector3d(0.0, 0.0, MARS_RADIUS + LANDER_SIZE/2.0);
    velocity = vector3d(0.0, 0.0, 5027.0);
    orientation = quat_t();
    angular_velocity = vector3d();
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 4:
    // an elliptical orbit that clips the atmosphere each time round, losing energy
    position = vector3d(0.0, 0.0, MARS_RADIUS + 100000.0);
    velocity = vector3d(4000.0, 0.0, 0.0);
    orientation = quat_t(0.7071067811865476, 0.0, 0.7071067811865476, 0.0);
    angular_velocity = vector3d();
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 5:
    // a descent from rest at the edge of the exosphere
    position = vector3d(0.0, -(MARS_RADIUS + EXOSPHERE), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = quat_t(0.7071067811865476, 0.7071067811865476, 0.0, 0.0);
    angular_velocity = vector3d();
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = false;
    break;

  case 6:
    // a tumbling descent from 10km altitude
    position = vector3d(0.0, -(MARS_RADIUS + 10000.0), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = quat_t();
    angular_velocity = vector3d(1.0, 0.4, 2.1);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 7:
    // an initially perturbed hover at 500m altitude
    position = vector3d(0.0, -(MARS_RADIUS + 550.0), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = quat_t(0.8, 0.7, 0.1, 0.0).norm();
    angular_velocity = vector3d(0.05, 0.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    // STUDENTS -- Change this variable to false to give the
    //             autopilot an initial error to correct
    stabilized_attitude = true;
    autopilot_enabled = true;
    break;

  case 8:
    // a long way away from Mars
    position = vector3d(1000*MARS_RADIUS, 0.0, 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = quat_t(1.0, 0.0, 0.0, 0.0);
    angular_velocity = vector3d(0.8, 0.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    
    // Disable all but one thruster
    thruster_direction[0] = vector3d();
    thruster_direction[2] = vector3d();
    thruster_direction[3] = vector3d();
    break;

  case 9:
    break;

  }
}
