% Define some global variables to share with functions
global I Ii F r;

% Define moments of inertia and compose inertia matrix I
A = 2;
B = 2;
C = 4;

% Compute the inertia matrix and its inverse
I  = diag([A,B,C]);
Ii = diag(1 ./ [A,B,C]);

% Define initial conditions for the simulation
E     = [0;pi/2;0];  % Orientation - ZXZ Euler angles (rad)
omega = [0;0;4];     % Body-fixed angular-velocity (rad/s)

% Define simulation parameters
dt = 0.01;  % Time per iteration
N  = 1000;  % Number of iterations

% Constant force acting on the body
F = [0;0;-10];  % Force acting, in global coordinates
r = [0;0;1];    % Point of action, in body coordinates

% Track z-axis over time
track = zeros(N,3);

% Construct rotation matrix R from orientation
R = eul_to_matrix(E);

% Loop over all time steps
for n = 1:N

  phi = E(1); theta = E(2); psi = E(3);

  % Store z-axis at this time step
  track(n,:) = R(3,:);

  if sin(theta) == 0; disp('Gimbal lock!'); break; end

  % Construct matrix J to convert from AV to Euler angles
  J  = [sin(psi)/sin(theta),  cos(psi)/sin(theta), 0;
        cos(psi),            -sin(psi),            0;
       -sin(psi)/tan(theta), -cos(psi)/tan(theta), 1];

  % Calculate angular accelerations
  omega_dot = calculate_accelerations(R, omega);

  % Calculate angular velocity at half timestep
  omega_half = omega + omega_dot * (dt * 0.5);

  % Calculate new orientation estimate
  E_dot  = J * omega_half;
  E = E + E_dot * dt;

  % Construct rotation matrix R from orientation
  R = eul_to_matrix(E);

  % Calculate new angular accelerations
  omega_dot = calculate_accelerations(R, omega + omega_dot * dt);

  % Calculate updated angular velocity estimate
  omega = omega_half + omega_dot * (dt * 0.5);

end

% 3D plot of z-axis over time
figure(1);
clf;
plot3(track(1:2:n,1), track(1:2:n,2), track(1:2:n,3));
axis([-1 1 -1 1 -1 1]);
