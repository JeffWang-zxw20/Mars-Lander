function omega_dot = calculate_accelerations (R, omega)

% Inertia matrix from global scope
global I Ii F r;

% Calculate torque acting
Q = cross(r, R*F);

% Matrix form of Euler's equations
omega_dot = Ii * (cross(I*omega, omega) + Q);
