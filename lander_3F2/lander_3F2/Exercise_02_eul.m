% Thruster positions (relative to lander's center)
%  For convenience, the matrix is constructed with
%  one thruster's xyz coordinates on each line.
p = [0.45, 0.45,-0.2;
     0.45,-0.45,-0.2;
    -0.45, 0.45,-0.2;
    -0.45,-0.45,-0.2]';

% Thruster directions (within lander's reference frame)
d = [0.00, 0.00,-1.00;
     0.00, 0.00,-1.00;
     0.00, 0.00,-1.00;
     0.00, 0.00,-1.00]';

% Normalize columns of d
sized = size(d);
for j=1:sized
  d(:,j) = d(:,j) ./ norm(d(:,j), 2);
end

% Matrix of cross-products
S = cross(p, d);

% Maximum thrust from each thruster (N)
T = 373.665;
% Calculate maximum vertical thrust
Tv = T * -sum(d(3,:))

% Principal moments of inertia of the lander
m   = 200;
Ixx = m*0.45;
Iyy = m*0.45;
Izz = m*0.3;
% Construct diagonal inertia matrix
I  = diag([Ixx, Iyy, Izz]);
Ii = diag(1./([Ixx, Iyy, Izz]));

alpha = (Iyy-Izz)/Ixx;
beta  = (Izz-Ixx)/Iyy;
gamma = (Ixx-Iyy)/Izz;

% Orientation Euler angles at equilibrium
ang = [0;pi/2;0];
phi = ang(1); theta = ang(2); psi = ang(3);

% Angular velocity at equilibrium
omega = [0;0;0];

% Construct the A matrix from four 3x3 blocks
A11 = [0, -(omega(1)*sin(psi)+omega(2)*cos(psi))/(tan(theta)*sin(theta)), (omega(1)*cos(psi)-omega(2)*sin(psi))/sin(theta);
       0, 0, -(omega(1)*sin(psi)+omega(2)*cos(psi));
       0, (omega(1)*sin(psi)+omega(2)*cos(psi))/(sin(theta)*sin(theta)), (-omega(1)*cos(psi)+omega(2)*sin(psi))/tan(theta)];
A12 = [sin(psi)/sin(theta), cos(psi)/sin(theta), 0;
       cos(psi),           -sin(psi),            0;
      -sin(psi)/tan(theta),-cos(psi)/tan(theta), 1];
A21 = [0, 0, 0;
       0, 0, 0;
       0, 0, 0];
A22 = [0,              alpha*omega(3), alpha*omega(2);
       beta*omega(3),  0,              beta*omega(1);
       gamma*omega(2), gamma*omega(1), 0];
A = [A11, A12;
     A21, A22]

% Construct the B matrix from two 3xn blocks
B11 = zeros(3, sized(2));
B21 = Ii * -T * S;
B = [B11;
     B21]

% Construct controllability matrix
P = [B, A*B, (A^2)*B, (A^3)*B, (A^4)*B, (A^5)*B];

fprintf('-- Controllability --\n\n');
rankP = rank(P);
fprintf('rank(P) = %u\n', rankP);
if rankP ~= max(size(A))
  disp('Uncontrollable state space basis:');
  disp(null(P'));
end
fprintf('\n');

% Desired closed-loop poles
poles = [-1, -1, -1, -1, -1, -1];

% Shift the poles for Matlab users
% NB: Students using GNU Octave do not need this line
poles = poles + 0.01*[-1, -1, -1, 1, 1, 1];

% Calculate the required gain matrix
K = place(A, B, poles);

fprintf('-- Gain matrix K (in C++ syntax) --\n\n');
% Print matrix K in a format that can be copied directly into C++
[M,N] = size(K);
fprintf('  double K[%u][%u] = {\n', M, N);
for j=1:M
  fprintf('    { ');
  for k=1:N
    fprintf('%8f', K(j, k));
    if k<N; fprintf(', '); end
  end
  if j<M; fprintf(' },\n'); end
end
fprintf(' }\n  };\n\n');
