function [R,Ri] = eul_to_matrix (E)

phi = E(1); theta = E(2); psi = E(3);

R1 = [cos(phi), sin(phi), 0;
     -sin(phi), cos(phi), 0;
      0,        0,        1];

R2 = [1, 0,          0;
      0, cos(theta), sin(theta);
      0,-sin(theta), cos(theta)];

R3 = [cos(psi), sin(psi), 0;
     -sin(psi), cos(psi), 0;
      0,        0,        1];

R = R3 * R2 * R1;
Ri = R';
