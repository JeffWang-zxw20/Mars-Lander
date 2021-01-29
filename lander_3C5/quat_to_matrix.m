function [R,Ri] = quat_to_matrix (Q)

s = Q(1); q1 = Q(2); q2 = Q(3); q3 = Q(4);

R = [s*s+q1*q1-q2*q2-q3*q3, 2*(q1*q2+s*q3), 2*(q1*q3-s*q2);
       2*(q1*q2-s*q3), s*s-q1*q1+q2*q2-q3*q3, 2*(s*q1+q2*q3);
       2*(s*q2+q1*q3), 2*(q2*q3-s*q1), s*s-q1*q1-q2*q2+q3*q3];

Ri = R';
