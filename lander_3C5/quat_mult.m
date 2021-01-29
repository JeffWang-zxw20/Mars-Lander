function Q = quat_mult (Q1, Q2)

Q = [Q1(1)*Q2(1) - dot(Q1(2:4), Q2(2:4));
     Q1(1)*Q2(2:4) + Q2(1)*Q1(2:4) + cross(Q1(2:4), Q2(2:4))];
