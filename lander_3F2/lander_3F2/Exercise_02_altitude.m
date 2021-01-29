% Maximum thrust from all thrusters combined (N)
Tv = 1494.7;

% Mass of lander (initial)
m = 200;

% State-space matrices
% STUDENTS - Enter A and B matrices as they appear in the handout
A = [   ;
        ];
B = [     ;
         ];

% Desired pole locations
% STUDENTS - Put the desired eigenvalue locations here
%            Complex values may be entered as a + i * b
poles = [   ];

% Calculate gain matrix
K = place(A, B, poles);

% Print the resulting gain values
fprintf('k1 = %f\nk2 = %f\n', K(1), K(2));
