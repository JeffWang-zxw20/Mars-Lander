# uncomment the next line if running in a notebook
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

# ---------------- Physical constants ----------------
# mass of the body (lander)
m = 1
# mass of Mars, gravitational constant, Mars radius
M = 6.42e23
G = 6.6743e-11
R_MARS = 3.3895e6  # Mars mean radius [m]
mu = G * M

# ---------------- Assignment 2: Scenario selector ----------------
# We place Mars at the origin. The body starts on the +z axis at height ALTITUDE above
# the surface. The initial velocity for Scenarios 2–4 is perpendicular to the position
# vector, as required by the handout.
#
# SCENARIO meanings:
#   1: Straight down descent      (v0 = 0)
#   2: Circular orbit             (|v0| = sqrt(mu/r0), v0 ⟂ x0)
#   3: Elliptical orbit           (|v0| = beta*sqrt(mu/r0), beta∈(0,sqrt(2)), beta≠1)
#   4: Hyperbolic escape          (|v0| = gamma*sqrt(2mu/r0), gamma>1), v0 ⟂ x0
#
# How to use:
#   • Change SCENARIO to 1, 2, 3, or 4.
#   • For Scenario 3, set BETA (e.g. 0.8 for apocentre start, or 1.2 for pericentre).
#   • For Scenario 4, set GAMMA > 1 (e.g. 1.05).
SCENARIO = 2     # <- choose 1, 2, 3, or 4
ALTITUDE = 1e5   # [m] initial height above Mars surface
BETA =0.98* np.sqrt(2)       # used only in Scenario 3 (elliptic): 0 < BETA < sqrt(2), BETA != 1
GAMMA = 1.05     # used only in Scenario 4 (hyperbolic): GAMMA > 1

# Position on +z axis at the chosen altitude
x0 = np.array([[0.0],
               [0.0],
               [R_MARS + ALTITUDE]])

# Build a unit tangent direction v̂ that is perpendicular to x0 (robust for any x0)
_r = x0.reshape(3,)
_rn = float(np.linalg.norm(_r))
_rhat = _r / _rn
_tmp = np.array([0.0, 1.0, 0.0]) if abs(np.dot(_rhat, np.array([0.0, 1.0, 0.0]))) < 0.9 else np.array([1.0, 0.0, 0.0])
_vhat = np.cross(_tmp, _rhat)
_vhat = _vhat / np.linalg.norm(_vhat)

# Choose |v0| according to the scenario
if SCENARIO == 1:
    v0_mag = 0.0
    _scenario_msg = "Scenario 1: straight-down descent (v0=0)."
elif SCENARIO == 2:
    v0_mag = np.sqrt(mu / _rn)  # v= sqrt(GM/r). Derived by Gravity force = centripetal force
    _scenario_msg = "Scenario 2: circular orbit (|v0|=sqrt(mu/r0))."
elif SCENARIO == 3:
    v_circ = np.sqrt(mu / _rn)
    v0_mag = float(BETA) * v_circ  # just escape is sqrt(2)*v_circ
    _scenario_msg = f"Scenario 3: elliptical orbit (beta={BETA:.3f} of v_circ)."
elif SCENARIO == 4:
    v_esc = np.sqrt(2.0 * mu / _rn)
    v0_mag = float(GAMMA) * v_esc
    _scenario_msg = f"Scenario 4: hyperbolic escape (gamma={GAMMA:.3f} of v_esc)."
else:
    raise ValueError("SCENARIO must be 1, 2, 3, or 4.")

v0_vec = v0_mag * _vhat
v0 = v0_vec.reshape(3, 1)

print(_scenario_msg)
print(f"r0 = {_rn:.3f} m, |v0| = {v0_mag:.6f} m/s")
# -----------------------------------------------------------------

# ---------------- Simulation grid ----------------
t_max = 10000
# Note: dt controls *output sampling* for reference solver; internal step is adaptive.
dt = 1.0
# t=0..t_max-dt (N points). If you want an inclusive grid, use np.linspace and adjust.
t_array = np.arange(0, t_max, dt)

# ---------------- Containers (Euler & Verlet) ----------------
# Euler
x_list = []
v_list = []
# Verlet
x_list_vlt = []
v_list_vlt = []
# Placeholders for any exact analytic solution (unused here)
x_list_exact = []
v_list_exact = []

# ---------------- Euler integration ----------------
# at t = 0*dt
x_list.append(x0)
v_list.append(v0)
x_t = x0
v_t = v0
for t in t_array[1:]:
    # calculate new position and velocity at t + dt 
    x_t_length_square = np.inner(x_t.T, x_t.T)
    x_t_unit = x_t / np.sqrt(x_t_length_square)
    F_G = -((G * M * m) * x_t_unit / x_t_length_square)
    a_t = F_G / m

    # append current state to trajectories
    x_t_pls1 = x_t + dt * v_t
    v_t_pls1 = v_t + dt * a_t
    x_list.append(x_t_pls1)
    v_list.append(v_t_pls1)

    # Update variables
    x_t = np.copy(x_t_pls1)
    v_t = np.copy(v_t_pls1)

# ---------------- Verlet integration ----------------
# at t = 0*dt
x_list_vlt.append(x0)
v_list_vlt.append(v0)

# from t = 1*dt
x_t = x0
v_t = v0
x_t_length_square = np.inner(x_t.T, x_t.T)
x_t_unit = x_t / np.sqrt(x_t_length_square)
F_G = -((G * M * m) * x_t_unit / x_t_length_square)
a_t = F_G / m

x_t_pls1 = x_t + dt * v_t
v_t_pls1 = v_t + dt * a_t

x_list_vlt.append(x_t_pls1)
v_list_vlt.append(v_t_pls1)

# Setup the current state and previous one step state
x_t = x_list_vlt[1]
x_t_mins1 = x_list_vlt[0]

# from t = 2*dt onwards
for t in t_array[2:]:
    # First calculate the what should be the value at t + dt
    x_t_length_square = np.inner(x_t.T, x_t.T)
    x_t_unit = x_t / np.sqrt(x_t_length_square)
    F_G = -((G * M * m) * x_t_unit / x_t_length_square)
    a_t = F_G / m

    x_t_pls1 = 2 * x_t - x_t_mins1 + dt**2 * a_t
    x_list_vlt.append(x_t_pls1)
    v_list_vlt.append((1 / dt) * (x_t_pls1 - x_t))

    # Update variables
    x_t_mins1 = np.copy(x_t)
    x_t = np.copy(x_t_pls1)

# ---------------- Convert lists to arrays ----------------
x_array = np.array(x_list)
v_array = np.array(v_list)

x_array_exact = np.array(x_list_exact)
v_array_exact = np.array(v_list_exact)

x_array_vlt = np.array(x_list_vlt)
v_array_vlt = np.array(v_list_vlt)

# ---- Shape normalisation for plotting (squeeze 3x1 -> 3) ----
# x_array, v_array, x_array_vlt, v_array_vlt may have shape (N, 3, 1) from column-vector appends
if x_array.ndim == 3 and x_array.shape[-1] == 1:
    x_array_plot = x_array.reshape((x_array.shape[0], 3))
else:
    x_array_plot = x_array

if v_array.ndim == 3 and v_array.shape[-1] == 1:
    v_array_plot = v_array.reshape((v_array.shape[0], 3))
else:
    v_array_plot = v_array

if x_array_vlt.ndim == 3 and x_array_vlt.shape[-1] == 1:
    x_array_vlt_plot = x_array_vlt.reshape((x_array_vlt.shape[0], 3))
else:
    x_array_vlt_plot = x_array_vlt

if v_array_vlt.ndim == 3 and v_array_vlt.shape[-1] == 1:
    v_array_vlt_plot = v_array_vlt.reshape((v_array_vlt.shape[0], 3))
else:
    v_array_vlt_plot = v_array_vlt

# ---------------- High-accuracy reference using SciPy ----------------
# Mars fixed at origin. State y = [x, y, z, vx, vy, vz].

def hit_surface(t, y):
    """Stop when we reach the Mars surface: ||r|| = R_MARS (useful for Scenario 1)."""
    r = y[0:3]
    return np.linalg.norm(r) - R_MARS
hit_surface.terminal = True
hit_surface.direction = -1  # crossing from outside (positive) to inside (negative)


def _rhs(t, y):
    r = y[0:3]
    v = y[3:6]
    r2 = float(np.dot(r, r))
    if r2 == 0.0:
        a = np.zeros(3)
    else:
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        a = -G * M * r * inv_r3
    return np.hstack((v, a))

# Flatten initial 3x1 arrays to 1D for the solver
_y0 = np.hstack((x0.reshape(3,), v0.reshape(3,)))
_t0 = float(t_array[0])
_t1 = float(t_array[-1])
# sol_ref = solve_ivp(_rhs, (_t0, _t1), _y0, t_eval=t_array,
#                     method="RK45", rtol=1e-10, atol=1e-13, events=hit_surface)
sol_ref = solve_ivp(_rhs, (_t0, _t1), _y0, t_eval=t_array,
                    method="RK45", rtol=1e-10, atol=1e-13)
if not sol_ref.success:
    raise RuntimeError("solve_ivp failed: " + sol_ref.message)
# Time vector actually returned by the solver (may be shorter if we hit the surface)
t_ref = sol_ref.t
Y_ref = sol_ref.y.T  # shape (N,6)
x_array_ref = Y_ref[:, 0:3]
v_array_ref = Y_ref[:, 3:6]

# ---------------- Plots ----------------
# Altitude (above Mars surface) vs time: Euler vs Verlet vs RK45 reference
alt_eu  = np.linalg.norm(x_array_plot, axis=1) - R_MARS
alt_vlt = np.linalg.norm(x_array_vlt_plot, axis=1) - R_MARS
alt_ref = np.linalg.norm(x_array_ref, axis=1) - R_MARS

plt.figure(10)
plt.clf()
plt.xlabel('time (s)')
plt.ylabel('altitude above Mars surface (m)')
plt.grid(True, alpha=0.3)
plt.plot(t_array, alt_eu,  linestyle='--', linewidth=1.3, alpha=0.9, label='Euler')
plt.plot(t_array, alt_vlt, linestyle='-.', linewidth=1.6, alpha=0.9, label='Verlet')
plt.plot(t_ref,   alt_ref, linestyle='-',  linewidth=2.2, alpha=1.0, label='solve_ivp (RK45)')
plt.legend()
plt.tight_layout()

# 3D trajectory: Euler vs Verlet vs RK45 reference
fig = plt.figure(11)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_array_plot[:, 0],      x_array_plot[:, 1],      x_array_plot[:, 2],      linestyle='--', linewidth=1.0, label='Euler')
ax.plot(x_array_vlt_plot[:, 0],  x_array_vlt_plot[:, 1],  x_array_vlt_plot[:, 2],  linestyle='-.', linewidth=1.2, label='Verlet')
ax.plot(x_array_ref[:, 0],       x_array_ref[:, 1],       x_array_ref[:, 2],       linestyle='-',  linewidth=2.0, label='solve_ivp (RK45)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Position trajectory comparison')
try:
    ax.set_box_aspect((1, 1, 1))  # equal aspect if supported
except Exception:
    pass
ax.legend(loc='best')
plt.tight_layout()
plt.show()
