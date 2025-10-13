# uncomment the next line if running in a notebook
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
m = 1
k = 1
x = 0
v = 1

# simulation time, timestep and time
t_max = 1000
dt = 2.5 # For example, if you take m = k= v0 = 1 
            # and calculate the trajectory from t = 0
            # to t = 1000, you should find that the Verlet integrator 
            # is stable for ∆t = 1 but not for ∆t = 2.
t_array = np.arange(0, t_max, dt)

# initialise empty lists to record trajectories
x_list = []
v_list = []

# initialise empty lists to record trajectories (for Verlet)
x_list_vlt = []
v_list_vlt = []

# Get general solution
x0 = x
v0 = v
omega = np.sqrt(k/m)
x_list_exact = []
v_list_exact = []







# Euler integration
for t in t_array:

    # append current state to trajectories
    x_list.append(x)
    v_list.append(v)

    # Append exact solution
    x_list_exact.append( x0*np.cos(omega*t) + (v0/omega)*np.sin(omega*t) )
    v_list_exact.append( -x0*omega*np.sin(omega*t) + v0*np.cos(omega*t)  )

    # calculate new position and velocity
    a = -k * x / m
    x = x + dt * v
    v = v + dt * a






# Verlet integration
# at t = 0*dt
x_list_vlt.append(x0)
v_list_vlt.append(v0)

# from t =1*dt 
x_list_vlt.append(x0 + dt*v0)
v_list_vlt.append(v0 - dt*k*x0/m)

# Setup the current state and previous one step state
x_t = x_list_vlt[1]
x_t_mins1 = x_list_vlt[0]
# v_t = x_list_vlt[1]
# v_t_mins1 = v_list_vlt[0]

# from t =2*dt onwards
for t in t_array[2:]:
    # First calculate the what should be the value at t + dt
    x_t_pls1 = 2*x_t - x_t_mins1 + dt**2 * (-k*x_t/m)
    x_list_vlt.append(x_t_pls1)
    v_list_vlt.append( (1/dt) * (x_t_pls1 - x_t) )

    # Update variables
    x_t_mins1 = np.copy(x_t)
    x_t = np.copy(x_t_pls1)






# convert trajectory lists into arrays, so they can be sliced (useful for Assignment 2)
x_array = np.array(x_list)
v_array = np.array(v_list)

x_array_exact = np.array(x_list_exact)
v_array_exact = np.array(v_list_exact)

x_array_vlt = np.array(x_list_vlt)
v_array_vlt = np.array(v_list_vlt)

# plot the position-time graph
plt.figure(1)
plt.clf()
plt.xlabel('time (s)')
plt.grid()
# plt.plot(t_array, x_array, label='x Euler (m)')
plt.plot(t_array, x_array_vlt, label='x Verlet (m)')
plt.plot(t_array, x_array_exact, label='x Exact (m)')
plt.legend()
plt.show()


plt.figure(2)
plt.clf()
plt.xlabel('time (s)')
plt.grid()
# plt.plot(t_array, v_array, label='x Euler (m)')
plt.plot(t_array, v_array_vlt, label='x Verlet (m)')
plt.plot(t_array, v_array_exact, label='v exact (m/s)')
plt.legend()
plt.show()
