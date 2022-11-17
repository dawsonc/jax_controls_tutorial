import jax.numpy as jnp
import numpy as np
import scipy

MU = 3.986e14  # Earth's gravitational parameter (m^3 / s^2)
A_GEO = 42164e3  # GEO semi-major axis (m)
A_LEO = 353e3  # GEO semi-major axis (m)
M_CHASER = 500  # chaser satellite mass
N = jnp.sqrt(MU / A_LEO ** 3)  # mean-motion parameter

num_action_dims = 3
num_state_dims = 6

# State indices
X = 0
Y = 1
Z = 2
VX = 3
VY = 4
VZ = 5

# Control indices
TX = 0
TY = 1
TZ = 2

lower_bounds = -10 * jnp.ones((num_state_dims,))
upper_bounds = -lower_bounds

d_min = 2.0
buffer = 0.5


def safe_label_fn(x):
    safe_mask = jnp.linalg.norm(x[:Z+1]) >= d_min + buffer
    return safe_mask


def dang_label_fn(x):
    dang_mask = jnp.linalg.norm(x[:Z+1]) <= d_min - buffer
    return dang_mask


def dynamics_fn(current_state, control_input):
    # Extract states and controls
    px, _, pz, vx, vy, vz = current_state
    ux, uy, uz = control_input

    # Positions just integrate velocities
    dxdt = jnp.zeros_like(current_state)
    dxdt = dxdt.at[0].add(vx)
    dxdt = dxdt.at[1].add(vy)
    dxdt = dxdt.at[2].add(vz)

    # Velocities follow CHW dynamics (See Jewison & Erwin CDC 2016)
    # Note that the dynamics in z are decoupled from those in xy,
    # so we can also consider just the xy projection if we need to.
    dx2_dt2 = 2 * N * vy + 3 * N ** 2 * px + ux / M_CHASER
    dy2_dt2 = -2 * N * vx + uy / M_CHASER
    dz2_dt2 = -(N ** 2) * pz + uz / M_CHASER
    dxdt = dxdt.at[3].add(dx2_dt2)
    dxdt = dxdt.at[4].add(dy2_dt2)
    dxdt = dxdt.at[5].add(dz2_dt2)

    return dxdt

def get_lqr_gains():
    # Define continuous-time A and B matrices
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 0] = 3 * N ** 2
    A[3, 4] = 2 * N
    A[4, 3] = -2 * N
    A[5, 2] = -(N ** 2)

    B = np.zeros((6, 3))
    B[3, 0] = 1 / M_CHASER
    B[4, 1] = 1 / M_CHASER
    B[5, 2] = 1 / M_CHASER
    
    # Define cost matrices
    Q = 1 * np.eye(num_state_dims)
    R = 2 * np.eye(num_action_dims)
    
    # Solve Ricatti equation
    X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.array(scipy.linalg.inv(R) @ (B.T @ X))
    
    return jnp.array(K)

K = get_lqr_gains()

def simple_controller(x):
    return -K.dot(x)
