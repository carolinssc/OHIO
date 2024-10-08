import numpy as np
import mujoco
import torch

def LQR(model, state_ref, N):
    """
    Args:
        model: mujoco model of environment
        state_ref: Reference state for dynamics linearization
        N: Horizon lenght

    Returns:
        K: Optimal control gains
        A: State matrix
        B: Control matrix
    """
    n = model.nv  # state dimension
    m = model.nu  # control dimension
    na = model.na  # number of actuators
    
    Q = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
     
    )  # state cost matrix
   

    R = np.eye(m) * 0.1 # control cost matrix
  
    A = np.zeros((2 * n + na, 2 * n + na)) # Allocate the A and B matrices
    B = np.zeros((2 * n + na, m))

    epsilon = 1e-6
    flg_centered = True
    
    copied_data = mujoco.MjData(model)
    copied_data.qpos = state_ref[: model.nq]
    copied_data.qvel = state_ref[model.nq : model.nq + model.nv]
    mujoco.mj_forward(model, copied_data)

    mujoco.mjd_transitionFD(model, copied_data, epsilon, flg_centered, A, B, None, None) # Compute the A and B matrices

    N += 1
    
    K = np.zeros((N - 1, m, 4))

    P_prev =  np.array(
    [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    for t in range(N - 1, 0, -1): #Riccati recursion
        K[t - 1] = -np.linalg.solve(R + B.T @ P_prev @ B, B.T @ P_prev @ A)
        P = Q + A.T @ P_prev @ (A + B @ K[t - 1].reshape(m, 4))
        np.copyto(P_prev, P)

    del copied_data

    return K, A, B



def get_matrices(model, state_ref):
    """get A and B matrices as a linearization of the dynamics around a reference state

    Args:
        model: mujoco model of environment
        state_ref: Reference state for dynamics linearization

    Returns:
        A: State matrix
        B: Control matrix
    """
    n = model.nv  # state dimension
    m = model.nu  # control dimension
    na = model.na  # number of actuators
    # Allocate the A and B matrices, compute them.
    A = np.zeros((2 * n + na, 2 * n + na))
    B = np.zeros((2 * n + na, m))

    epsilon = 1e-6
    flg_centered = True

    copied_data = mujoco.MjData(model)
    copied_data.qpos = state_ref[: model.nq]
    copied_data.qvel = state_ref[model.nq : model.nq + model.nv]
    mujoco.mj_forward(model, copied_data)

    mujoco.mjd_transitionFD(model, copied_data, epsilon, flg_centered, A, B, None, None)

    del copied_data

    return A, B


def get_A_B_learned(model, s):
    A, B = model(torch.tensor(s, dtype=torch.float32))
    A = A.reshape(4, 4)
    B = B.reshape(4, 2)
    return A.detach(), B.detach()

