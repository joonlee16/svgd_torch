import torch
import torch.nn.functional as F
import numpy as np
import time
from svgd import *  
from helper_functions import animate_mpc  
from config import *  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# ----------------------------
# Double Integrator Rollout
# ----------------------------
def double_integrator_rollout(x0, u, dt=DT):
    """
    x0: (N, 4)
    u: (N, T, 2)
    returns: (N, T, 4)
    """
    N, T, _ = u.shape
    traj = []
    x = x0.clone()
    for t in range(T):
        ax, ay = u[:, t, 0], u[:, t, 1]
        px, py, vx, vy = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        px_next = px + vx * dt
        py_next = py + vy * dt
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt

        x = torch.stack([px_next, py_next, vx_next, vy_next], dim=1)
        traj.append(x)
    traj = torch.stack(traj, dim=1)  # (N, T, 4)
    return traj


# ----------------------------
# Inter-agent Collision Penalty
# ----------------------------
def inter_collision_penalty(pos_t):
    diff = pos_t[:, None, :] - pos_t[None, :, :]
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-3)
    dist = dist + torch.eye(N, device=device) * 1e6
    penal = torch.where(dist < 2 * R_col,
                         R * (2 * R_col - dist),
                         2 * R_col - dist)
    return penal


# ----------------------------
# Cost Function
# ----------------------------
def make_cost(Q, R, S, desired_terminal):
    def cost_fn(u, x0):
        controls = u.view(N, -1, 2)  # (N, T, 2)
        traj = double_integrator_rollout(x0, controls)  # (N, T, 4)

        # velocity penalty
        vel_penal = Q * torch.sum(traj[:, :, 2:]**2)

        # obstacle penalty
        reshaped_traj = traj[:, :, :2].reshape(-1, 2)
        diffs = reshaped_traj[:, None, :] - obstacles[None, :, :]
        dists = torch.norm(diffs, dim=-1)
        obs_col_penal = torch.where(dists < radii+padding,
                                    R * (radii+padding - dists),
                                    radii+padding - dists)
        obstacle_penal = torch.sum(obs_col_penal)

        # inter-agent collisions
        inter_col_penal = torch.sum(torch.stack([inter_collision_penalty(traj[:, t, :2]) for t in range(T)]))

        # terminal cost
        terminal_cost = torch.sum(S * (traj[:, -1, :2] - desired_terminal)**2)

        return -(vel_penal + terminal_cost + obstacle_penal + inter_col_penal)
    return cost_fn


# ----------------------------
# Main Simulation
# ----------------------------
if __name__ == "__main__":
    all_samples = []
    best_trajs = []
    time_history = []

    state = torch.zeros((N,4), device=device)  # initialize robot states
    cost_function = make_cost(Q, R, S, x_goal.to(device))

    for t in range(TIME_ITER):
        init_time = time.time()

        # Sample random control particles
        theta = torch.randn(NUM_PARTICLE, N*T*DIM_U, device=device) * 5.0

        # Run SVGD
        for _ in range(SVGD_ITER):
            theta = svgd_step(theta, lambda u: cost_function(u, state), lr=0.05)

        # Roll out all trajectories
        sample_trajs = torch.stack([double_integrator_rollout(state, u.view(N, T, DIM_U)) for u in theta])
        all_samples.append(sample_trajs.cpu().detach().numpy())

        # Pick best trajectory
        costs = torch.tensor([cost_function(u, state).item() for u in theta])
        best_idx = torch.argmax(costs)
        best_traj = sample_trajs[best_idx]
        best_trajs.append(best_traj.cpu().detach().numpy())

        # Update state to next step
        state = best_traj[:, 1, :].clone()  # move to next time

        time_history.append(time.time() - init_time)

    animate_mpc(all_samples, best_trajs, x_goal.cpu().numpy())

    print("max comp time", max(time_history[1:]))
    print("average comp time", sum(time_history[1:])/(TIME_ITER-1))
    print("min comp time", min(time_history[1:]))
