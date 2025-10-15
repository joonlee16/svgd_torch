import torch
import numpy as np

def generate_positions(N, min_dist, minval, maxval, device="cpu", seed=None):
    torch.manual_seed(seed)
    positions = []
    while len(positions) < N:
        candidate = torch.empty(1, 2).uniform_(minval, maxval).to(device)
        if positions:
            dists = torch.norm(torch.stack(positions) - candidate, dim=1)
            if torch.all(dists >= min_dist):
                positions.append(candidate[0])
        else:
            positions.append(candidate[0])
    return torch.stack(positions)

# -------------------------
# Simulation Parameters
# -------------------------
TIME_ITER = 100
SVGD_ITER = 1000
NUM_PARTICLE = 100
T = 30
DIM_U = 2
DT = 0.05
R_com = 3.0
R_col = 0.3
desired_r = 3
padding = 0.15
LEADER_NUM = 4
N = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = np.random.randint(0, 10000)

# -------------------------
# Initial States
# -------------------------
init_pos = generate_positions(N, R_col, -2.0, 2.0, device=device, seed=seed)
init_vel = torch.ones((N, 2), device=device)
state = torch.cat([init_pos, init_vel], dim=1)

# -------------------------
# Goal States
# -------------------------
x_goal = generate_positions(N, R_col, 9.0, 12.0, device=device, seed=seed)

# -------------------------
# Obstacles
# -------------------------
obstacles = torch.tensor([
    [2.0, 2.0],
    [5.0, 4.0],
    [6.0, 6.3],
    [9.0, 5.0],
    [8.3, 8.5],
    [4.5, 2.5],
    [3.5, 8.0],
], device=device)

radii = torch.tensor([0.5, 0.7, 0.85, 1.5, 0.9, 0.4, 1.4], device=device)

# -------------------------
# Cost Weights
# -------------------------
Q, R, S, P = 50.0, 10000.0, 1500.0*torch.ones(2, device=device), 4e3
