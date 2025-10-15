import torch
import numpy as np
from log_prob import *      # should return a torch-compatible log_prob function
from kernels import *       # rbf_kernel rewritten in torch
from helper_functions import animate_svgd
from config import *

# -------------------------
# SVGD Step
# -------------------------
def svgd_step(theta, log_prob_fn, lr):
    """
    theta: (n_particles, N, T, dim_u) or (n_particles, dim_u)
    log_prob_fn: function that takes theta and returns (n_particles,) log-probs
    """
    n = theta.shape[0]

    # Flatten particles if multi-dimensional
    theta_flat = theta.view(n, -1)
    theta_flat.requires_grad_(True)

    # Compute gradients
    log_p = log_prob_fn(theta_flat)      # (n,)
    grads = torch.autograd.grad(log_p.sum(), theta_flat)[0]  # (n, dim_flat)

    # Compute kernel and its gradient
    K, grad_K = rbf_kernel(theta_flat)  # torch version
    phi_hat = (K @ grads + grad_K.sum(dim=1)) / n

    with torch.no_grad():
        new_theta_flat = theta_flat + lr * phi_hat

    return new_theta_flat.view_as(theta)

# -------------------------
# Run SVGD with history
# -------------------------
def run_svgd_with_history(theta_init, log_prob_fn, n_steps=1000, lr=1e-3, record_every=10):
    n_particles = theta_init.shape[0]
    dim_flat = int(np.prod(theta_init.shape[1:]))
    n_records = n_steps // record_every + 1

    history = torch.zeros((n_records, n_particles, *theta_init.shape[1:]), device=theta_init.device)
    theta = theta_init.clone()
    record_idx = 0
    history[record_idx] = theta
    record_idx += 1

    for i in range(1, n_steps + 1):
        theta = svgd_step(theta, log_prob_fn, lr)
        if i % record_every == 0:
            history[record_idx] = theta
            record_idx += 1

    return theta, history

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_particles = NUM_PARTICLE
    dim_u = DIM_U

    # Initialize particles
    x0 = torch.randn(n_particles, 1, dim_u, device=device) * 3.0

    # Log probability (supports batched inputs)
    log_prob_fn = log_prob_gaussian_mix(dim=dim_u, num_peaks=4, device=device)

    # Run SVGD
    new_theta, history = run_svgd_with_history(x0, log_prob_fn, n_steps=SVGD_ITER, lr=0.05)

    # Animate results
    animate_svgd(history, log_prob_fn)
