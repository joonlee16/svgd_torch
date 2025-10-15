import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from config import *

# -------------------------
# SVGD Animation (PyTorch version)
# -------------------------
def animate_svgd(history, log_prob_fn, record_every=10, xlim=(-5,5), ylim=(-5,5)):
    """
    history: (n_records, n_particles, d) torch tensor
    log_prob_fn: function that evaluates log probability on batched input (torch)
    """
    history_np = np.squeeze(history.cpu().numpy(), axis=2)  # convert to numpy for plotting
    n_records, n_particles, d = history_np.shape

    fig, ax = plt.subplots(figsize=(5, 5))
    scat = ax.scatter([], [], s=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("SVGD Particle Evolution")

    # Density contour
    xx, yy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    with torch.no_grad():
        logp_vals = log_prob_fn(torch.tensor(grid, dtype=torch.float32)).cpu().numpy()
    density = np.exp(logp_vals).reshape(xx.shape)
    ax.contourf(xx, yy, density, levels=30, cmap="Blues", alpha=0.4)

    def update(frame):
        scat.set_offsets(history_np[frame])
        ax.set_title(f"SVGD Step {frame * record_every}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=n_records, interval=100, blit=True)
    plt.show()


# -------------------------
# MPC Animation
# -------------------------
def animate_mpc(all_samples, best_trajs, goals, obstacles=[], radii=[]):
    """
    all_samples: list of length TIME_ITER, each (S, N, T, 4)
    best_trajs:  list of length TIME_ITER, each (N, T, 4)
    goals: (N, 2)
    obstacles: (M,2)
    radii: (M,)
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 13)

    # Plot goals
    ax.scatter(goals[:, 0], goals[:, 1], c='red', s=60, marker='x', label='Goals')

    # Obstacles
    obstacle_patches = []
    for (x, y), r in zip(obstacles, radii):
        circle = Circle((x, y), r, color='black', alpha=0.3)
        ax.add_patch(circle)
        obstacle_patches.append(circle)

    S, N, T, _ = all_samples[0].shape
    colors = plt.cm.tab10(np.arange(N) % 10)

    # Best trajectory lines
    traj_lines = [ax.plot([], [], '-', lw=2, color=colors[i])[0] for i in range(N)]
    robot_dots = [ax.scatter([], [], c=[colors[i]], s=40, zorder=3) for i in range(N)]

    # Faint sampled trajectories
    sample_lines = [[ax.plot([], [], '-', lw=1, alpha=0.1, color=colors[i])[0]
                     for _ in range(S)] for i in range(N)]

    def update(frame):
        samples = all_samples[frame]  # (S, N, T, 4)
        best = best_trajs[frame]      # (N, T, 4)

        for i in range(N):
            for s in range(S):
                # Uncomment to show faint trajectories
                # traj_lines[i].set_data(samples[s,i,:,0], samples[s,i,:,1])
                pass
            traj_lines[i].set_data(best[i,:,0], best[i,:,1])
            robot_dots[i].set_offsets(best[i,0,:2])

        ax.set_title(f"MPC Step {frame}")
        return sum(sample_lines, []) + traj_lines + robot_dots + obstacle_patches

    ani = animation.FuncAnimation(fig, update, frames=len(all_samples),
                                  interval=200, blit=True)
    plt.legend()
    plt.show()
