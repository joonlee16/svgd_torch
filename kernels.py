import torch

# -------------------------
# Helper Functions
# -------------------------
def sq_dists(theta):
    """
    Compute squared pairwise distances
    theta: (n, d)
    returns: (n, n)
    """
    x_norm = (theta ** 2).sum(dim=1, keepdim=True)  # (n,1)
    sq = x_norm + x_norm.T - 2.0 * (theta @ theta.T)
    return torch.clamp(sq, min=0.0)

def median_heur(theta, eps=1e-6):
    """
    Median heuristic for bandwidth selection
    theta: (n, d)
    """
    n = theta.shape[0]
    sq = sq_dists(theta)
    iu = torch.triu_indices(n, n, offset=1)
    upper_vals = sq[iu[0], iu[1]]
    return upper_vals.median() / (torch.log(torch.tensor(n, dtype=torch.float32)) + eps)

# -------------------------
# RBF Kernel Implementation
# -------------------------
def rbf_kernel(theta):
    """
    Compute RBF kernel matrix and its gradient
    theta: (n, d)
    Returns:
        K: (n, n)
        grad_K: (n, n, d)
    """
    theta_sq = sq_dists(theta)
    h = median_heur(theta)
    K = torch.exp(-theta_sq / h)

    # Compute gradient
    theta_i = theta[:, None, :]  # (n,1,d)
    theta_j = theta[None, :, :]  # (1,n,d)
    diffs = theta_i - theta_j    # (n,n,d)
    coef = -2.0 / h
    grad_K = coef * diffs * K[:, :, None]  # (n,n,d)

    return K, grad_K
