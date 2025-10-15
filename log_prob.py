import torch

def log_prob_gaussian_mix(dim, num_peaks, device="cpu", seed=None):
    """
    Returns a function that evaluates the log probability of a Gaussian mixture.
    Supports batched input: x of shape (n_particles, dim)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Randomly sample the means
    mus = [torch.rand(dim, device=device) * 8 - 4 for _ in range(num_peaks)]  # [-4, 4] range

    sigma = 0.6
    const_term = dim * 0.5 * torch.log(torch.tensor(2 * torch.pi * sigma**2, device=device))

    def distribution(x):
        """
        x: (n_particles, dim) or (dim,)
        Returns: (n_particles,) log probabilities
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # make it (1, dim)
        n_particles = x.shape[0]

        log_probs = []
        for mu in mus:
            diff = x - mu  # (n_particles, dim)
            norm_term = -0.5 * (diff ** 2).sum(dim=1) / (sigma ** 2)
            log_probs.append(norm_term - const_term)

        # Stack and apply logsumexp with equal weights
        stacked = torch.stack(log_probs, dim=1) + torch.log(torch.tensor(1.0 / num_peaks, device=device))
        return torch.logsumexp(stacked, dim=1)  # (n_particles,)

    return distribution
