import torch
from botorch.utils.sampling import draw_sobol_samples

class DynamicBalancedSubspace:
    def __init__(self, model, bounds, energy_threshold=0.9, mode="energy", spectral_ratio=2.0, d_max=5, d_min=2):
        """
        Args:
            model: The fitted PairwiseGP or MultiwiseGP.
            bounds: Tensor of [2, dim] for [-1, 1] range.
            energy_threshold: Float (0, 1) to decide d based on eigenvalue sum.
        """
        self.model = model
        self.bounds = bounds
        self.energy_threshold = energy_threshold
        self.spectral_ratio = spectral_ratio
        self.d_max = d_max
        self.d_min = d_min
        self.mode = mode

    def get_active_subspace(self, x_ref, num_gradient_samples=4096, eps=0.15):
        """
        Identifies the dynamic d-dimensional subspace by analyzing 
        surrogate gradients near a reference point.
        """
        dim = self.bounds.shape[-1]
        
        # Local Sampling for Gradient Estimation
        # Define a small neighborhood around x_ref 
        local_lower = torch.clamp(x_ref - eps, self.bounds[0], self.bounds[1])
        local_upper = torch.clamp(x_ref + eps, self.bounds[0], self.bounds[1])
        local_bounds = torch.stack([local_lower.squeeze(0), local_upper.squeeze(0)])
        # Draw D samples using draw_sobol_samples
        x_samples = draw_sobol_samples(bounds=local_bounds, n=num_gradient_samples, q=1).squeeze(1)
        x_samples.requires_grad_(True)
        # Extract Gradients from the GP Posterior Mean
        # Using the posterior mean as the latent utility surrogate
        posterior = self.model.posterior(x_samples)
        mu = posterior.mean.sum()
        grads = torch.autograd.grad(mu, x_samples)[0]
        # Active Subspace Matrix C = E[grad * grad^T]
        C = (grads.unsqueeze(-1) @ grads.unsqueeze(-2)).mean(dim=0)
        # Eigendecomposition and Thresholding
        evals, evecs = torch.linalg.eigh(C)
        evals, evecs = evals.flip(0), evecs.flip(1) # Sort descending
        # print("evals---", evals)
        
        # # Spectral Gap (ratio between eigenvalues)
        if self.mode == "spectral":
            ratios = evals[:-1] / evals[1:]
            # print("ratios---",ratios)
            # Find the first place where the ratio is large (e.g., > 2.0)
            significant_gaps = torch.where(ratios > self.spectral_ratio)[0]
            d = significant_gaps[0].item() + 1 if significant_gaps.numel() > 0 else 1
            # print("significant_gaps---",significant_gaps)
        else:
            # Energy-based
            total_variance = torch.sum(evals)
            if total_variance < 1e-9: # Handle flat landscapes or uninitialized models
                return evecs[:, :2], evals[:2], 2  # was returning only 2 values; callers unpack 3
            cumulative_energy = torch.cumsum(evals, dim=0) / total_variance
            # print("cum-energy---",cumulative_energy)
            d_indices = torch.where(cumulative_energy >= self.energy_threshold)[0]
            d = d_indices[0].item() + 1 if d_indices.numel() > 0 else 1
            # print("d_indices---",d_indices)
        
        d = max(d, self.d_min)
        d = min(d, self.d_max) 
        return evecs[:, :d], evals[:d], d

    def generate_gallery(self, x_best, x_ei, num_samples=9, num_gradient_samples=4096, eps=0.15, scale_tol=0.1, pert_scale=1.5):
        """
        Constructs a d-dimensional manifold bridging x_best and x_ei.
        """
        U_d, evals, d = self.get_active_subspace(x_ei[0:1], num_gradient_samples=num_gradient_samples, eps=eps)
        
        latent_bounds = torch.stack([
            torch.full((d,), -1.0, device=x_best.device, dtype=x_best.dtype),
            torch.full((d,), 1.0, device=x_best.device, dtype=x_best.dtype)
        ])
        sub_coords = draw_sobol_samples(bounds=latent_bounds, n=num_samples, q=1).squeeze(1)
        eval_weights = torch.sqrt(evals / evals.max())
        weighted_coords = sub_coords * eval_weights
        v_bridge = x_ei[0:1] - x_best
        lerp_factors = torch.rand(num_samples, 1, device=x_best.device) 
        
        scale = torch.norm(v_bridge + scale_tol) * pert_scale
        perturbations = (weighted_coords @ U_d.t()) * scale
        result = x_best + (lerp_factors * v_bridge) + perturbations
        return torch.clamp(result, self.bounds[0], self.bounds[1]), d, x_ei
