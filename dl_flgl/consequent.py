"""Label graph regularization, half-squared fitting losses; FISTA with a valid bound."""
import torch

EPS = 1e-12

def label_graph_laplacian(y):
    """Unnormalized graph Laplacian from positive training-label Pearson edges.

    Negative/undefined correlations and self-edges have zero weight.
    No PSD projection or spectral rescaling is applied.
    """
    if y.shape[0] < 2:
        return y.new_zeros((y.shape[1], y.shape[1]))
    corr = torch.nan_to_num(torch.corrcoef(y.T).reshape(y.shape[1], y.shape[1]),
                            nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = ((corr + corr.T) / 2).clamp_min(0)
    adjacency.fill_diagonal_(0)
    return torch.diag(adjacency.sum(dim=1)) - adjacency

def update_relation(z, y):
    accumulated = z.clamp(0, 1).T @ y
    return accumulated / (accumulated.sum(dim=0, keepdim=True) + 1e-8)

def soft_targets(y, s):
    return (y @ s.T).clamp(0, 1)

def smooth_loss(p, phi, y, target, r, c):
    z = phi @ p
    return (0.5 * c.alpha * (z - y).square().sum()
            + 0.5 * (1 - c.alpha) * (z - target).square().sum()
            + c.beta1 * ((p @ r) * p).sum() + c.beta2 * p.square().sum())

def smooth_gradient(p, phi, y, target, r, c):
    z = phi @ p
    return (phi.T @ (c.alpha * (z - y) + (1 - c.alpha) * (z - target))
            + 2 * c.beta1 * p @ r + 2 * c.beta2 * p)

def soft_threshold(p, threshold):
    return p.sign() * (p.abs() - threshold).clamp_min(0)

def optimize(phi, y, target, r, c, p):
    # ||Phi||_F^2 is an upper bound on ||Phi||_2^2. Avoid a large Gram matrix.
    lip = (phi.square().sum()
           + 2 * c.beta1 * torch.linalg.matrix_norm(r, ord='fro') + 2 * c.beta2).clamp_min(EPS)
    accelerated = p.clone()
    t = 1.0
    history = []
    for iteration in range(c.maxIter):
        updated = soft_threshold(accelerated - smooth_gradient(accelerated, phi, y, target, r, c) / lip, c.beta3 / lip)
        objective = smooth_loss(updated, phi, y, target, r, c) + c.beta3 * updated.abs().sum()
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite objective")
        history.append(float(objective))
        residual = lip * (updated - soft_threshold(updated - smooth_gradient(updated, phi, y, target, r, c) / lip, c.beta3 / lip))
        relative_residual = float(torch.linalg.vector_norm(residual) / max(1.0, float(torch.linalg.vector_norm(updated))))
        if relative_residual <= c.minimumLossMargin:
            p = updated
            break
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        accelerated = updated + (t - 1) / t_next * (updated - p)
        p, t = updated, t_next
    return p, {"iterations": iteration + 1, "objective": history,
               "proximal_residual": relative_residual, "converged": relative_residual <= c.minimumLossMargin,
               "lipschitz_bound": float(lip)}
