"""Standalone dual-loop model. Inputs and outputs use sample-by-label layout."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from .antecedent import SSFCM
from .consequent import label_graph_laplacian, soft_targets, update_relation, optimize

@dataclass
class Config:
    B: int = 10
    B_prime: int = 2
    C: int = 3
    M: Optional[int] = None
    M_fraction: float = 0.3
    m: float = 2.0
    h: float = 1.0
    alpha: float = 0.6
    beta1: float = 0.01
    beta2: float = 0.01
    beta3: float = 0.001
    num_epochs: int = 5
    maxIter: int = 500
    minimumLossMargin: float = 1e-5
    fcm_max_iter: int = 100
    fcm_tol: float = 1e-6
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        for name in ('B', 'B_prime', 'C', 'num_epochs', 'maxIter', 'fcm_max_iter'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.M is not None and (type(self.M) is not int or self.M <= 0):
            raise ValueError("M must be a positive integer or null")
        numbers = [self.m, self.h, self.alpha, self.beta1, self.beta2, self.beta3, self.minimumLossMargin, self.fcm_tol, self.M_fraction]
        if not np.isfinite(numbers).all():
            raise ValueError("Parameters must be finite")
        if not self.B_prime < self.B or self.m <= 1 or self.h <= 0:
            raise ValueError("Require B_prime < B, m > 1, h > 0")
        if not 0 <= self.alpha <= 1 or min(self.beta1, self.beta2, self.beta3) < 0:
            raise ValueError("Require alpha in [0,1] and nonnegative regularization")
        if not 0 < self.M_fraction <= 1 or min(self.minimumLossMargin, self.fcm_tol) <= 0:
            raise ValueError("Invalid subspace fraction or tolerance")

class DLFLGL:
    def __init__(self, config=None):
        self.config = Config() if config is None else config

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or min(x.shape + y.shape) == 0:
            raise ValueError("Expected nonempty X[N,D] and Y[N,L]")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Inputs must be finite and Y must contain 0/1")
        self.antecedent_ = SSFCM(self.config).fit(x, y)
        device = torch.device(self.config.device)
        self.P_ = torch.zeros((len(self.antecedent_.rules_) * (x.shape[1] + 1), y.shape[1]), dtype=torch.float64, device=device)
        phi = torch.as_tensor(self.antecedent_.transform(x), dtype=torch.float64, device=device)
        labels = torch.as_tensor(y, dtype=torch.float64, device=device)
        self.L_ = label_graph_laplacian(labels)
        self.S_ = torch.full((y.shape[1], y.shape[1]), 1 / y.shape[1], dtype=torch.float64, device=device)
        self.history_ = []
        for epoch in range(self.config.num_epochs):
            target = soft_targets(labels, self.S_)
            self.P_, history = optimize(phi, labels, target, self.L_, self.config, self.P_)
            relation = update_relation(phi @ self.P_, labels)
            history['relation_change'] = float(torch.linalg.matrix_norm(relation - self.S_))
            history['epoch'] = epoch + 1
            self.history_.append(history)
            self.S_ = relation
        return self

    def decision_function(self, x):
        if not hasattr(self, 'P_'):
            raise RuntimeError("Call fit before prediction")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.antecedent_.n_features_ or not np.isfinite(x).all():
            raise ValueError("Invalid prediction features")
        phi = torch.as_tensor(self.antecedent_.transform(x), dtype=self.P_.dtype, device=self.P_.device)
        return (phi @ self.P_).detach().cpu().numpy()

    def predict_membership(self, x):
        return np.clip(self.decision_function(x), 0, 1)

    def predict(self, x, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0,1]")
        return (self.predict_membership(x) >= threshold).astype(int)
