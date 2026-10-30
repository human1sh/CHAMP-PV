"""PyTorch module definitions for Hybrid Physics Model and baseline architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nn_features(x):
    """Transforms raw 10-feature tensor into scaled representation for neural network residual."""
    age, G, Tm, Tx, wind, pm, hum, dsr, dsc, gam0 = x.unbind(1)
    return torch.stack([
        age,
        G / 100,
        Tm,
        Tx - Tm,
        wind,
        pm / 50,
        hum / 100,
        torch.log1p(dsr),
        torch.log1p(dsc),
        gam0 * 100
    ], dim=1)


class Norm(nn.Module):
    """Normalizes network features using tracked mean and standard deviation from training data."""
    def __init__(self, Xtr):
        super().__init__()
        f = nn_features(Xtr)
        self.register_buffer('mu', f.mean(0))
        self.register_buffer('sd', f.std(0) + 1e-6)

    def forward(self, x):
        return (nn_features(x) - self.mu) / self.sd


class HybridPV(nn.Module):
    """Physics-Constrained Hybrid Neural Network (PCHNN) for PV Performance Ratio."""
    def __init__(self, Xtr, n_sys, n_tech, use_nn=True):
        super().__init__()
        self.use_nn = use_nn
        self.norm = Norm(Xtr)

        # Learnable physical parameters
        self.eta_raw = nn.Parameter(torch.full((n_sys,), 0.2))    # Baseline efficiency (~0.85)
        self.d_raw = nn.Parameter(torch.full((n_tech,), -1.6))    # Technology degradation (~0.5%/yr)
        self.g_raw = nn.Parameter(torch.tensor(0.5413))           # Temperature coefficient scale
        self.beta_raw = nn.Parameter(torch.tensor(0.0))           # Temperature blending factor
        self.k_raw = nn.Parameter(torch.tensor(-1.386))           # Irradiance heating coefficient
        self.a_raw = nn.Parameter(torch.full((2,), -2.944))      # Soiling parameters
        self.ll_raw = nn.Parameter(torch.tensor(-2.197))          # Low-light loss factor

        # Residual Neural Network correction MLP (10 -> 32 -> 32 -> 1)
        self.net = nn.Sequential(nn.Linear(10, 32),
                                nn.SiLU(),
                                nn.Linear(32, 32),
                                nn.SiLU(),
                                nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def physics(self, x, sid, tid):
        """Computes explicit parametric physics formula for Performance Ratio."""
        age, G, Tm, Tx, wind, pm, hum, dsr, dsc, gam0 = x.unbind(1)
        eta = 0.3 + torch.sigmoid(self.eta_raw[sid])
        dtec = 0.03 * torch.sigmoid(self.d_raw[tid])
        beta = 1.5 * torch.sigmoid(self.beta_raw)
        k = 0.2 * torch.sigmoid(self.k_raw)
        Tc = Tm + beta * (Tx - Tm) + k * G
        temp = 1 + F.softplus(self.g_raw) * gam0 * (Tc - 25.0)
        a = 0.2 * torch.sigmoid(self.a_raw)
        soil = a[0] * (pm / 50) * (torch.clamp(dsr, max=30.0) / 30) + a[1] * (torch.clamp(dsc, max=365.0) / 365)
        lowl = 1 - 0.5 * torch.sigmoid(self.ll_raw) * torch.exp(-G / 150)
        return eta * temp * (1 - dtec * age) * (1 - soil) * lowl

    def forward(self, x, sid, tid):
        """Combines physical baseline with bounded residual neural network correction."""
        out = self.physics(x, sid, tid)
        if self.use_nn:
            out = out * (1 + 0.25 * torch.tanh(self.net(self.norm(x)).squeeze(1)))
        return out


class MLPBaseline(nn.Module):
    """Purely data-driven unconstrained Multilayer Perceptron baseline."""
    def __init__(self, Xtr, y_mu=0.73, y_sd=0.08):
        super().__init__()
        self.norm = Norm(Xtr)
        self.y_mu = y_mu
        self.y_sd = y_sd
        self.net = nn.Sequential(nn.Linear(10, 64),
                                nn.SiLU(),
                                nn.Linear(64, 64),
                                nn.SiLU(),
                                nn.Linear(64, 1))

    def forward(self, x, sid=None, tid=None):
        return self.y_mu + self.y_sd * self.net(self.norm(x)).squeeze(1)