"""Model training loops, autograd physics constraint penalties, and evaluation metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm


def train(model, X_all, S_all, T_all, Y_all, tr, epochs=60, bs=2048, lam_mono=0.0, lr_phys=1e-2, lr_nn=2e-3, seed=0, show_progress=False):
    """Trains PyTorch models with separate optimizer learning rates and autograd monotonic penalties."""
    torch.manual_seed(seed)
    nn_p = [p for n, p in model.named_parameters() if n.startswith('net.')]
    ph_p = [p for n, p in model.named_parameters() if not n.startswith('net.')]
    groups = ([{'params': ph_p, 'lr': lr_phys}] if ph_p else []) + \
             ([{'params': nn_p, 'lr': lr_nn}] if (nn_p and getattr(model, 'use_nn', True)) else [])
    opt = torch.optim.Adam(groups)
    tr_t = torch.as_tensor(tr)
    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training Epochs", leave=False)
    for ep in epoch_iter:
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = X_all[b].clone().requires_grad_(lam_mono > 0)
            pred = model(xb, S_all[b], T_all[b])
            loss = F.mse_loss(pred, Y_all[b])
            if lam_mono > 0:
                g_age = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0][:, 0]
                loss = loss + lam_mono * torch.relu(g_age).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if show_progress and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(loss=f"{loss.item():.5f}")
    return model


@torch.no_grad()
def predict(model, X_all, S_all, T_all, idx):
    """Generates predictions for a given index set with disabled PyTorch autograd."""
    model.eval()
    idx = torch.as_tensor(idx)
    return model(X_all[idx], S_all[idx], T_all[idx]).numpy()


def violation_rate(model, X_all, S_all, T_all, idx, n=5000):
    """Calculates the proportion of test samples violating aging monotonicity (dPR/dage > 0)."""
    model.eval()
    idx = torch.as_tensor(idx)[torch.randperm(len(idx))[:n]]
    x = X_all[idx].clone().requires_grad_(True)
    with torch.enable_grad():
        g = torch.autograd.grad(model(x, S_all[idx], T_all[idx]).sum(), x)[0][:, 0]
    return (g > 0).float().mean().item()


def score(y_true, y_pred):
    """Computes R2 and MAE score metrics."""
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred))


def read_params(model, techs):
    """Extracts interpretable physical parameters from a trained Hybrid/Physics-only model."""
    return dict(degradation_pct_per_yr=dict(zip(techs, (100 * 0.03 * torch.sigmoid(model.d_raw)).detach().numpy().round(2))),
                gamma_scale_vs_datasheet=round(F.softplus(model.g_raw).item(), 3),
                cell_temp_beta=round((1.5 * torch.sigmoid(model.beta_raw)).item(), 3),
                cell_temp_k_degC_per_Wm2=round((0.2 * torch.sigmoid(model.k_raw)).item(), 4),
                soiling_rates=(0.2 * torch.sigmoid(model.a_raw)).detach().numpy().round(4).tolist())