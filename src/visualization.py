"""Publication-quality figure generation utilities for model evaluation and diagnostics."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from .training import predict, read_params
from .data import HGB_FEATS


def set_plot_style():
    """Sets publication-quality Seaborn and Matplotlib figure aesthetic parameters."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12,
                        'axes.labelsize': 14,
                        'axes.titlesize': 14,
                        'xtick.labelsize': 12,
                        'ytick.labelsize': 12,
                        'legend.fontsize': 11,
                        'figure.titlesize': 16})

def plot_age_extrapolation(df, store, X_all, S_all, T_all, get_split_fn, output_dir='output'):
    """Plots PR model extrapolation trends across system age (0 to 18 years)."""
    set_plot_style()
    split = 'age  : train <8 y, test >=10 y'
    tr, _ = get_split_fn(df, split)
    allidx = np.where(df.is_clean & df.sid.isin(df.sid.values[tr]))[0]
    bins = np.arange(0, 19, 1.0)
    ab = pd.cut(df.system_age_years.values[allidx], bins)
    curves = {'Actual Data': df.pr.values[allidx]}
    curves['HistGradientBoosting'] = store[(split, 'HGB')][0].predict(df.loc[allidx, HGB_FEATS])
    curves['MLP (No Physics)'] = predict(store[(split, 'MLP (no physics)')][0], X_all, S_all, T_all, allidx)
    curves['Physics Only'] = predict(store[(split, 'Physics only')][0], X_all, S_all, T_all, allidx)
    curves['Hybrid (Monotonic Pen.)'] = predict(store[(split, 'Hybrid, lam_mono=1')][0], X_all, S_all, T_all, allidx)
    plt.figure(figsize=(11, 6))
    colors = {'Actual Data': 'black',
            'HistGradientBoosting': '#d95f02',
            'MLP (No Physics)': '#e7298a',
            'Physics Only': '#7570b3',
            'Hybrid (Monotonic Pen.)': '#1b9e77'}
    for nm, v in curves.items():
        s = pd.Series(v).groupby(ab, observed=False).mean()
        is_act = (nm == 'Actual Data')
        plt.plot(bins[:-1] + 0.5, s.values,
                '-o' if is_act else '--' if 'MLP' in nm or 'HGB' in nm else '-',
                lw=3 if is_act or 'Hybrid' in nm else 2,
                color=colors[nm], label=nm)
    plt.axvspan(8, 10, color='gray', alpha=0.2, label='Excluded Gap (8-10y)')
    plt.axvline(8, ls=':', c='gray')
    plt.axvline(10, ls=':', c='gray')
    plt.title('Performance Ratio Extrapolation on Unseen Older Panels')
    plt.xlabel('System Age (Years) [Train < 8y | Test ≥ 10y]')
    plt.ylabel('Mean Performance Ratio (PR)')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vis_1_age_extrapolation.png'), dpi=300)
    plt.close()

def plot_monotonicity_violations(results_df, output_dir='output'):
    """Bar chart comparing unphysical aging monotonicity violation rates across models."""
    set_plot_style()
    df_plot = results_df.dropna(subset=['mono_violation']).copy()
    df_plot['mono_violation_pct'] = df_plot['mono_violation'] * 100
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_plot,
                    x='model',
                    y='mono_violation_pct',
                    hue='split',
                    palette='Set2')
    plt.title('Physical Monotonicity Violation Rate (dPR/dage > 0)')
    plt.ylabel('Violation Rate (%) [Lower is Better]')
    plt.xlabel('Model Architecture')
    plt.xticks(rotation=15)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, xytext=(0, 3),
                        textcoords='offset points')
    plt.legend(title='Evaluation Split', frameon=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vis_2_monotonicity_violations.png'), dpi=300)
    plt.close()

def plot_residual_bounded_correction(X_all, store, output_dir='output'):
    """Distribution plot showing bounded output residual correction factor of Hybrid model."""
    set_plot_style()
    split = 'time : train <2023, test >=2023'
    model = store[(split, 'Hybrid, lam_mono=1')][0]
    model.eval()
    with torch.no_grad():
        norm_x = model.norm(X_all)
        nn_out = model.net(norm_x).squeeze(1)
        bounded_correction = 0.25 * torch.tanh(nn_out).numpy()
    plt.figure(figsize=(9, 5))
    sns.histplot(bounded_correction, kde=True, color='#1b9e77', bins=50)
    plt.axvline(0, color='red', linestyle='--', label='Pure Physics Baseline (0%)')
    plt.axvline(-0.25, color='black', linestyle=':', label='Bounds (±25%)')
    plt.axvline(0.25, color='black', linestyle=':')
    plt.title('Distribution of Hybrid Neural Network Residual Corrections')
    plt.xlabel(r'Bounded Residual Correction Factor $0.25 \times \tanh(\text{NN}(x))$')
    plt.ylabel('Frequency (Sample Days)')
    plt.legend(frameon=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vis_3_residual_distribution.png'), dpi=300)
    plt.close()

def plot_anomaly_detection_curves(df, store, X_all, S_all, T_all, get_split_fn, output_dir='output'):
    """Generates ROC and Precision-Recall curves for fault/outage detection from residuals."""
    set_plot_style()
    split = 'time : train <2023, test >=2023'
    tr, _ = get_split_fn(df, split)
    te_all = np.where((df.date >= '2023-01-01') & df.sid.isin(df.sid.values[tr]))[0]
    y_true = df.outage_flag.values[te_all]
    y_obs = df.pr.values[te_all]
    p_hybrid = predict(store[(split, 'Hybrid, lam_mono=1')][0], X_all, S_all, T_all, te_all)
    score_hybrid = (p_hybrid - y_obs) / np.maximum(p_hybrid, 1e-5)
    p_hgb = store[(split, 'HGB')][0].predict(df.loc[te_all, HGB_FEATS])
    score_hgb = (p_hgb - y_obs) / np.maximum(p_hgb, 1e-5)
    p_mlp = predict(store[(split, 'MLP (no physics)')][0], X_all, S_all, T_all, te_all)
    score_mlp = (p_mlp - y_obs) / np.maximum(p_mlp, 1e-5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scores = {'Hybrid (lam_mono=1)': (score_hybrid, '#1b9e77'),
              'HistGradientBoosting': (score_hgb, '#d95f02'),
              'MLP (No Physics)': (score_mlp, '#e7298a')}
    for name, (score, color) in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        precision, recall, _ = precision_recall_curve(y_true, score)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2, label=f'{name} (PR-AUC = {pr_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title('ROC Curve: Inverter/String Outage Detection')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right', frameon=True)
    axes[1].set_title('Precision-Recall Curve: Outage Detection')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left', frameon=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vis_4_anomaly_detection_roc_pr.png'), dpi=300)
    plt.close()

def plot_learned_degradation_rates(splits, store, techs, output_dir='output'):
    """Bar chart comparing learned degradation rates across datasets and panel technologies."""
    set_plot_style()
    deg_data = []
    for split_name in splits:
        model = store[(split_name, 'Physics only')][0]
        params = read_params(model, techs)
        for tech, rate in params['degradation_pct_per_yr'].items():
            deg_data.append({'Split': 'Time Split' if 'time' in split_name else 'Age Split',
                            'Technology': tech,
                            'Degradation (%/yr)': rate})
    df_deg = pd.DataFrame(deg_data)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_deg, x='Technology', y='Degradation (%/yr)', hue='Split', palette='Set1')
    plt.title('Learned Effective Degradation Rates by Panel Technology')
    plt.ylabel('Degradation Rate (% / Year)')
    plt.xlabel('Panel Technology')
    plt.axhspan(0.3, 0.7, color='gray', alpha=0.15, label='Literature Expected Range (0.3 - 0.7%)')
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.legend(frameon=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vis_5_learned_degradation_rates.png'), dpi=300)
    plt.close()