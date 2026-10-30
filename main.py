"""Main pipeline entrypoint to execute training, evaluation, and diagnostic visualization."""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm
from src.data import load_and_preprocess_data, get_split, HGB_FEATS
from src.models import HybridPV, MLPBaseline
from src.training import train, predict, violation_rate, score
from src.visualization import (
    plot_age_extrapolation,
    plot_monotonicity_violations,
    plot_residual_bounded_correction,
    plot_anomaly_detection_curves,
    plot_learned_degradation_rates
)


def main():
    parser = argparse.ArgumentParser(description="Physics-Constrained Hybrid PV Model Pipeline")
    parser.add_argument('--data_path', type=str, default='data/solar_panel_degradation.csv', help='CSV path')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output figures')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='List of random seeds')
    parser.add_argument('--show_epoch_progress', action='store_true', help='Display nested progress bar for training epochs')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare data
    print("Loading and preprocessing data...")
    df, X_all, S_all, T_all, Y_all, techs, systems = load_and_preprocess_data(args.data_path)
    print(f"Dataset ready: {len(df):,} usable rows across {len(systems)} systems and {len(techs)} technologies.\n")
    splits = ['time : train <2023, test >=2023', 'age  : train <8 y, test >=10 y']
    rows, store, store_all = [], {}, {}
    for split in splits:
        print(f"=== Running Split: {split} ===")
        tr, te = get_split(df, split)
        y_te = Y_all[te].numpy()
        Xtr = X_all[tr]

        # 1. Gradient Boosting Baseline
        for sd in tqdm(args.seeds, desc="HistGradientBoosting (Seeds)"):
            hgb = HistGradientBoostingRegressor(random_state=sd).fit(df.loc[tr, HGB_FEATS], df.pr.values[tr])
            p = hgb.predict(df.loc[te, HGB_FEATS])
            rows.append(dict(split=split, model='HistGradientBoosting', seed=sd, **score(y_te, p), mono_violation=np.nan))
            store_all[(split, 'HGB', sd)] = (hgb, p)
            if sd == args.seeds[0]:
                store[(split, 'HGB')] = (hgb, p)

        # 2. PyTorch Neural & Hybrid Models
        configs = [('MLP (no physics)', lambda: MLPBaseline(Xtr), 0.0),
                    ('Physics only', lambda: HybridPV(Xtr, len(systems), len(techs), use_nn=False), 0.0),
                    ('Hybrid, lam_mono=0', lambda: HybridPV(Xtr, len(systems), len(techs)), 0.0),
                    ('Hybrid, lam_mono=1', lambda: HybridPV(Xtr, len(systems), len(techs)), 1.0)]
        for name, build, lam in configs:
            for sd in tqdm(args.seeds, desc=f"{name:22s} (Seeds)"):
                torch.manual_seed(sd)
                m = train(build(), X_all, S_all, T_all, Y_all, tr, lam_mono=lam, seed=sd, show_progress=args.show_epoch_progress)
                p = predict(m, X_all, S_all, T_all, te)
                v_rate = violation_rate(m, X_all, S_all, T_all, te)
                rows.append(dict(split=split, model=name, seed=sd, **score(y_te, p), mono_violation=v_rate))
                store_all[(split, name, sd)] = (m, p)
                if sd == args.seeds[0]:
                    store[(split, name)] = (m, p)
        print(f"Completed split: {split}\n")

    # Generate summary results
    runs = pd.DataFrame(rows)
    g = runs.groupby(['split', 'model'], sort=False)
    metrics = ['R2', 'MAE', 'mono_violation']
    results = g[metrics].mean().reset_index().round(4)
    std = g[metrics].std().reset_index()
    summary = results.copy()
    for c in ['R2', 'MAE']:
        summary[c] = [f'{m:.4f} ± {np.nan_to_num(s):.4f}' for m, s in zip(results[c], std[c])]
    summary['mono_violation'] = ['-' if np.isnan(m) else f'{100*m:.1f}% ± {100*np.nan_to_num(s):.1f}%' for m, s in zip(results.mono_violation, std.mono_violation)]
    print("================ FINAL MODEL RESULTS ================")
    print(summary.to_string(index=False))

    # Generate Visualizations into output folder
    print(f"\nGenerating and saving figures to '{args.output_dir}/' folder...")
    plot_age_extrapolation(df, store, X_all, S_all, T_all, get_split, output_dir=args.output_dir)
    plot_monotonicity_violations(results, output_dir=args.output_dir)
    plot_residual_bounded_correction(X_all, store, output_dir=args.output_dir)
    plot_anomaly_detection_curves(df, store, X_all, S_all, T_all, get_split, output_dir=args.output_dir)
    plot_learned_degradation_rates(splits, store, techs, output_dir=args.output_dir)
    print(f"All 5 diagnostic plots saved successfully into '{args.output_dir}/'.")


if __name__ == '__main__':
    main()