"""Data loading, cleaning, feature transformation, and dataset splitting utilities."""

import numpy as np
import pandas as pd
import torch

COLS = ['system_age_years',
        'G',
        'ambient_temp_mean_c',
        'ambient_temp_max_c',
        'wind_speed_ms',
        'pm10_ugm3',
        'humidity_pct',
        'days_since_rain',
        'days_since_cleaning',
        'temp_coeff_pmax_per_c']
HGB_FEATS = COLS + ['poa_irradiance_kwh_m2',
                    'nominal_efficiency_pct',
                    'dc_ac_ratio',
                    'cumulative_poa_kwh_m2',
                    'latitude',
                    'elevation_m',
                    'capacity_kwp']


def load_and_preprocess_data(csv_path='data/solar_panel_degradation.csv'):
    """Loads raw PV dataset, performs data cleaning, and computes key operational features."""
    raw = pd.read_csv(csv_path, parse_dates=['date', 'install_date'])
    d = raw.copy()

    # Clean non-physical sentinel values
    d.loc[d.ambient_temp_mean_c < -90, 'ambient_temp_mean_c'] = np.nan
    d.loc[d.poa_irradiance_kwh_m2 > 2.2 * d.ghi_kwh_m2 + 0.5, 'poa_irradiance_kwh_m2'] = np.nan

    # Operational data mask
    keep = (d.poa_irradiance_kwh_m2.gt(2) &
            d.energy_ac_kwh.notna() &
            d.ambient_temp_mean_c.notna() &
            d.ambient_temp_max_c.notna() &
            (d.snow_depth_cm.fillna(0) == 0) &
            (d.curtailment_flag == 0))
    d = d[keep].reset_index(drop=True)

    # Derived physics features
    d['pr'] = d.energy_ac_kwh / (d.capacity_kwp * d.poa_irradiance_kwh_m2)
    d['G'] = d.poa_irradiance_kwh_m2 * 1000 / d.daylight_hours

    # Fill sparse weather features
    for col in ['wind_speed_ms', 'pm10_ugm3', 'humidity_pct']:
        d[col] = d[col].fillna(d[col].median())

    # Flag clean operational days
    d['is_clean'] = (d.outage_flag == 0) & d.pr.between(0.2, 1.0)

    # Encode categorical variables
    techs = sorted(d.panel_technology.unique())
    systems = sorted(d.system_id.unique())
    d['tid'] = d.panel_technology.map({t: i for i, t in enumerate(techs)})
    d['sid'] = d.system_id.map({s: i for i, s in enumerate(systems)})
    assert d[COLS].notna().all().all(), f"NaNs found in feature columns: {d[COLS].isna().sum()}"

    # Build tensor datasets
    X_all = torch.tensor(d[COLS].values, dtype=torch.float32)
    S_all = torch.tensor(d.sid.values, dtype=torch.long)
    T_all = torch.tensor(d.tid.values, dtype=torch.long)
    Y_all = torch.tensor(d.pr.values, dtype=torch.float32)
    return d, X_all, S_all, T_all, Y_all, techs, systems


def get_split(df, split_name):
    """Retrieves train and test sample indices for a given experimental split."""
    splits = {'time : train <2023, test >=2023': (df.date < '2023-01-01', df.date >= '2023-01-01'),
                'age  : train <8 y, test >=10 y': (df.system_age_years < 8, df.system_age_years >= 10)}
    if split_name not in splits:
        raise ValueError(f"Unknown split: {split_name}. Available: {list(splits.keys())}")
    tr_mask, te_mask = splits[split_name]
    tr = np.where(tr_mask & df.is_clean)[0]
    te = np.where(te_mask & df.is_clean & df.sid.isin(df.sid.values[tr]))[0]
    return tr, te