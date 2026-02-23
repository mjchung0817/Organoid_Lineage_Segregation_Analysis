#!/usr/bin/env python3
"""
Predictive CN sidetrack pipeline.

Stages:
1) Freeze PCA target space from organoid-level spatial features.
2) Build probabilistic CN inputs (Poisson virtual-organoid summaries).
3) Train/evaluate Ridge, Poly-Ridge, and GAM-like models with strict holdouts:
   - LOR: leave-one-(experiment,replicate)-out
   - LOEX: leave-one-experiment-out
4) Report per-PC metrics, trajectory-level metrics, and back-projected feature errors.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

# Headless plotting + writable cache defaults for sandboxed runs.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_cn_sidetrack')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, SplineTransformer, StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: scikit-learn is required for predictive_cn_sidetrack. "
        "Install project requirements first, e.g. `pip install -r requirements.txt`."
    ) from exc


# -----------------------------
# Utility
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_cn_map(cn_map_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not cn_map_str:
        return out
    parts = [p.strip() for p in cn_map_str.split(',') if p.strip()]
    for p in parts:
        if ':' not in p:
            raise ValueError(f"Invalid cn-map entry '{p}'. Expected format exp:lambda")
        k, v = p.split(':', 1)
        out[k.strip()] = float(v.strip())
    return out


def setup_plot_style() -> None:
    sns.set_theme(style='whitegrid', context='talk')


def save_fold_metrics_plot(df_metrics: pd.DataFrame, out_png: str) -> None:
    if df_metrics.empty:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    r2_df = df_metrics.dropna(subset=['R2']).copy()
    if not r2_df.empty:
        sns.boxplot(
            data=r2_df,
            x='PC',
            y='R2',
            hue='Model',
            ax=axes[0]
        )
        axes[0].set_title('Fold-Level R2 by PC')
        axes[0].axhline(0.0, linestyle='--', linewidth=1, color='black', alpha=0.6)
        axes[0].set_ylabel('R2')
    else:
        axes[0].set_visible(False)

    sns.boxplot(
        data=df_metrics,
        x='PC',
        y='RMSE',
        hue='Model',
        ax=axes[1]
    )
    axes[1].set_title('Fold-Level RMSE by PC')
    axes[1].set_ylabel('RMSE')

    # Keep one legend to reduce clutter.
    if axes[1].legend_ is not None:
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend_.remove()
    else:
        handles, labels = [], []
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(3, len(labels)), frameon=True)

    plt.suptitle('Baseline Validation: Fold-Level PC Prediction Metrics', y=1.03, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_feature_deconvolution_plot(df_feat_summary: pd.DataFrame, out_png: str) -> None:
    if df_feat_summary.empty:
        return

    setup_plot_style()
    df_plot = df_feat_summary.copy()
    df_plot['Model_Split'] = df_plot['Model'].astype(str) + ' | ' + df_plot['Split_Type'].astype(str)

    mae_pivot = df_plot.pivot_table(index='Feature', columns='Model_Split', values='MAE', aggfunc='mean')
    rmse_pivot = df_plot.pivot_table(index='Feature', columns='Model_Split', values='RMSE', aggfunc='mean')

    fig, axes = plt.subplots(1, 2, figsize=(24, max(8, 0.45 * len(mae_pivot.index))))
    sns.heatmap(mae_pivot, cmap='YlGnBu', linewidths=0.4, linecolor='white', ax=axes[0])
    axes[0].set_title('Feature Deconvolution Error (MAE)')
    axes[0].set_xlabel('Model | Split')
    axes[0].set_ylabel('Feature')

    sns.heatmap(rmse_pivot, cmap='OrRd', linewidths=0.4, linecolor='white', ax=axes[1])
    axes[1].set_title('Feature Deconvolution Error (RMSE)')
    axes[1].set_xlabel('Model | Split')
    axes[1].set_ylabel('')

    plt.suptitle('Baseline Validation: Back-Projected Feature Error Summary', y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_stress_test_side_by_side_plot(df_metrics: pd.DataFrame, df_traj: pd.DataFrame, out_png: str) -> None:
    """
    Single figure comparing LOR vs LOEX directly (no pooled aggregation across split types).
    """
    if df_metrics.empty:
        return

    setup_plot_style()
    split_order = ['LOR', 'LOEX']
    split_label = {
        'LOR': 'LOR (leave-one-replicate-out)',
        'LOEX': 'LOEX (leave-one-experiment-out)',
    }

    fig, axes = plt.subplots(3, 2, figsize=(21, 16), squeeze=False)

    for c, split in enumerate(split_order):
        dms = df_metrics[df_metrics['Split_Type'].astype(str) == split].copy()
        dts = df_traj[df_traj['Split_Type'].astype(str) == split].copy() if not df_traj.empty else pd.DataFrame()

        if dms.empty:
            axes[0, c].set_visible(False)
            axes[1, c].set_visible(False)
            axes[2, c].set_visible(False)
            continue

        # Row 1: R2 by model and PC
        sns.barplot(
            data=dms, x='Model', y='R2', hue='PC', ax=axes[0, c], errorbar=None
        )
        axes[0, c].axhline(0.0, linestyle='--', linewidth=1, color='black', alpha=0.7)
        axes[0, c].set_title(f"{split_label.get(split, split)}: R2 by PC", fontweight='bold')
        axes[0, c].set_xlabel('')
        axes[0, c].tick_params(axis='x', rotation=15)

        # Row 2: RMSE by model and PC
        sns.barplot(
            data=dms, x='Model', y='RMSE', hue='PC', ax=axes[1, c], errorbar=None
        )
        axes[1, c].set_title(f"{split_label.get(split, split)}: RMSE by PC", fontweight='bold')
        axes[1, c].set_xlabel('')
        axes[1, c].tick_params(axis='x', rotation=15)

        # Row 3: Trajectory metrics by model
        if not dts.empty:
            traj_long = pd.concat([
                dts[['Model', 'Centroid_Path_Error']].rename(columns={'Centroid_Path_Error': 'Value'}).assign(Metric='Path Error'),
                dts[['Model', 'Endpoint_Error']].rename(columns={'Endpoint_Error': 'Value'}).assign(Metric='Endpoint Error'),
            ], ignore_index=True)

            sns.barplot(
                data=traj_long, x='Model', y='Value', hue='Metric', ax=axes[2, c], errorbar=None
            )
            axes[2, c].set_title(f"{split_label.get(split, split)}: Trajectory Distance Errors", fontweight='bold')
            axes[2, c].set_xlabel('')
            axes[2, c].tick_params(axis='x', rotation=15)

            # Annotate angle error per model.
            angle_map = dts.set_index('Model')['Direction_Angle_Error_Deg'].to_dict()
            y_top = axes[2, c].get_ylim()[1]
            for i, m in enumerate([t.get_text() for t in axes[2, c].get_xticklabels()]):
                if m in angle_map and pd.notna(angle_map[m]):
                    axes[2, c].text(
                        i, y_top * 0.94, f"angle={angle_map[m]:.1f}°",
                        ha='center', va='top', fontsize=9, color='black'
                    )
        else:
            axes[2, c].set_visible(False)

    # Legend cleanup: keep only one legend per row.
    for r in range(3):
        for c in range(2):
            if c == 1 and axes[r, c].legend_ is not None:
                axes[r, c].legend_.remove()

    plt.suptitle('Stress-Test Comparison: LOR vs LOEX', y=1.01, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_high_cn_extrapolation_performance_plot(df_extrap: pd.DataFrame, out_png: str) -> None:
    if df_extrap.empty:
        return

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))

    pc_df = df_extrap[df_extrap['PC'] != 'TRAJECTORY'].copy()
    traj_df = df_extrap[df_extrap['PC'] == 'TRAJECTORY'].copy()

    if not pc_df.empty:
        sns.barplot(data=pc_df, x='Model', y='R2', hue='PC', ax=axes[0, 0], errorbar=None)
        axes[0, 0].axhline(0.0, linestyle='--', linewidth=1, color='black', alpha=0.7)
        axes[0, 0].set_title('High-CN Extrapolation: R2 by PC')
        axes[0, 0].set_xlabel('')
        axes[0, 0].tick_params(axis='x', rotation=15)

        sns.barplot(data=pc_df, x='Model', y='RMSE', hue='PC', ax=axes[0, 1], errorbar=None)
        axes[0, 1].set_title('High-CN Extrapolation: RMSE by PC')
        axes[0, 1].set_xlabel('')
        axes[0, 1].tick_params(axis='x', rotation=15)
    else:
        axes[0, 0].set_visible(False)
        axes[0, 1].set_visible(False)

    if not traj_df.empty:
        traj_long = pd.concat([
            traj_df[['Model', 'RMSE']].rename(columns={'RMSE': 'Value'}).assign(Metric='Centroid Path Error'),
            traj_df[['Model', 'MAE']].rename(columns={'MAE': 'Value'}).assign(Metric='Endpoint Error'),
        ], ignore_index=True)

        sns.barplot(data=traj_long, x='Model', y='Value', hue='Metric', ax=axes[1, 0], errorbar=None)
        axes[1, 0].set_title('High-CN Extrapolation: Trajectory Distance Errors')
        axes[1, 0].set_xlabel('')
        axes[1, 0].tick_params(axis='x', rotation=15)

        sns.barplot(data=traj_df, x='Model', y='Direction_Angle_Error_Deg', ax=axes[1, 1], errorbar=None)
        axes[1, 1].set_title('High-CN Extrapolation: Direction Angle Error (deg)')
        axes[1, 1].set_xlabel('')
        axes[1, 1].tick_params(axis='x', rotation=15)
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    # Cleanup duplicate legends from top-row plots.
    for ax in (axes[0, 1],):
        if ax.legend_ is not None:
            ax.legend_.remove()

    plt.suptitle('Low-CN Train -> High-CN Test Performance', y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_high_cn_trajectory_plot(
    df_preds: pd.DataFrame,
    df_extrap: pd.DataFrame,
    pc_names: List[str],
    out_png: str,
) -> None:
    if df_preds.empty or len(pc_names) < 2:
        return

    model_order = sorted(df_preds['Model'].astype(str).unique())
    if not model_order:
        return

    pair_list: List[Tuple[str, str]] = [(pc_names[0], pc_names[1])]
    if len(pc_names) >= 3:
        pair_list.append((pc_names[1], pc_names[2]))

    setup_plot_style()
    n_rows = len(pair_list)
    n_cols = len(model_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    # Trajectory annotation lookup from extrap metrics.
    traj_info = {}
    if not df_extrap.empty:
        traj_df = df_extrap[df_extrap['PC'] == 'TRAJECTORY'].copy()
        for _, row in traj_df.iterrows():
            traj_info[str(row['Model'])] = {
                'path': row.get('RMSE', np.nan),
                'endpoint': row.get('MAE', np.nan),
                'angle': row.get('Direction_Angle_Error_Deg', np.nan),
            }

    for c, model in enumerate(model_order):
        mdf = df_preds[df_preds['Model'].astype(str) == model].copy()
        if mdf.empty:
            continue

        dox_vals = pd.to_numeric(mdf['Dox_Concentration'], errors='coerce').astype(float)
        cvals = dox_vals.values

        for r, (pcx, pcy) in enumerate(pair_list):
            ax = axes[r, c]
            tx = f'True_{pcx}'
            ty = f'True_{pcy}'
            px = f'Pred_{pcx}'
            py = f'Pred_{pcy}'

            # Sample points
            ax.scatter(mdf[tx], mdf[ty], c=cvals, cmap='viridis', alpha=0.28, s=28, marker='o', label='True samples')
            ax.scatter(mdf[px], mdf[py], c=cvals, cmap='plasma', alpha=0.28, s=28, marker='x', label='Pred samples')

            # Dox centroids
            cent_true = mdf.groupby('Dox_Concentration')[[tx, ty]].mean().sort_index()
            cent_pred = mdf.groupby('Dox_Concentration')[[px, py]].mean().sort_index()
            common_dox = [d for d in cent_true.index if d in cent_pred.index]

            if common_dox:
                t_arr = cent_true.loc[common_dox].values
                p_arr = cent_pred.loc[common_dox].values
                ax.plot(t_arr[:, 0], t_arr[:, 1], '-o', color='black', linewidth=2.2, markersize=5.5, label='True centroid path')
                ax.plot(p_arr[:, 0], p_arr[:, 1], '--x', color='crimson', linewidth=2.2, markersize=6.5, label='Pred centroid path')

                # annotate dox on true centroids
                for dox, (xv, yv) in zip(common_dox, t_arr):
                    ax.text(xv, yv, f'{int(dox)}', fontsize=8.5, color='black', alpha=0.9)

            ax.set_xlabel(pcx)
            ax.set_ylabel(pcy)
            ax.set_title(f'{model}: {pcx} vs {pcy}')

            if r == 0 and c == 0:
                ax.legend(loc='best', fontsize=9, frameon=True)

        # Add trajectory metric annotation in first row panel
        info = traj_info.get(model, {})
        if info:
            txt = (
                f"Path err: {info.get('path', np.nan):.2f}\n"
                f"Endpoint err: {info.get('endpoint', np.nan):.2f}\n"
                f"Angle err: {info.get('angle', np.nan):.1f} deg"
            )
            axes[0, c].text(
                0.02, 0.98, txt, transform=axes[0, c].transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
            )

    plt.suptitle('Low-CN Train -> High-CN Test: Predicted vs True Trajectories', y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def stable_row_seed(base_seed: int, key: str) -> int:
    # Deterministic seed from key without relying on hash randomization.
    h = 2166136261
    for ch in key.encode('utf-8'):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int((h + base_seed) & 0xFFFFFFFF)


def load_spatial_module(project_root: str):
    module_path = os.path.join(project_root, '20260208_spatial_state_trajectory_PCA.py')
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find spatial PCA script at: {module_path}")

    spec = importlib.util.spec_from_file_location('spatial_pca_module', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------
# Data assembly and PCA freezing
# -----------------------------

def collect_feature_matrix(spatial_mod, project_root: str, experiments: List[str], organoid_limit: int) -> pd.DataFrame:
    all_records = []

    for exp in experiments:
        if exp not in spatial_mod.DATASET_MAP:
            raise ValueError(f"Unknown experiment: {exp}")

        base_path = os.path.join(project_root, spatial_mod.DATASET_MAP[exp])
        if not os.path.isdir(base_path):
            print(f"[warn] Missing dataset directory for {exp}: {base_path}")
            continue

        all_files = glob.glob(os.path.join(base_path, '**/*.csv'), recursive=True)
        csv_files = spatial_mod.filter_first_n_organoids(all_files, organoid_limit)
        print(f"[{exp}] using {len(csv_files)} organoids (from {len(all_files)} csv files)")

        for file_path in csv_files:
            fname = os.path.basename(file_path)
            dox_match = re.search(r'(\d+)dox', fname)
            if not dox_match:
                continue
            dox = int(dox_match.group(1))
            replicate = os.path.basename(os.path.dirname(file_path))

            try:
                df = pd.read_csv(file_path)
                metrics = spatial_mod.compute_all_metrics(df)
            except Exception as exc:
                print(f"[warn] failed on {file_path}: {exc}")
                continue

            if metrics is None:
                continue

            metrics['Dox_Concentration'] = dox
            metrics['Replicate'] = replicate
            metrics['Experiment'] = exp
            metrics['File'] = file_path
            all_records.append(metrics)

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


@dataclass
class FrozenPCAArtifacts:
    scores: pd.DataFrame
    processed_features: pd.DataFrame
    valid_features: List[str]
    components: np.ndarray          # shape: [n_components, n_features]
    scaler_mean: np.ndarray         # shape: [n_features]
    scaler_scale: np.ndarray        # shape: [n_features]
    explained_variance_ratio: np.ndarray
    feature_order: List[str]
    pca_fit_basis: str


def freeze_pca_space(
    spatial_mod,
    df_features: pd.DataFrame,
    experiments: List[str],
    replicate_adjust: str,
    trajectory_group_by: str,
    pca_fit_basis: str,
    max_components: int,
    target_pcs: int,
) -> FrozenPCAArtifacts:
    if df_features.empty:
        raise ValueError('Feature matrix is empty. Cannot freeze PCA.')

    meta_cols = ['Dox_Concentration', 'Replicate', 'Experiment', 'File']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    valid_features = [c for c in feature_cols if df_features[c].notna().sum() > 0]
    if not valid_features:
        raise ValueError('No valid features available for PCA.')

    if replicate_adjust == 'raw':
        df_mode = spatial_mod.normalize_composition_features(df_features, clip_fractions=True)
    elif replicate_adjust == 'residualized':
        df_mode_norm = spatial_mod.normalize_composition_features(df_features, clip_fractions=True)
        df_mode = spatial_mod.residualize_by_replicate_within_dox(df_mode_norm, valid_features)
    else:
        raise ValueError("replicate_adjust must be 'raw' or 'residualized'")

    df_mode_imp, _, _ = spatial_mod.impute_missing_features(df_mode, valid_features)
    df_pca = df_mode_imp.dropna(subset=valid_features).copy()
    if len(df_pca) < 5:
        raise ValueError(f'Not enough organoids for PCA after preprocessing: n={len(df_pca)}')

    if pca_fit_basis == 'auto':
        pca_fit_basis_eff = (
            'exp_dox_centroids'
            if trajectory_group_by == 'experiment' and len(experiments) > 1
            else 'all_organoids'
        )
    else:
        pca_fit_basis_eff = pca_fit_basis

    X = df_pca[valid_features].values
    X_fit, n_fit_samples = spatial_mod.build_pca_fit_matrix(df_pca, valid_features, pca_fit_basis_eff)
    if n_fit_samples < 2:
        raise ValueError(f"Not enough PCA fit samples for basis '{pca_fit_basis_eff}'")

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_scaled = scaler.transform(X)

    n_components = min(len(valid_features), n_fit_samples - 1, max_components)
    if n_components < target_pcs:
        raise ValueError(
            f'Frozen PCA has n_components={n_components}, smaller than target_pcs={target_pcs}. '
            f'Adjust pca-fit-basis/max-components/target-pcs.'
        )

    pca = PCA(n_components=n_components)
    pca.fit(X_fit_scaled)
    scores = pca.transform(X_scaled)

    df_scores = df_pca[meta_cols].copy().reset_index(drop=True)
    for i in range(n_components):
        df_scores[f'PC{i+1}'] = scores[:, i]

    return FrozenPCAArtifacts(
        scores=df_scores,
        processed_features=df_pca[meta_cols + valid_features].copy().reset_index(drop=True),
        valid_features=valid_features,
        components=pca.components_.copy(),
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        explained_variance_ratio=pca.explained_variance_ratio_.copy(),
        feature_order=valid_features,
        pca_fit_basis=pca_fit_basis_eff,
    )


# -----------------------------
# Input engineering
# -----------------------------

def add_probabilistic_cn_inputs(
    df_scores: pd.DataFrame,
    cn_map: Dict[str, float],
    cn_cells: int,
    base_seed: int,
) -> pd.DataFrame:
    df = df_scores.copy()

    cn_lambda = []
    cn_sample_mean = []
    cn_sample_std = []
    cn_sample_p10 = []
    cn_sample_p50 = []
    cn_sample_p90 = []

    for _, row in df.iterrows():
        exp = str(row['Experiment'])
        file_key = str(row['File'])
        lam = cn_map.get(exp, np.nan)
        cn_lambda.append(lam)

        if pd.isna(lam):
            cn_sample_mean.append(np.nan)
            cn_sample_std.append(np.nan)
            cn_sample_p10.append(np.nan)
            cn_sample_p50.append(np.nan)
            cn_sample_p90.append(np.nan)
            continue

        rs = stable_row_seed(base_seed, file_key)
        rng = np.random.default_rng(rs)
        draws = rng.poisson(lam=lam, size=cn_cells)

        cn_sample_mean.append(float(np.mean(draws)))
        cn_sample_std.append(float(np.std(draws, ddof=1)))
        cn_sample_p10.append(float(np.percentile(draws, 10)))
        cn_sample_p50.append(float(np.percentile(draws, 50)))
        cn_sample_p90.append(float(np.percentile(draws, 90)))

    df['CN_Lambda'] = cn_lambda
    df['CN_Sample_Mean'] = cn_sample_mean
    df['CN_Sample_Std'] = cn_sample_std
    df['CN_Sample_P10'] = cn_sample_p10
    df['CN_Sample_P50'] = cn_sample_p50
    df['CN_Sample_P90'] = cn_sample_p90

    dox = pd.to_numeric(df['Dox_Concentration'], errors='coerce').astype(float)
    df['Dox_Raw'] = dox
    df['Dox_Log1p10'] = np.log10(dox + 1.0)

    return df


# -----------------------------
# Models
# -----------------------------

def poly_interaction_transform(X: np.ndarray) -> np.ndarray:
    # Input columns: [Dox_Log1p10, CN_Sample_Mean]
    dox = X[:, 0]
    cn = X[:, 1]
    return np.column_stack([
        dox,
        dox ** 2,
        dox ** 3,
        cn,
        dox * cn,
        (dox ** 2) * cn,
    ])


def selected_poly_feature_names() -> List[str]:
    return [
        'Dox_Log1p10',
        'Dox_Log1p10^2',
        'Dox_Log1p10^3',
        'CN_Sample_Mean',
        'Dox_Log1p10*CN_Sample_Mean',
        'Dox_Log1p10^2*CN_Sample_Mean',
    ]


class MultiTargetGAMLike:
    """
    Uses pyGAM if available; otherwise uses spline+rige fallback per target.
    """

    def __init__(self, n_knots: int = 5, alpha: float = 1.0):
        self.n_knots = n_knots
        self.alpha = alpha
        self.backend = None
        self.models = []

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.models = []
        try:
            from pygam import LinearGAM, s, te  # type: ignore
            self.backend = 'pygam'
            for i in range(Y.shape[1]):
                model = LinearGAM(s(0) + s(1) + te(0, 1)).fit(X, Y[:, i])
                self.models.append(model)
        except Exception:
            self.backend = 'spline_ridge_fallback'
            for i in range(Y.shape[1]):
                model = Pipeline([
                    ('spline', SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False)),
                    ('ridge', Ridge(alpha=self.alpha)),
                ])
                model.fit(X, Y[:, i])
                self.models.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError('Model not fit yet.')
        preds = [m.predict(X) for m in self.models]
        return np.column_stack(preds)


class RidgeLinearModel:
    def __init__(self, alpha: float = 1.0):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha)),
        ])

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class PolyRidgeModel:
    """
    Polynomial Ridge Regression = linear Ridge on polynomially expanded features.
    """
    def __init__(self, alpha: float = 1.0, mode: str = 'selected', degree: int = 3):
        self.mode = mode
        self.degree = degree

        if self.mode == 'full':
            poly_step = PolynomialFeatures(degree=self.degree, include_bias=False)
        elif self.mode == 'selected':
            poly_step = FunctionTransformer(poly_interaction_transform, validate=True)
        else:
            raise ValueError(f"Unsupported poly mode: {self.mode}")

        self.model = Pipeline([
            ('poly', poly_step),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha)),
        ])

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_names(self) -> List[str]:
        if self.mode == 'selected':
            return selected_poly_feature_names()
        return [f'poly_feature_{i}' for i in range(self.model.named_steps['poly'].n_output_features_)]


# -----------------------------
# Splits and metrics
# -----------------------------

def make_lor_splits(df: pd.DataFrame) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    key = df['Experiment'].astype(str) + '::' + df['Replicate'].astype(str)
    splits = []
    for group in sorted(key.unique()):
        test_idx = np.where(key.values == group)[0]
        train_idx = np.where(key.values != group)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((f'lor_{group}', train_idx, test_idx))
    return splits


def make_loex_splits(df: pd.DataFrame) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    splits = []
    for exp in sorted(df['Experiment'].astype(str).unique()):
        test_idx = np.where(df['Experiment'].astype(str).values == exp)[0]
        train_idx = np.where(df['Experiment'].astype(str).values != exp)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((f'loex_{exp}', train_idx, test_idx))
    return splits


def per_pc_metrics(y_true: np.ndarray, y_pred: np.ndarray, pc_names: List[str]) -> List[Dict[str, float]]:
    rows = []
    for i, pc in enumerate(pc_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        if len(yt) >= 2 and np.nanstd(yt) > 0:
            r2 = float(r2_score(yt, yp))
        else:
            r2 = np.nan

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))

        rows.append({
            'PC': pc,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
        })
    return rows


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return np.nan
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))


def trajectory_metrics(
    fold_df: pd.DataFrame,
    pc_names: List[str],
    true_prefix: str = 'True_',
    pred_prefix: str = 'Pred_',
) -> Dict[str, float]:
    path_errs = []
    endpoint_errs = []
    angle_errs = []

    for exp, exp_df in fold_df.groupby('Experiment'):
        true_cols = [f'{true_prefix}{pc}' for pc in pc_names]
        pred_cols = [f'{pred_prefix}{pc}' for pc in pc_names]

        cent_true = exp_df.groupby('Dox_Concentration')[true_cols].mean()
        cent_pred = exp_df.groupby('Dox_Concentration')[pred_cols].mean()

        common_dox = sorted(set(cent_true.index).intersection(set(cent_pred.index)))
        if len(common_dox) < 2:
            continue

        t_arr = cent_true.loc[common_dox].values
        p_arr = cent_pred.loc[common_dox].values

        dists = np.linalg.norm(t_arr - p_arr, axis=1)
        path_errs.append(float(np.mean(dists)))

        endpoint_errs.append(float(np.linalg.norm(t_arr[-1] - p_arr[-1])))

        local_angles = []
        for i in range(len(common_dox) - 1):
            vt = t_arr[i + 1] - t_arr[i]
            vp = p_arr[i + 1] - p_arr[i]
            ang = _angle_deg(vt, vp)
            if not np.isnan(ang):
                local_angles.append(ang)
        if local_angles:
            angle_errs.append(float(np.mean(local_angles)))

    return {
        'Centroid_Path_Error': float(np.mean(path_errs)) if path_errs else np.nan,
        'Endpoint_Error': float(np.mean(endpoint_errs)) if endpoint_errs else np.nan,
        'Direction_Angle_Error_Deg': float(np.mean(angle_errs)) if angle_errs else np.nan,
    }


# -----------------------------
# Reconstruction
# -----------------------------

def reconstruct_features_from_pred_pcs(
    y_pred: np.ndarray,
    components: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    n_total_components: int,
    n_pred_components: int,
) -> np.ndarray:
    # Fill unmodeled PCs with 0 (mean in standardized space).
    z_full = np.zeros((y_pred.shape[0], n_total_components), dtype=float)
    z_full[:, :n_pred_components] = y_pred

    x_std_hat = z_full @ components  # [n_samples, n_features]
    x_hat = (x_std_hat * scaler_scale.reshape(1, -1)) + scaler_mean.reshape(1, -1)
    return x_hat


# -----------------------------
# Main run
# -----------------------------

def run(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(project_root, 'predictive_cn_sidetrack', 'runs', f'run_{run_stamp}')
    artifacts_dir = os.path.join(run_dir, 'artifacts')
    metrics_dir = os.path.join(run_dir, 'metrics')
    preds_dir = os.path.join(run_dir, 'predictions')
    ensure_dir(run_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(metrics_dir)
    ensure_dir(preds_dir)

    print('=== Predictive CN sidetrack ===')
    print(f'Run dir: {run_dir}')

    experiments = args.experiments
    cn_map = parse_cn_map(args.cn_map)

    # 1) Load spatial module + build feature matrix
    spatial_mod = load_spatial_module(project_root)
    df_features = collect_feature_matrix(spatial_mod, project_root, experiments, args.organoid_limit)
    if df_features.empty:
        raise RuntimeError('No organoid features were collected. Check dataset paths and filters.')

    feature_csv = os.path.join(artifacts_dir, 'feature_matrix_raw.csv')
    df_features.to_csv(feature_csv, index=False)

    # 2) Freeze PCA space
    frozen = freeze_pca_space(
        spatial_mod=spatial_mod,
        df_features=df_features,
        experiments=experiments,
        replicate_adjust=args.replicate_adjust,
        trajectory_group_by=args.trajectory_group_by,
        pca_fit_basis=args.pca_fit_basis,
        max_components=args.max_components,
        target_pcs=args.target_pcs,
    )

    # Save frozen artifacts
    frozen.scores.to_csv(os.path.join(artifacts_dir, 'frozen_pca_scores.csv'), index=False)
    frozen.processed_features.to_csv(os.path.join(artifacts_dir, 'processed_feature_matrix.csv'), index=False)

    pd.DataFrame(frozen.components.T, index=frozen.feature_order,
                 columns=[f'PC{i+1}' for i in range(frozen.components.shape[0])]) \
        .to_csv(os.path.join(artifacts_dir, 'frozen_pca_loadings.csv'))

    pd.DataFrame({
        'Feature': frozen.feature_order,
        'Scaler_Mean': frozen.scaler_mean,
        'Scaler_Scale': frozen.scaler_scale,
    }).to_csv(os.path.join(artifacts_dir, 'frozen_scaler_params.csv'), index=False)

    pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(frozen.explained_variance_ratio))],
        'Explained_Variance_Ratio': frozen.explained_variance_ratio,
        'Cumulative_Variance': np.cumsum(frozen.explained_variance_ratio),
    }).to_csv(os.path.join(artifacts_dir, 'frozen_explained_variance.csv'), index=False)

    with open(os.path.join(artifacts_dir, 'valid_features.txt'), 'w', encoding='utf-8') as f:
        for feat in frozen.feature_order:
            f.write(f'{feat}\n')

    # 3) Build modeling dataset
    df_model = add_probabilistic_cn_inputs(
        frozen.scores,
        cn_map=cn_map,
        cn_cells=args.cn_cells,
        base_seed=args.seed,
    )

    # Remove rows missing CN lambda mapping
    before_n = len(df_model)
    df_model = df_model.dropna(subset=['CN_Lambda', 'CN_Sample_Mean', 'Dox_Log1p10']).copy()
    print(f'Dropped rows without CN mapping: {before_n - len(df_model)}')

    target_pcs = [f'PC{i+1}' for i in range(args.target_pcs)]
    for pc in target_pcs:
        if pc not in df_model.columns:
            raise ValueError(f'Requested target PC missing: {pc}')

    # Join observed processed features for reconstruction error analysis
    key_cols = ['File', 'Experiment', 'Replicate', 'Dox_Concentration']
    df_obs_feat = frozen.processed_features[key_cols + frozen.feature_order].copy()

    # Core base input matrix
    X_base = df_model[['Dox_Log1p10', 'CN_Sample_Mean']].values
    Y = df_model[target_pcs].values

    # 4) Build split sets
    split_jobs = []
    if args.eval_lor:
        split_jobs.extend([('LOR',) + s for s in make_lor_splits(df_model)])
    if args.eval_loex:
        split_jobs.extend([('LOEX',) + s for s in make_loex_splits(df_model)])

    if not split_jobs:
        raise ValueError('No evaluation splits selected. Enable --eval-lor and/or --eval-loex.')

    # 5) Model registry
    models = {
        'ridge_linear': lambda: RidgeLinearModel(alpha=args.ridge_alpha),
        'poly_ridge': lambda: PolyRidgeModel(alpha=args.poly_alpha, mode=args.poly_mode, degree=args.poly_degree),
        'gam_like': lambda: MultiTargetGAMLike(n_knots=args.gam_knots, alpha=args.gam_alpha),
    }

    metrics_rows = []
    traj_rows = []
    fold_pred_rows = []
    recon_rows = []

    for split_type, fold_name, train_idx, test_idx in split_jobs:
        X_train, X_test = X_base[train_idx], X_base[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        fold_meta = df_model.iloc[test_idx][key_cols].reset_index(drop=True)

        for model_name, builder in models.items():
            model = builder()
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            # per-PC metrics
            pc_metric_list = per_pc_metrics(Y_test, Y_pred, target_pcs)
            for pm in pc_metric_list:
                metrics_rows.append({
                    'Split_Type': split_type,
                    'Fold': fold_name,
                    'Model': model_name,
                    'PC': pm['PC'],
                    'R2': pm['R2'],
                    'RMSE': pm['RMSE'],
                    'MAE': pm['MAE'],
                    'Test_N': len(test_idx),
                })

            # build fold prediction dataframe
            pred_df = fold_meta.copy()
            for i, pc in enumerate(target_pcs):
                pred_df[f'True_{pc}'] = Y_test[:, i]
                pred_df[f'Pred_{pc}'] = Y_pred[:, i]
            pred_df['Split_Type'] = split_type
            pred_df['Fold'] = fold_name
            pred_df['Model'] = model_name

            fold_pred_rows.append(pred_df)

            # trajectory metrics
            t_metrics = trajectory_metrics(pred_df, target_pcs)
            traj_rows.append({
                'Split_Type': split_type,
                'Fold': fold_name,
                'Model': model_name,
                **t_metrics,
            })

            # back-projection to feature space
            xhat = reconstruct_features_from_pred_pcs(
                y_pred=Y_pred,
                components=frozen.components,
                scaler_mean=frozen.scaler_mean,
                scaler_scale=frozen.scaler_scale,
                n_total_components=frozen.components.shape[0],
                n_pred_components=args.target_pcs,
            )

            recon_df = fold_meta.copy()
            for j, feat in enumerate(frozen.feature_order):
                recon_df[f'PredFeat_{feat}'] = xhat[:, j]
            recon_df['Split_Type'] = split_type
            recon_df['Fold'] = fold_name
            recon_df['Model'] = model_name
            recon_rows.append(recon_df)

    # Save prediction-level outputs
    df_fold_preds = pd.concat(fold_pred_rows, ignore_index=True) if fold_pred_rows else pd.DataFrame()
    df_fold_preds.to_csv(os.path.join(preds_dir, 'fold_pc_predictions.csv'), index=False)

    df_recon = pd.concat(recon_rows, ignore_index=True) if recon_rows else pd.DataFrame()
    df_recon.to_csv(os.path.join(preds_dir, 'fold_feature_reconstruction_predictions.csv'), index=False)

    # Save metric outputs
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(metrics_dir, 'per_pc_metrics_by_fold.csv'), index=False)
    save_fold_metrics_plot(df_metrics, os.path.join(metrics_dir, 'per_pc_metrics_by_fold.png'))

    df_traj = pd.DataFrame(traj_rows)
    df_traj.to_csv(os.path.join(metrics_dir, 'trajectory_metrics_by_fold.csv'), index=False)
    save_stress_test_side_by_side_plot(
        df_metrics,
        df_traj,
        os.path.join(metrics_dir, 'stress_test_LOR_vs_LOEX.png')
    )

    # Summaries
    metric_summary = (
        df_metrics.groupby(['Split_Type', 'Model', 'PC'], as_index=False)[['R2', 'RMSE', 'MAE']]
        .mean(numeric_only=True)
    )
    metric_summary.to_csv(os.path.join(metrics_dir, 'per_pc_metrics_summary.csv'), index=False)

    traj_summary = (
        df_traj.groupby(['Split_Type', 'Model'], as_index=False)[
            ['Centroid_Path_Error', 'Endpoint_Error', 'Direction_Angle_Error_Deg']
        ].mean(numeric_only=True)
    )
    traj_summary.to_csv(os.path.join(metrics_dir, 'trajectory_metrics_summary.csv'), index=False)

    # Reconstruction errors against observed processed features
    if not df_recon.empty:
        recon_merge = df_recon.merge(df_obs_feat, on=key_cols, how='left')

        feat_err_rows = []
        for _, row in recon_merge.iterrows():
            for feat in frozen.feature_order:
                pred = row.get(f'PredFeat_{feat}', np.nan)
                obs = row.get(feat, np.nan)
                if pd.isna(pred) or pd.isna(obs):
                    continue
                feat_err_rows.append({
                    'Split_Type': row['Split_Type'],
                    'Fold': row['Fold'],
                    'Model': row['Model'],
                    'Feature': feat,
                    'Abs_Error': abs(float(pred) - float(obs)),
                    'Sq_Error': (float(pred) - float(obs)) ** 2,
                })

        df_feat_err = pd.DataFrame(feat_err_rows)
        df_feat_err.to_csv(os.path.join(metrics_dir, 'feature_reconstruction_errors_long.csv'), index=False)

        if not df_feat_err.empty:
            feat_summary = (
                df_feat_err.groupby(['Split_Type', 'Model', 'Feature'], as_index=False)
                .agg(MAE=('Abs_Error', 'mean'), RMSE=('Sq_Error', lambda s: float(np.sqrt(np.mean(s)))))
            )
            feat_summary_csv = os.path.join(metrics_dir, 'feature_reconstruction_errors_summary.csv')
            feat_summary.to_csv(feat_summary_csv, index=False)
            save_feature_deconvolution_plot(
                feat_summary,
                os.path.join(metrics_dir, 'feature_reconstruction_errors_summary.png')
            )

    # High-CN extrapolation scenario
    cn_lambdas = {exp: cn_map.get(exp, np.nan) for exp in sorted(df_model['Experiment'].unique())}
    valid_cn = {k: v for k, v in cn_lambdas.items() if pd.notna(v)}

    extrap_rows = []
    extrap_pred_rows = []
    if len(valid_cn) >= 2:
        max_cn = max(valid_cn.values())
        high_exp = sorted([k for k, v in valid_cn.items() if v == max_cn])
        test_mask = df_model['Experiment'].isin(high_exp).values
        train_mask = ~test_mask

        if np.sum(train_mask) > 5 and np.sum(test_mask) > 3:
            X_train, X_test = X_base[train_mask], X_base[test_mask]
            Y_train, Y_test = Y[train_mask], Y[test_mask]
            fold_meta = df_model.loc[test_mask, key_cols].reset_index(drop=True)

            for model_name, builder in models.items():
                model = builder()
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)

                pm_list = per_pc_metrics(Y_test, Y_pred, target_pcs)
                for pm in pm_list:
                    extrap_rows.append({
                        'Model': model_name,
                        'PC': pm['PC'],
                        'R2': pm['R2'],
                        'RMSE': pm['RMSE'],
                        'MAE': pm['MAE'],
                        'Train_Experiments': ';'.join(sorted(df_model.loc[train_mask, 'Experiment'].unique())),
                        'Test_Experiments': ';'.join(high_exp),
                        'Scenario': 'lower_cn_train_high_cn_test',
                    })

                pred_df = fold_meta.copy()
                for i, pc in enumerate(target_pcs):
                    pred_df[f'True_{pc}'] = Y_test[:, i]
                    pred_df[f'Pred_{pc}'] = Y_pred[:, i]
                pred_df['Model'] = model_name
                pred_df['Scenario'] = 'lower_cn_train_high_cn_test'
                extrap_pred_rows.append(pred_df.copy())
                t_metrics = trajectory_metrics(pred_df, target_pcs)
                extrap_rows.append({
                    'Model': model_name,
                    'PC': 'TRAJECTORY',
                    'R2': np.nan,
                    'RMSE': t_metrics['Centroid_Path_Error'],
                    'MAE': t_metrics['Endpoint_Error'],
                    'Train_Experiments': ';'.join(sorted(df_model.loc[train_mask, 'Experiment'].unique())),
                    'Test_Experiments': ';'.join(high_exp),
                    'Scenario': 'lower_cn_train_high_cn_test',
                    'Direction_Angle_Error_Deg': t_metrics['Direction_Angle_Error_Deg'],
                })

    if extrap_rows:
        df_extrap = pd.DataFrame(extrap_rows)
        df_extrap.to_csv(os.path.join(metrics_dir, 'high_cn_extrapolation_metrics.csv'), index=False)

        save_high_cn_extrapolation_performance_plot(
            df_extrap,
            os.path.join(metrics_dir, 'high_cn_extrapolation_performance.png')
        )

        if extrap_pred_rows:
            df_extrap_preds = pd.concat(extrap_pred_rows, ignore_index=True)
            df_extrap_preds.to_csv(
                os.path.join(preds_dir, 'high_cn_extrapolation_pc_predictions.csv'),
                index=False
            )
            save_high_cn_trajectory_plot(
                df_extrap_preds,
                df_extrap,
                target_pcs,
                os.path.join(metrics_dir, 'high_cn_extrapolation_trajectory_paths.png')
            )

    # Run manifest
    manifest = {
        'run_stamp': run_stamp,
        'experiments': experiments,
        'replicate_adjust': args.replicate_adjust,
        'trajectory_group_by': args.trajectory_group_by,
        'pca_fit_basis_requested': args.pca_fit_basis,
        'pca_fit_basis_effective': frozen.pca_fit_basis,
        'max_components': args.max_components,
        'target_pcs': args.target_pcs,
        'target_pc_names': target_pcs,
        'organoid_limit': args.organoid_limit,
        'cn_map': cn_map,
        'cn_cells': args.cn_cells,
        'seed': args.seed,
        'models': {
            'ridge_linear': {'alpha': args.ridge_alpha, 'inputs': ['Dox_Log1p10', 'CN_Sample_Mean']},
            'poly_ridge': {
                'alpha': args.poly_alpha,
                'mode': args.poly_mode,
                'degree': args.poly_degree,
                'inputs_selected_mode': selected_poly_feature_names(),
            },
            'gam_like': {'gam_knots': args.gam_knots, 'alpha_fallback': args.gam_alpha},
        },
        'valid_features': frozen.feature_order,
        'explained_variance_ratio': frozen.explained_variance_ratio.tolist(),
    }

    with open(os.path.join(run_dir, 'run_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Minimal markdown summary
    summary_path = os.path.join(run_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('# Predictive CN Sidetrack Summary\n\n')
        f.write(f'- Run: `{run_stamp}`\n')
        f.write(f'- Experiments: {", ".join(experiments)}\n')
        f.write(f'- PCA basis (effective): `{frozen.pca_fit_basis}`\n')
        f.write(f'- Replicate adjust: `{args.replicate_adjust}`\n')
        f.write(f'- Target PCs: {", ".join(target_pcs)}\n\n')

        if not metric_summary.empty:
            f.write('## Per-PC Metrics (Mean Across Folds)\n\n')
            try:
                f.write(metric_summary.to_markdown(index=False))
            except Exception:
                f.write(metric_summary.to_string(index=False))
            f.write('\n\n')

        if not traj_summary.empty:
            f.write('## Trajectory Metrics (Mean Across Folds)\n\n')
            try:
                f.write(traj_summary.to_markdown(index=False))
            except Exception:
                f.write(traj_summary.to_string(index=False))
            f.write('\n')

    print('Done.')
    print(f'Artifacts: {artifacts_dir}')
    print(f'Metrics:   {metrics_dir}')
    print(f'Outputs:   {run_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictive CN sidetrack pipeline (freeze PCA + LOR/LOEX modeling)')

    parser.add_argument('--experiments', nargs='+', default=['exp1', 'exp2_high_cn', 'exp2_low_cn'],
                        help='Experiments to include.')

    parser.add_argument('--replicate-adjust', choices=['raw', 'residualized'], default='residualized',
                        help='Feature preprocessing mode before freezing PCA.')
    parser.add_argument('--trajectory-group-by', choices=['dox', 'experiment'], default='experiment',
                        help='Grouping context for PCA fit-basis auto behavior.')
    parser.add_argument('--pca-fit-basis', choices=['auto', 'all_organoids', 'exp_dox_centroids', 'exp_centroids'],
                        default='exp_dox_centroids',
                        help='How scaler/PCA are fit during frozen target creation.')

    parser.add_argument('--organoid-limit', type=int, default=3,
                        help='Organoids per (replicate,dox). <=0 uses all.')
    parser.add_argument('--max-components', type=int, default=5,
                        help='Upper bound for frozen PCA components.')
    parser.add_argument('--target-pcs', type=int, default=3,
                        help='Number of leading PCs to predict.')

    parser.add_argument('--cn-map', type=str,
                        default='exp1:4.5,exp2_high_cn:9,exp2_low_cn:4,exp3:9',
                        help='CN lambda map, format exp:lambda,exp:lambda')
    parser.add_argument('--cn-cells', type=int, default=100,
                        help='Number of virtual cells per organoid for Poisson CN sampling.')

    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--poly-alpha', type=float, default=1.0)
    parser.add_argument('--poly-mode', choices=['selected', 'full'], default='selected',
                        help='Poly basis: selected terms (agreed biology-driven basis) or full sklearn PolynomialFeatures.')
    parser.add_argument('--poly-degree', type=int, default=3,
                        help='Polynomial degree when --poly-mode full (ignored for selected mode).')
    parser.add_argument('--gam-alpha', type=float, default=1.0)
    parser.add_argument('--gam-knots', type=int, default=5)

    parser.add_argument('--eval-lor', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate leave-one-(experiment,replicate)-out splits.')
    parser.add_argument('--eval-loex', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate leave-one-experiment-out splits.')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run(args)
