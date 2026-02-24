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
    df_plot = df_metrics.copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    r2_df = df_plot.dropna(subset=['R2']).copy()
    if not r2_df.empty:
        sns.boxplot(
            data=r2_df,
            x='PC',
            y='R2',
            hue='Model',
            ax=axes[0],
            showfliers=False,
        )
        axes[0].set_title('Fold-Level R2 by PC')
        axes[0].axhline(0.0, linestyle='--', linewidth=1, color='black', alpha=0.6)
        axes[0].set_ylabel('R2')
    else:
        axes[0].set_visible(False)

    sns.boxplot(
        data=df_plot,
        x='PC',
        y='RMSE',
        hue='Model',
        ax=axes[1],
        showfliers=False,
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
    split_order = ['RANDOM', 'LOR', 'LOEX']
    split_label = {
        'RANDOM': 'Random CV (sample-level split)',
        'LOR': 'LOR (leave-one-replicate-out)',
        'LOEX': 'LOEX (leave-one-experiment-out)',
    }

    fig, axes = plt.subplots(3, 3, figsize=(31, 16), squeeze=False)

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
        for c in range(3):
            if c > 0 and axes[r, c].legend_ is not None:
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
        axes[0, 0].set_title('Custom Holdout: R2 by PC')
        axes[0, 0].set_xlabel('')
        axes[0, 0].tick_params(axis='x', rotation=15)

        sns.barplot(data=pc_df, x='Model', y='RMSE', hue='PC', ax=axes[0, 1], errorbar=None)
        axes[0, 1].set_title('Custom Holdout: RMSE by PC')
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
        axes[1, 0].set_title('Custom Holdout: Trajectory Distance Errors')
        axes[1, 0].set_xlabel('')
        axes[1, 0].tick_params(axis='x', rotation=15)

        sns.barplot(data=traj_df, x='Model', y='Direction_Angle_Error_Deg', ax=axes[1, 1], errorbar=None)
        axes[1, 1].set_title('Custom Holdout: Direction Angle Error (deg)')
        axes[1, 1].set_xlabel('')
        axes[1, 1].tick_params(axis='x', rotation=15)
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    # Cleanup duplicate legends from top-row plots.
    for ax in (axes[0, 1],):
        if ax.legend_ is not None:
            ax.legend_.remove()

    plt.suptitle('Custom Train/Test Holdout Performance', y=1.02, fontweight='bold')
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

    plt.suptitle('Custom Train/Test Holdout: Predicted vs True Trajectories', y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_lor_pc1_1d_trace_plot(df_fold_preds: pd.DataFrame, out_png: str) -> None:
    if df_fold_preds.empty:
        return
    if 'True_PC1' not in df_fold_preds.columns or 'Pred_PC1' not in df_fold_preds.columns:
        return

    dfl = df_fold_preds[df_fold_preds['Split_Type'].astype(str) == 'LOR'].copy()
    if dfl.empty:
        return

    model_order = sorted(dfl['Model'].astype(str).unique())
    exp_order = sorted(dfl['Experiment'].astype(str).unique())
    if not model_order or not exp_order:
        return

    setup_plot_style()
    fig, axes = plt.subplots(len(model_order), 1, figsize=(14, 4.2 * len(model_order)), squeeze=False)
    palette = sns.color_palette('tab10', n_colors=len(exp_order))
    dox_ticks = sorted(pd.to_numeric(dfl['Dox_Concentration'], errors='coerce').dropna().astype(float).unique())

    for r, model in enumerate(model_order):
        ax = axes[r, 0]
        mdf = dfl[dfl['Model'].astype(str) == model].copy()
        if mdf.empty:
            ax.set_visible(False)
            continue

        for exp, color in zip(exp_order, palette):
            edf = mdf[mdf['Experiment'].astype(str) == exp].copy()
            if edf.empty:
                continue
            cent = (
                edf.groupby('Dox_Concentration', as_index=False)[['True_PC1', 'Pred_PC1']]
                .mean(numeric_only=True)
                .sort_values('Dox_Concentration')
            )
            dox = pd.to_numeric(cent['Dox_Concentration'], errors='coerce').astype(float).values
            ax.plot(dox, cent['True_PC1'].values, '-', color=color, linewidth=2.2, alpha=0.95, label=f'{exp} true')
            ax.plot(dox, cent['Pred_PC1'].values, '--', color=color, linewidth=2.2, alpha=0.95, label=f'{exp} pred')

        ax.axhline(0.0, linestyle=':', linewidth=1, color='black', alpha=0.6)
        ax.set_title(f'LOR PC1 1D Trace: {model}', fontweight='bold')
        ax.set_xlabel('Dox Concentration')
        ax.set_ylabel('PC1 centroid')
        if dox_ticks:
            ax.set_xticks(dox_ticks)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), frameon=True)

    plt.suptitle('LOR Stress Test: PC1 1D Trajectories (True vs Predicted)', y=1.01, fontweight='bold')
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


def apply_cluster_size_endo_mode(df_features: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in {'keep', 'drop', 'log1p'}:
        raise ValueError(f"Unsupported cluster-size-endo mode: {mode}")

    out = df_features.copy()
    feat = 'Cluster_Size_Endo'
    if feat not in out.columns:
        print('[warn] Cluster_Size_Endo column is not present; ablation mode has no effect.')
        return out

    col = pd.to_numeric(out[feat], errors='coerce')
    before_min = float(np.nanmin(col.values)) if col.notna().any() else float('nan')
    before_max = float(np.nanmax(col.values)) if col.notna().any() else float('nan')
    before_mean = float(np.nanmean(col.values)) if col.notna().any() else float('nan')

    if mode == 'drop':
        out = out.drop(columns=[feat])
        print(
            f"[ablation] Cluster_Size_Endo mode=drop (before: mean={before_mean:.3f}, "
            f"min={before_min:.3f}, max={before_max:.3f})"
        )
        return out

    if mode == 'log1p':
        # Stabilize heavy-tailed positive values while preserving order.
        clipped = np.clip(col.values, a_min=0.0, a_max=None)
        out[feat] = np.where(np.isnan(col.values), np.nan, np.log1p(clipped))
        after = pd.to_numeric(out[feat], errors='coerce')
        after_min = float(np.nanmin(after.values)) if after.notna().any() else float('nan')
        after_max = float(np.nanmax(after.values)) if after.notna().any() else float('nan')
        after_mean = float(np.nanmean(after.values)) if after.notna().any() else float('nan')
        print(
            f"[ablation] Cluster_Size_Endo mode=log1p "
            f"(before mean/min/max={before_mean:.3f}/{before_min:.3f}/{before_max:.3f}; "
            f"after mean/min/max={after_mean:.3f}/{after_min:.3f}/{after_max:.3f})"
        )
        return out

    print(
        f"[ablation] Cluster_Size_Endo mode=keep (mean/min/max={before_mean:.3f}/{before_min:.3f}/{before_max:.3f})"
    )
    return out


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

    # Re-anchor each experiment trajectory at its Dox=0 centroid in PCA space.
    pc_cols = sorted(
        [c for c in df.columns if re.fullmatch(r'PC\d+', str(c))],
        key=lambda c: int(str(c)[2:]),
    )
    dox_numeric = pd.to_numeric(df['Dox_Concentration'], errors='coerce').astype(float)
    if pc_cols:
        baseline_mask = np.isclose(dox_numeric.values, 0.0, atol=1e-8)
        if np.any(baseline_mask):
            origin_df = (
                df.loc[baseline_mask, ['Experiment'] + pc_cols]
                .groupby('Experiment', as_index=False)
                .mean(numeric_only=True)
            )
            offset_cols = {pc: f'{pc}_OriginOffset' for pc in pc_cols}
            origin_df = origin_df.rename(columns=offset_cols)
            df = df.merge(origin_df, on='Experiment', how='left')

            missing_origin = sorted(
                df.loc[df[f'{pc_cols[0]}_OriginOffset'].isna(), 'Experiment'].astype(str).unique()
            )
            if missing_origin:
                print(
                    '[warn] Missing Dox=0 origin for experiments: '
                    f'{", ".join(missing_origin)}. Leaving those trajectories uncentered.'
                )

            for pc in pc_cols:
                off_col = f'{pc}_OriginOffset'
                df[pc] = df[pc] - df[off_col].fillna(0.0)
            df = df.drop(columns=[f'{pc}_OriginOffset' for pc in pc_cols])
        else:
            print('[warn] No Dox=0 samples found. Skipping PC origin centering.')
    else:
        print('[warn] No PC columns detected in frozen scores. Skipping PC origin centering.')

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

    dox = dox_numeric
    df['Dox_Raw'] = dox
    df['Dox_Log1p10'] = np.log10(dox + 1.0)

    return df


def add_experiment_one_hot_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    exp_ohe = pd.get_dummies(df['Experiment'].astype(str), prefix='ExperimentID', dtype=float)
    exp_ohe = exp_ohe.reindex(sorted(exp_ohe.columns), axis=1)
    out = pd.concat([df.reset_index(drop=True), exp_ohe.reset_index(drop=True)], axis=1)
    return out, exp_ohe.columns.tolist()


def resolve_cn_encodings(requested: List[str]) -> List[str]:
    canonical = ['lambda', 'sample_mean', 'summary']
    if not requested:
        return canonical
    if 'all' in requested:
        return canonical
    out = []
    for enc in requested:
        if enc not in canonical:
            raise ValueError(f'Unsupported CN encoding: {enc}')
        if enc not in out:
            out.append(enc)
    return out


def get_cn_input_columns(encoding: str) -> List[str]:
    if encoding == 'lambda':
        return ['CN_Lambda']
    if encoding == 'sample_mean':
        return ['CN_Sample_Mean']
    if encoding == 'summary':
        return ['CN_Sample_Mean', 'CN_Sample_Std', 'CN_Sample_P10', 'CN_Sample_P50', 'CN_Sample_P90']
    raise ValueError(f'Unsupported CN encoding: {encoding}')


# -----------------------------
# Models
# -----------------------------

def poly_interaction_transform(X: np.ndarray) -> np.ndarray:
    # Input columns: [Dox_Log1p10, CN_primary, optional extra CN stats + nuisance one-hot terms...]
    dox = X[:, 0]
    cn = X[:, 1]
    core = np.column_stack([
        dox,
        dox ** 2,
        dox ** 3,
        cn,
        dox * cn,
        (dox ** 2) * cn,
    ])
    if X.shape[1] > 2:
        return np.column_stack([core, X[:, 2:]])
    return core


def selected_poly_feature_names(extra_feature_names: List[str] | None = None) -> List[str]:
    names = [
        'Dox_Log1p10',
        'Dox_Log1p10^2',
        'Dox_Log1p10^3',
        'CN_Primary',
        'Dox_Log1p10*CN_Primary',
        'Dox_Log1p10^2*CN_Primary',
    ]
    if extra_feature_names:
        names.extend(extra_feature_names)
    return names


class MultiTargetGAMLike:
    """
    Uses pyGAM if available; otherwise uses spline+rige fallback per target.
    """

    def __init__(self, n_knots: int = 5, alpha: float = 1.0, lam: float = 1.0):
        self.n_knots = n_knots
        self.alpha = alpha
        self.lam = lam
        self.backend = None
        self.models = []
        self.fallback_spline = None

    def _build_fallback_design(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if self.fallback_spline is None:
            raise RuntimeError('Fallback spline is not initialized.')
        X_spline = self.fallback_spline.fit_transform(X[:, :2]) if fit else self.fallback_spline.transform(X[:, :2])
        if X.shape[1] > 2:
            return np.column_stack([X_spline, X[:, 2:]])
        return X_spline

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.models = []
        self.fallback_spline = None
        try:
            from pygam import LinearGAM, l, s, te  # type: ignore
            self.backend = 'pygam'
            terms = s(0) + s(1) + te(0, 1)
            for i in range(2, X.shape[1]):
                terms += l(i)
            for i in range(Y.shape[1]):
                model = LinearGAM(terms, lam=self.lam).fit(X, Y[:, i])
                self.models.append(model)
        except Exception:
            self.backend = 'spline_ridge_fallback'
            self.fallback_spline = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False)
            X_design = self._build_fallback_design(X, fit=True)
            for i in range(Y.shape[1]):
                model = Ridge(alpha=self.alpha)
                model.fit(X_design, Y[:, i])
                self.models.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError('Model not fit yet.')
        if self.backend == 'spline_ridge_fallback':
            X_design = self._build_fallback_design(X, fit=False)
            preds = [m.predict(X_design) for m in self.models]
        else:
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


def make_random_splits(
    n_samples: int,
    n_splits: int = 10,
    test_size: float = 0.2,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    if n_samples < 5:
        return []
    n_test = max(1, int(round(n_samples * test_size)))
    n_test = min(n_test, n_samples - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    splits = []
    for i in range(n_splits):
        test_idx = np.sort(rng.choice(indices, size=n_test, replace=False))
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        train_idx = indices[train_mask]
        splits.append((f'random_{i+1:02d}', train_idx, test_idx))
    return splits


def tune_gam_lam_nested(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    lam_grid: List[float],
    n_inner_splits: int,
    inner_test_size: float,
    seed: int,
    n_knots: int,
    alpha: float,
) -> Tuple[float, pd.DataFrame]:
    if not lam_grid:
        return 1.0, pd.DataFrame()

    inner_jobs = make_random_splits(
        n_samples=len(X_train),
        n_splits=n_inner_splits,
        test_size=inner_test_size,
        seed=seed,
    )
    if not inner_jobs:
        return float(lam_grid[0]), pd.DataFrame()

    rows = []
    for lam in lam_grid:
        split_rmses = []
        backend_name = None
        for _, tr_idx, va_idx in inner_jobs:
            model = MultiTargetGAMLike(n_knots=n_knots, alpha=alpha, lam=float(lam))
            model.fit(X_train[tr_idx], Y_train[tr_idx])
            y_pred = model.predict(X_train[va_idx])
            backend_name = model.backend

            rmse_vals = []
            for i in range(Y_train.shape[1]):
                rmse_vals.append(float(np.sqrt(mean_squared_error(Y_train[va_idx][:, i], y_pred[:, i]))))
            split_rmses.append(float(np.mean(rmse_vals)))

        rows.append({
            'Candidate_Lam': float(lam),
            'Inner_Mean_RMSE': float(np.mean(split_rmses)),
            'Inner_Std_RMSE': float(np.std(split_rmses, ddof=1)) if len(split_rmses) > 1 else 0.0,
            'Backend': backend_name if backend_name else 'unknown',
        })

    df_tune = pd.DataFrame(rows).sort_values('Inner_Mean_RMSE', ascending=True).reset_index(drop=True)
    best_lam = float(df_tune.iloc[0]['Candidate_Lam'])
    return best_lam, df_tune


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

    feature_csv_unmod = os.path.join(artifacts_dir, 'feature_matrix_raw_unmodified.csv')
    df_features.to_csv(feature_csv_unmod, index=False)
    df_features = apply_cluster_size_endo_mode(df_features, args.cluster_size_endo_mode)
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
    df_model, exp_ohe_cols = add_experiment_one_hot_inputs(df_model)
    if exp_ohe_cols:
        print(f'Added nuisance one-hot inputs: {", ".join(exp_ohe_cols)}')
    else:
        print('[warn] No nuisance one-hot inputs were created.')

    target_pcs = [f'PC{i+1}' for i in range(args.target_pcs)]
    for pc in target_pcs:
        if pc not in df_model.columns:
            raise ValueError(f'Requested target PC missing: {pc}')

    # Join observed processed features for reconstruction error analysis
    key_cols = ['File', 'Experiment', 'Replicate', 'Dox_Concentration']
    df_obs_feat = frozen.processed_features[key_cols + frozen.feature_order].copy()

    # 4) Build split sets
    split_jobs = []
    if args.eval_random:
        split_jobs.extend([
            ('RANDOM',) + s
            for s in make_random_splits(
                n_samples=len(df_model),
                n_splits=args.random_splits,
                test_size=args.random_test_size,
                seed=args.seed,
            )
        ])
    if args.eval_lor:
        split_jobs.extend([('LOR',) + s for s in make_lor_splits(df_model)])
    if args.eval_loex:
        split_jobs.extend([('LOEX',) + s for s in make_loex_splits(df_model)])
    if not split_jobs:
        raise ValueError('No evaluation splits selected. Enable --eval-lor and/or --eval-loex.')

    cn_encodings = resolve_cn_encodings(args.cn_encodings)
    print(f"CN encodings for ablation: {', '.join(cn_encodings)}")

    all_metric_frames: List[pd.DataFrame] = []
    all_traj_frames: List[pd.DataFrame] = []
    all_fold_pred_frames: List[pd.DataFrame] = []
    all_recon_frames: List[pd.DataFrame] = []
    all_feat_err_frames: List[pd.DataFrame] = []
    all_extrap_frames: List[pd.DataFrame] = []
    all_extrap_pred_frames: List[pd.DataFrame] = []
    all_gam_tune_frames: List[pd.DataFrame] = []
    encoding_run_configs: Dict[str, Dict[str, object]] = {}

    for cn_encoding in cn_encodings:
        cn_cols = get_cn_input_columns(cn_encoding)
        input_feature_cols = ['Dox_Log1p10'] + cn_cols + exp_ohe_cols
        missing_input = [c for c in input_feature_cols if c not in df_model.columns]
        if missing_input:
            raise ValueError(
                f"Missing columns for CN encoding '{cn_encoding}': {', '.join(missing_input)}"
            )

        print(f"\n--- Encoding: {cn_encoding} | inputs: {', '.join(input_feature_cols)}")
        X_model = df_model[input_feature_cols].to_numpy(dtype=float)
        exp_feature_idx = list(range(1 + len(cn_cols), X_model.shape[1]))
        Y = df_model[target_pcs].values

        model_builders = {
            'ridge_linear': lambda: RidgeLinearModel(alpha=args.ridge_alpha),
            'poly_ridge': lambda: PolyRidgeModel(alpha=args.poly_alpha, mode=args.poly_mode, degree=args.poly_degree),
        }
        model_order = ['ridge_linear', 'poly_ridge', 'spline']
        gam_tune_rows = []

        metrics_rows = []
        traj_rows = []
        fold_pred_rows = []
        recon_rows = []
        feat_err_rows = []
        extrap_rows = []
        extrap_pred_rows = []

        for split_type, fold_name, train_idx, test_idx in split_jobs:
            X_train = X_model[train_idx].copy()
            X_test = X_model[test_idx].copy()
            if split_type == 'LOEX' and exp_feature_idx:
                # Prevent identity leakage at LOEX inference time.
                X_test[:, exp_feature_idx] = 0.0
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            fold_meta = df_model.iloc[test_idx][key_cols].reset_index(drop=True)

            for model_name in model_order:
                gam_lam_used = np.nan
                gam_tuned = False
                if model_name == 'spline':
                    gam_lam_used = float(args.gam_lam)
                    if args.gam_tune_lam and split_type in args.gam_lam_tune_splits:
                        tuned_lam, df_tune = tune_gam_lam_nested(
                            X_train=X_train,
                            Y_train=Y_train,
                            lam_grid=args.gam_lam_grid,
                            n_inner_splits=args.gam_inner_splits,
                            inner_test_size=args.gam_inner_test_size,
                            seed=args.seed,
                            n_knots=args.gam_knots,
                            alpha=args.gam_alpha,
                        )
                        gam_lam_used = float(tuned_lam)
                        gam_tuned = True
                        if not df_tune.empty:
                            df_t = df_tune.copy()
                            df_t['CN_Encoding'] = cn_encoding
                            df_t['Split_Type'] = split_type
                            df_t['Fold'] = fold_name
                            df_t['Selected_Lam'] = gam_lam_used
                            df_t['Is_Selected'] = np.isclose(df_t['Candidate_Lam'].values, gam_lam_used)
                            gam_tune_rows.append(df_t)
                    model = MultiTargetGAMLike(
                        n_knots=args.gam_knots,
                        alpha=args.gam_alpha,
                        lam=float(gam_lam_used),
                    )
                else:
                    model = model_builders[model_name]()
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)

                pc_metric_list = per_pc_metrics(Y_test, Y_pred, target_pcs)
                for pm in pc_metric_list:
                    metrics_rows.append({
                        'CN_Encoding': cn_encoding,
                        'Split_Type': split_type,
                        'Fold': fold_name,
                        'Model': model_name,
                        'PC': pm['PC'],
                        'R2': pm['R2'],
                        'RMSE': pm['RMSE'],
                        'MAE': pm['MAE'],
                        'GAM_Lam': gam_lam_used,
                        'GAM_Tuned': gam_tuned,
                        'Test_N': len(test_idx),
                    })

                pred_df = fold_meta.copy()
                for i, pc in enumerate(target_pcs):
                    pred_df[f'True_{pc}'] = Y_test[:, i]
                    pred_df[f'Pred_{pc}'] = Y_pred[:, i]
                pred_df['CN_Encoding'] = cn_encoding
                pred_df['Split_Type'] = split_type
                pred_df['Fold'] = fold_name
                pred_df['Model'] = model_name
                pred_df['GAM_Lam'] = gam_lam_used
                pred_df['GAM_Tuned'] = gam_tuned
                fold_pred_rows.append(pred_df)

                t_metrics = trajectory_metrics(pred_df, target_pcs)
                traj_rows.append({
                    'CN_Encoding': cn_encoding,
                    'Split_Type': split_type,
                    'Fold': fold_name,
                    'Model': model_name,
                    'GAM_Lam': gam_lam_used,
                    'GAM_Tuned': gam_tuned,
                    **t_metrics,
                })

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
                recon_df['CN_Encoding'] = cn_encoding
                recon_df['Split_Type'] = split_type
                recon_df['Fold'] = fold_name
                recon_df['Model'] = model_name
                recon_df['GAM_Lam'] = gam_lam_used
                recon_df['GAM_Tuned'] = gam_tuned
                recon_rows.append(recon_df)

        df_fold_preds = pd.concat(fold_pred_rows, ignore_index=True) if fold_pred_rows else pd.DataFrame()
        df_recon = pd.concat(recon_rows, ignore_index=True) if recon_rows else pd.DataFrame()
        df_metrics = pd.DataFrame(metrics_rows)
        df_traj = pd.DataFrame(traj_rows)

        df_fold_preds.to_csv(
            os.path.join(preds_dir, f'fold_pc_predictions_{cn_encoding}.csv'),
            index=False
        )
        df_recon.to_csv(
            os.path.join(preds_dir, f'fold_feature_reconstruction_predictions_{cn_encoding}.csv'),
            index=False
        )
        df_metrics.to_csv(
            os.path.join(metrics_dir, f'per_pc_metrics_by_fold_{cn_encoding}.csv'),
            index=False
        )
        df_traj.to_csv(
            os.path.join(metrics_dir, f'trajectory_metrics_by_fold_{cn_encoding}.csv'),
            index=False
        )

        save_lor_pc1_1d_trace_plot(
            df_fold_preds,
            os.path.join(metrics_dir, f'lor_pc1_1d_trajectory_traces_{cn_encoding}.png')
        )
        save_fold_metrics_plot(
            df_metrics,
            os.path.join(metrics_dir, f'per_pc_metrics_by_fold_{cn_encoding}.png')
        )
        save_stress_test_side_by_side_plot(
            df_metrics,
            df_traj,
            os.path.join(metrics_dir, f'stress_test_LOR_vs_LOEX_{cn_encoding}.png')
        )

        metric_summary_enc = (
            df_metrics.groupby(['CN_Encoding', 'Split_Type', 'Model', 'PC'], as_index=False)[['R2', 'RMSE', 'MAE']]
            .mean(numeric_only=True)
        )
        traj_summary_enc = (
            df_traj.groupby(['CN_Encoding', 'Split_Type', 'Model'], as_index=False)[
                ['Centroid_Path_Error', 'Endpoint_Error', 'Direction_Angle_Error_Deg']
            ].mean(numeric_only=True)
        )
        metric_summary_enc.to_csv(
            os.path.join(metrics_dir, f'per_pc_metrics_summary_{cn_encoding}.csv'),
            index=False
        )
        traj_summary_enc.to_csv(
            os.path.join(metrics_dir, f'trajectory_metrics_summary_{cn_encoding}.csv'),
            index=False
        )

        if not df_recon.empty:
            recon_merge = df_recon.merge(df_obs_feat, on=key_cols, how='left')
            for _, row in recon_merge.iterrows():
                for feat in frozen.feature_order:
                    pred = row.get(f'PredFeat_{feat}', np.nan)
                    obs = row.get(feat, np.nan)
                    if pd.isna(pred) or pd.isna(obs):
                        continue
                    feat_err_rows.append({
                        'CN_Encoding': cn_encoding,
                        'Split_Type': row['Split_Type'],
                        'Fold': row['Fold'],
                        'Model': row['Model'],
                        'Feature': feat,
                        'Abs_Error': abs(float(pred) - float(obs)),
                        'Sq_Error': (float(pred) - float(obs)) ** 2,
                    })

        df_feat_err = pd.DataFrame(feat_err_rows)
        if not df_feat_err.empty:
            df_feat_err.to_csv(
                os.path.join(metrics_dir, f'feature_reconstruction_errors_long_{cn_encoding}.csv'),
                index=False
            )
            feat_summary = (
                df_feat_err.groupby(['CN_Encoding', 'Split_Type', 'Model', 'Feature'], as_index=False)
                .agg(MAE=('Abs_Error', 'mean'), RMSE=('Sq_Error', lambda s: float(np.sqrt(np.mean(s)))))
            )
            feat_summary.to_csv(
                os.path.join(metrics_dir, f'feature_reconstruction_errors_summary_{cn_encoding}.csv'),
                index=False
            )
            save_feature_deconvolution_plot(
                feat_summary,
                os.path.join(metrics_dir, f'feature_reconstruction_errors_summary_{cn_encoding}.png')
            )

        # Custom train/test holdout scenario.
        available_experiments = set(df_model['Experiment'].astype(str).unique())
        requested_train_exps = [str(x) for x in args.stress_train_experiments]
        requested_test_exps = [str(x) for x in args.stress_test_experiments]
        overlap_exps = sorted(set(requested_train_exps).intersection(set(requested_test_exps)))
        if overlap_exps:
            raise ValueError(
                f'Stress-test train/test experiment lists overlap: {", ".join(overlap_exps)}'
            )

        train_exps = [e for e in requested_train_exps if e in available_experiments]
        test_exps = [e for e in requested_test_exps if e in available_experiments]
        missing_train = sorted(set(requested_train_exps) - set(train_exps))
        missing_test = sorted(set(requested_test_exps) - set(test_exps))
        if missing_train:
            print(f"[warn] Stress-test train experiments missing from data: {', '.join(missing_train)}")
        if missing_test:
            print(f"[warn] Stress-test test experiments missing from data: {', '.join(missing_test)}")

        if train_exps and test_exps:
            test_mask = df_model['Experiment'].isin(test_exps).values
            train_mask = df_model['Experiment'].isin(train_exps).values
            if np.sum(train_mask) > 5 and np.sum(test_mask) > 3:
                X_train = X_model[train_mask].copy()
                X_test = X_model[test_mask].copy()
                if exp_feature_idx:
                    X_test[:, exp_feature_idx] = 0.0
                Y_train, Y_test = Y[train_mask], Y[test_mask]
                fold_meta = df_model.loc[test_mask, key_cols].reset_index(drop=True)
                scenario_name = 'custom_train_test_holdout'

                for model_name in model_order:
                    gam_lam_used = np.nan
                    gam_tuned = False
                    if model_name == 'spline':
                        gam_lam_used = float(args.gam_lam)
                        if args.gam_tune_lam:
                            tuned_lam, df_tune = tune_gam_lam_nested(
                                X_train=X_train,
                                Y_train=Y_train,
                                lam_grid=args.gam_lam_grid,
                                n_inner_splits=args.gam_inner_splits,
                                inner_test_size=args.gam_inner_test_size,
                                seed=args.seed,
                                n_knots=args.gam_knots,
                                alpha=args.gam_alpha,
                            )
                            gam_lam_used = float(tuned_lam)
                            gam_tuned = True
                            if not df_tune.empty:
                                df_t = df_tune.copy()
                                df_t['CN_Encoding'] = cn_encoding
                                df_t['Split_Type'] = 'CUSTOM_STRESS'
                                df_t['Fold'] = 'custom_train_test_holdout'
                                df_t['Selected_Lam'] = gam_lam_used
                                df_t['Is_Selected'] = np.isclose(df_t['Candidate_Lam'].values, gam_lam_used)
                                gam_tune_rows.append(df_t)
                        model = MultiTargetGAMLike(
                            n_knots=args.gam_knots,
                            alpha=args.gam_alpha,
                            lam=float(gam_lam_used),
                        )
                    else:
                        model = model_builders[model_name]()
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)

                    pm_list = per_pc_metrics(Y_test, Y_pred, target_pcs)
                    for pm in pm_list:
                        extrap_rows.append({
                            'CN_Encoding': cn_encoding,
                            'Model': model_name,
                            'PC': pm['PC'],
                            'R2': pm['R2'],
                            'RMSE': pm['RMSE'],
                            'MAE': pm['MAE'],
                            'GAM_Lam': gam_lam_used,
                            'GAM_Tuned': gam_tuned,
                            'Train_Experiments': ';'.join(train_exps),
                            'Test_Experiments': ';'.join(test_exps),
                            'Scenario': scenario_name,
                        })

                    pred_df = fold_meta.copy()
                    for i, pc in enumerate(target_pcs):
                        pred_df[f'True_{pc}'] = Y_test[:, i]
                        pred_df[f'Pred_{pc}'] = Y_pred[:, i]
                    pred_df['CN_Encoding'] = cn_encoding
                    pred_df['Model'] = model_name
                    pred_df['GAM_Lam'] = gam_lam_used
                    pred_df['GAM_Tuned'] = gam_tuned
                    pred_df['Scenario'] = scenario_name
                    extrap_pred_rows.append(pred_df.copy())
                    t_metrics = trajectory_metrics(pred_df, target_pcs)
                    extrap_rows.append({
                        'CN_Encoding': cn_encoding,
                        'Model': model_name,
                        'PC': 'TRAJECTORY',
                        'R2': np.nan,
                        'RMSE': t_metrics['Centroid_Path_Error'],
                        'MAE': t_metrics['Endpoint_Error'],
                        'GAM_Lam': gam_lam_used,
                        'GAM_Tuned': gam_tuned,
                        'Train_Experiments': ';'.join(train_exps),
                        'Test_Experiments': ';'.join(test_exps),
                        'Scenario': scenario_name,
                        'Direction_Angle_Error_Deg': t_metrics['Direction_Angle_Error_Deg'],
                    })
            else:
                print(
                    '[warn] Not enough samples for custom stress-test holdout: '
                    f'train_n={int(np.sum(train_mask))}, test_n={int(np.sum(test_mask))}'
                )
        else:
            print('[warn] Custom stress-test holdout skipped due to empty train or test experiment set.')

        df_extrap = pd.DataFrame(extrap_rows)
        if not df_extrap.empty:
            df_extrap.to_csv(
                os.path.join(metrics_dir, f'high_cn_extrapolation_metrics_{cn_encoding}.csv'),
                index=False
            )
            save_high_cn_extrapolation_performance_plot(
                df_extrap,
                os.path.join(metrics_dir, f'high_cn_extrapolation_performance_{cn_encoding}.png')
            )
            if extrap_pred_rows:
                df_extrap_preds = pd.concat(extrap_pred_rows, ignore_index=True)
                df_extrap_preds.to_csv(
                    os.path.join(preds_dir, f'high_cn_extrapolation_pc_predictions_{cn_encoding}.csv'),
                    index=False
                )
                save_high_cn_trajectory_plot(
                    df_extrap_preds,
                    df_extrap,
                    target_pcs,
                    os.path.join(metrics_dir, f'high_cn_extrapolation_trajectory_paths_{cn_encoding}.png')
                )
        else:
            df_extrap_preds = pd.DataFrame()

        df_gam_tune = pd.concat(gam_tune_rows, ignore_index=True) if gam_tune_rows else pd.DataFrame()
        if not df_gam_tune.empty:
            df_gam_tune.to_csv(
                os.path.join(metrics_dir, f'gam_lam_tuning_by_fold_{cn_encoding}.csv'),
                index=False
            )
            all_gam_tune_frames.append(df_gam_tune)

        # Collect for combined exports.
        all_metric_frames.append(df_metrics)
        all_traj_frames.append(df_traj)
        all_fold_pred_frames.append(df_fold_preds)
        all_recon_frames.append(df_recon)
        if not df_feat_err.empty:
            all_feat_err_frames.append(df_feat_err)
        if not df_extrap.empty:
            all_extrap_frames.append(df_extrap)
        if not df_extrap_preds.empty:
            all_extrap_pred_frames.append(df_extrap_preds)

        encoding_run_configs[cn_encoding] = {
            'input_feature_cols': input_feature_cols,
            'experiment_ohe_cols': exp_ohe_cols,
            'exp_feature_idx': exp_feature_idx,
        }

    # Combined exports across CN encodings.
    df_metrics = pd.concat(all_metric_frames, ignore_index=True) if all_metric_frames else pd.DataFrame()
    df_traj = pd.concat(all_traj_frames, ignore_index=True) if all_traj_frames else pd.DataFrame()
    df_fold_preds = pd.concat(all_fold_pred_frames, ignore_index=True) if all_fold_pred_frames else pd.DataFrame()
    df_recon = pd.concat(all_recon_frames, ignore_index=True) if all_recon_frames else pd.DataFrame()
    df_feat_err = pd.concat(all_feat_err_frames, ignore_index=True) if all_feat_err_frames else pd.DataFrame()
    df_extrap = pd.concat(all_extrap_frames, ignore_index=True) if all_extrap_frames else pd.DataFrame()
    df_extrap_preds = pd.concat(all_extrap_pred_frames, ignore_index=True) if all_extrap_pred_frames else pd.DataFrame()

    df_fold_preds.to_csv(os.path.join(preds_dir, 'fold_pc_predictions.csv'), index=False)
    df_recon.to_csv(os.path.join(preds_dir, 'fold_feature_reconstruction_predictions.csv'), index=False)
    if not df_extrap_preds.empty:
        df_extrap_preds.to_csv(os.path.join(preds_dir, 'high_cn_extrapolation_pc_predictions.csv'), index=False)

    df_metrics.to_csv(os.path.join(metrics_dir, 'per_pc_metrics_by_fold.csv'), index=False)
    df_traj.to_csv(os.path.join(metrics_dir, 'trajectory_metrics_by_fold.csv'), index=False)

    metric_summary = (
        df_metrics.groupby(['CN_Encoding', 'Split_Type', 'Model', 'PC'], as_index=False)[['R2', 'RMSE', 'MAE']]
        .mean(numeric_only=True)
    ) if not df_metrics.empty else pd.DataFrame()
    traj_summary = (
        df_traj.groupby(['CN_Encoding', 'Split_Type', 'Model'], as_index=False)[
            ['Centroid_Path_Error', 'Endpoint_Error', 'Direction_Angle_Error_Deg']
        ].mean(numeric_only=True)
    ) if not df_traj.empty else pd.DataFrame()
    metric_summary.to_csv(os.path.join(metrics_dir, 'per_pc_metrics_summary.csv'), index=False)
    traj_summary.to_csv(os.path.join(metrics_dir, 'trajectory_metrics_summary.csv'), index=False)

    if not df_feat_err.empty:
        df_feat_err.to_csv(os.path.join(metrics_dir, 'feature_reconstruction_errors_long.csv'), index=False)
        feat_summary = (
            df_feat_err.groupby(['CN_Encoding', 'Split_Type', 'Model', 'Feature'], as_index=False)
            .agg(MAE=('Abs_Error', 'mean'), RMSE=('Sq_Error', lambda s: float(np.sqrt(np.mean(s)))))
        )
        feat_summary.to_csv(os.path.join(metrics_dir, 'feature_reconstruction_errors_summary.csv'), index=False)

    if not df_extrap.empty:
        df_extrap.to_csv(os.path.join(metrics_dir, 'high_cn_extrapolation_metrics.csv'), index=False)
    if all_gam_tune_frames:
        pd.concat(all_gam_tune_frames, ignore_index=True).to_csv(
            os.path.join(metrics_dir, 'gam_lam_tuning_by_fold.csv'),
            index=False
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
        'cluster_size_endo_mode': args.cluster_size_endo_mode,
        'cn_map': cn_map,
        'cn_cells': args.cn_cells,
        'cn_encodings': cn_encodings,
        'seed': args.seed,
        'custom_stress_holdout': {
            'train_experiments': [str(x) for x in args.stress_train_experiments],
            'test_experiments': [str(x) for x in args.stress_test_experiments],
        },
        'evaluation_splits': {
            'random_enabled': args.eval_random,
            'random_splits': args.random_splits,
            'random_test_size': args.random_test_size,
            'lor_enabled': args.eval_lor,
            'loex_enabled': args.eval_loex,
        },
        'gam_tuning': {
            'enabled': args.gam_tune_lam,
            'default_lam': args.gam_lam,
            'lam_grid': args.gam_lam_grid,
            'tune_splits': args.gam_lam_tune_splits,
            'inner_splits': args.gam_inner_splits,
            'inner_test_size': args.gam_inner_test_size,
        },
        'encoding_run_configs': encoding_run_configs,
        'models': {
            'ridge_linear': {
                'alpha': args.ridge_alpha,
                'definition': 'StandardScaler + Ridge on selected input columns per CN encoding.',
                'loex_test_override': 'ExperimentID_* columns are forced to 0 at LOEX prediction time.',
            },
            'poly_ridge': {
                'alpha': args.poly_alpha,
                'mode': args.poly_mode,
                'degree': args.poly_degree,
                'inputs_selected_mode': selected_poly_feature_names(exp_ohe_cols),
                'definition': 'Polynomial feature expansion + StandardScaler + Ridge.',
                'loex_test_override': 'ExperimentID_* columns are forced to 0 at LOEX prediction time.',
            },
            'spline': {
                'gam_knots': args.gam_knots,
                'alpha_fallback': args.gam_alpha,
                'definition': 'pyGAM if available; otherwise spline-basis + Ridge fallback.',
                'loex_test_override': 'ExperimentID_* columns are forced to 0 at LOEX prediction time.',
                'default_lam': args.gam_lam,
                'lam_tuning_enabled': args.gam_tune_lam,
            },
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
        f.write(f'- Cluster_Size_Endo mode: `{args.cluster_size_endo_mode}`\n')
        f.write(f'- CN encodings: {", ".join(cn_encodings)}\n')
        f.write(f'- Target PCs: {", ".join(target_pcs)}\n\n')
        f.write(
            f'- Custom stress holdout train: {", ".join([str(x) for x in args.stress_train_experiments])}\n'
        )
        f.write(
            f'- Custom stress holdout test: {", ".join([str(x) for x in args.stress_test_experiments])}\n\n'
        )

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
    parser.add_argument('--cluster-size-endo-mode', choices=['keep', 'drop', 'log1p'], default='keep',
                        help='Ablation mode for Cluster_Size_Endo before normalization/PCA.')
    parser.add_argument('--max-components', type=int, default=5,
                        help='Upper bound for frozen PCA components.')
    parser.add_argument('--target-pcs', type=int, default=3,
                        help='Number of leading PCs to predict.')

    parser.add_argument('--cn-map', type=str,
                        default='exp1:4.5,exp2_high_cn:9,exp2_low_cn:4,exp3:9',
                        help='CN lambda map, format exp:lambda,exp:lambda')
    parser.add_argument('--cn-cells', type=int, default=100,
                        help='Number of virtual cells per organoid for Poisson CN sampling.')
    parser.add_argument('--cn-encodings', nargs='+', default=['lambda', 'sample_mean', 'summary'],
                        choices=['all', 'lambda', 'sample_mean', 'summary'],
                        help='CN input encoding ablation set.')

    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--poly-alpha', type=float, default=1.0)
    parser.add_argument('--poly-mode', choices=['selected', 'full'], default='selected',
                        help='Poly basis: selected terms (agreed biology-driven basis) or full sklearn PolynomialFeatures.')
    parser.add_argument('--poly-degree', type=int, default=3,
                        help='Polynomial degree when --poly-mode full (ignored for selected mode).')
    parser.add_argument('--gam-lam', type=float, default=1.0,
                        help='Default pyGAM smoothing lambda when tuning is disabled.')
    parser.add_argument('--gam-tune-lam', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable nested inner-CV tuning over --gam-lam-grid.')
    parser.add_argument('--gam-lam-grid', nargs='+', type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0],
                        help='Candidate lambda values for pyGAM tuning.')
    parser.add_argument('--gam-lam-tune-splits', nargs='+', default=['LOEX'],
                        choices=['RANDOM', 'LOR', 'LOEX'],
                        help='Outer split types where GAM lambda tuning is applied.')
    parser.add_argument('--gam-inner-splits', type=int, default=3,
                        help='Number of inner random splits for GAM lambda tuning.')
    parser.add_argument('--gam-inner-test-size', type=float, default=0.2,
                        help='Inner random split test fraction for GAM lambda tuning.')
    parser.add_argument('--gam-alpha', type=float, default=1.0)
    parser.add_argument('--gam-knots', type=int, default=5)

    parser.add_argument('--eval-lor', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate leave-one-(experiment,replicate)-out splits.')
    parser.add_argument('--eval-loex', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate leave-one-experiment-out splits.')
    parser.add_argument('--eval-random', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate random sample-level CV splits (optimistic baseline).')
    parser.add_argument('--random-splits', type=int, default=10,
                        help='Number of random CV splits when --eval-random is enabled.')
    parser.add_argument('--random-test-size', type=float, default=0.2,
                        help='Fraction of samples in random test split.')
    parser.add_argument('--stress-train-experiments', nargs='+', default=['exp1', 'exp2_high_cn'],
                        help='Custom stress-test training experiments.')
    parser.add_argument('--stress-test-experiments', nargs='+', default=['exp2_low_cn'],
                        help='Custom stress-test held-out test experiments.')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run(args)
