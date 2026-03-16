import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from scipy.stats import kruskal
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# DATASET MAPPING
# ==============================================================================
DATASET_MAP = {
    'exp1': 'GATA-HA_Rep1-3_Ex1',
    'exp2_high_cn': 'GATA-HA-Ex2_high_copy_num',
    'exp2_low_cn': 'GATA-HA-Ex2_low_copy_num',
    'exp3': 'GATA-HA-BMP4+Wnt5a_Ex3'
}

# Analysis parameters (consistent with all other scripts)
EPSILON = 30.0
MIN_SAMPLES = 20
NMS_RADIUS = 100.0
PROXIMITY_THRESHOLD = 30.0

MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'P', 'X']

# Fixed marker per experiment (shape encodes experiment identity)
EXP_MARKERS = {
    'exp1': 'o',
    'exp2_low_cn': '^',
    'exp2_high_cn': 'x',
    'exp3': 'D',
}

DISPLAY_NAMES = {
    'NMS_Endo': 'NMS (Endo)', 'NMS_Meso': 'NMS (Meso)',
    'Total_Cells': 'Total Cells (max-norm)', 'Pct_Endo': 'Endo Fraction', 'Pct_Meso': 'Meso Fraction',
    'Radial_Mean_Endo': 'Radial Mean (Endo)', 'Radial_Std_Endo': 'Radial Std (Endo)',
    'Radial_Mean_Meso': 'Radial Mean (Meso)', 'Radial_Std_Meso': 'Radial Std (Meso)',
    'Cluster_Count_Endo': 'Cluster Count (Endo)', 'Cluster_Count_Meso': 'Cluster Count (Meso)',
    'Cluster_Size_Endo': 'Cluster Size (Endo)', 'Cluster_Size_Meso': 'Cluster Size (Meso)',
    'Intra_Distance_Endo': 'Intra Dist (Endo)', 'Intra_Distance_Meso': 'Intra Dist (Meso)',
    'Inter_Distance': 'Inter Dist (E-M)', 'Adjacency_Pct': 'Adjacency %'
}

# ==============================================================================
# HELPER: FILTER TO FIRST N ORGANOIDS PER REPLICATE PER CONDITION
# ==============================================================================
def filter_first_n_organoids(file_list, n_limit=3):
    if n_limit is None or n_limit <= 0:
        return sorted(file_list)

    grouped = {}
    for fpath in file_list:
        fname = os.path.basename(fpath)
        replicate = os.path.basename(os.path.dirname(fpath))
        dox_match = re.search(r'(\d+)dox', fname)
        if not dox_match:
            continue
        dox = int(dox_match.group(1))
        org_match = re.search(r'_(\d+)\.csv$', fname)
        if not org_match:
            continue
        org_num = int(org_match.group(1))
        cond_match = re.search(r"\+(.+?)_", fname)
        condition = cond_match.group(1).upper() if cond_match else "BASAL"

        key = (replicate, dox, condition)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((org_num, fpath))

    filtered = []
    for key, files in grouped.items():
        files_sorted = sorted(files, key=lambda x: x[0])
        filtered.extend([fpath for _, fpath in files_sorted[:n_limit]])
    return sorted(filtered)

# ==============================================================================
# METRIC: NORMALIZED MIXING SCORE (NMS)
# ==============================================================================
def compute_nms(coords_all, types_all, target_id, radius):
    target_mask = types_all == target_id
    n_target = int(np.sum(target_mask))
    n_foreign = len(types_all) - n_target

    if n_target < 10 or n_foreign < 10:
        return np.nan

    tree = cKDTree(coords_all)
    target_indices = np.where(target_mask)[0]
    coords_target = coords_all[target_indices]

    neighbors_list = tree.query_ball_point(coords_target, r=radius)

    nms_values = []
    for _, neighbors in enumerate(neighbors_list):
        if len(neighbors) < 2:
            continue
        neighbor_types = types_all[neighbors]
        n_self = int(np.sum(neighbor_types == target_id)) - 1
        n_foreign_local = int(np.sum(neighbor_types != target_id))

        if n_self <= 0:
            continue

        nms = (n_foreign_local / n_self) / (n_foreign / n_target)
        nms_values.append(nms)

    return np.mean(nms_values) if nms_values else np.nan

# ==============================================================================
# METRIC: DBSCAN CLUSTERING + DERIVED METRICS
# ==============================================================================
def compute_cluster_metrics(coords, eps, min_samples):
    if len(coords) < min_samples:
        return [], np.nan, np.nan

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    unique_labels = [l for l in np.unique(labels) if l != -1]

    clusters = [coords[labels == l] for l in unique_labels]
    count = len(clusters)
    size = np.mean([len(c) for c in clusters]) if clusters else np.nan

    return clusters, count, size

# ==============================================================================
# METRIC: INTRA-CELL-TYPE INTER-CLUSTER DISTANCE (edge-to-edge)
# ==============================================================================
def compute_intra_distance(clusters):
    if len(clusters) < 2:
        return np.nan
    pair_dists = []
    for i in range(len(clusters)):
        tree_i = cKDTree(clusters[i])
        for j in range(i + 1, len(clusters)):
            dist, _ = tree_i.query(clusters[j], k=1)
            pair_dists.append(np.min(dist))
    return np.mean(pair_dists)

# ==============================================================================
# METRIC: INTER-CELL-TYPE CLUSTER DISTANCE (Endo-Meso edge-to-edge)
# ==============================================================================
def compute_inter_distance(clusters_a, clusters_b):
    if len(clusters_a) == 0 or len(clusters_b) == 0:
        return np.nan
    pair_dists = []
    for cl_a in clusters_a:
        tree = cKDTree(cl_a)
        for cl_b in clusters_b:
            dists, _ = tree.query(cl_b, k=1)
            pair_dists.append(np.min(dists))
    return np.mean(pair_dists)

# ==============================================================================
# METRIC: ADJACENCY % (minority clusters touching majority)
# ==============================================================================
def compute_adjacency(clusters_a, clusters_b, threshold):
    if len(clusters_a) == 0 or len(clusters_b) == 0:
        return np.nan

    if len(clusters_a) >= len(clusters_b):
        majority, minority = clusters_a, clusters_b
    else:
        majority, minority = clusters_b, clusters_a

    touching = 0
    for min_cl in minority:
        min_tree = cKDTree(min_cl)
        for maj_cl in majority:
            dists, _ = min_tree.query(maj_cl, k=1)
            if np.min(dists) <= threshold:
                touching += 1
                break

    return (touching / len(minority)) * 100.0

# ==============================================================================
# AGGREGATE: ALL METRICS FOR ONE ORGANOID
# ==============================================================================
def compute_all_metrics(df):
    target_col = next((c for c in df.columns if 'cell_type' in c.lower()), None)
    if not target_col:
        return None

    x_col = 'Global X' if 'Global X' in df.columns else 'X'
    y_col = 'Global Y' if 'Global Y' in df.columns else 'Y'
    z_col = 'Global Z' if 'Global Z' in df.columns else 'Z'

    coords_all = df[[x_col, y_col, z_col]].values
    types_all = df[target_col].values

    endo_mask = types_all == 2.0
    meso_mask = types_all == 3.0
    coords_endo = coords_all[endo_mask]
    coords_meso = coords_all[meso_mask]

    result = {}

    # 0. Cell composition features
    n_total = len(types_all)
    n_endo = int(np.sum(endo_mask))
    n_meso = int(np.sum(meso_mask))
    result['Total_Cells'] = n_total
    result['Pct_Endo'] = (n_endo / n_total) if n_total > 0 else np.nan
    result['Pct_Meso'] = (n_meso / n_total) if n_total > 0 else np.nan

    # 0b. Radial position features (distance from organoid center)
    if n_total > 0:
        center = np.mean(coords_all, axis=0)
        if len(coords_endo) > 0:
            endo_r = np.linalg.norm(coords_endo - center, axis=1)
            result['Radial_Mean_Endo'] = float(np.mean(endo_r))
            result['Radial_Std_Endo'] = float(np.std(endo_r))
        else:
            result['Radial_Mean_Endo'] = np.nan
            result['Radial_Std_Endo'] = np.nan

        if len(coords_meso) > 0:
            meso_r = np.linalg.norm(coords_meso - center, axis=1)
            result['Radial_Mean_Meso'] = float(np.mean(meso_r))
            result['Radial_Std_Meso'] = float(np.std(meso_r))
        else:
            result['Radial_Mean_Meso'] = np.nan
            result['Radial_Std_Meso'] = np.nan
    else:
        result['Radial_Mean_Endo'] = np.nan
        result['Radial_Std_Endo'] = np.nan
        result['Radial_Mean_Meso'] = np.nan
        result['Radial_Std_Meso'] = np.nan

    # 1-2. NMS
    result['NMS_Endo'] = compute_nms(coords_all, types_all, 2.0, NMS_RADIUS)
    result['NMS_Meso'] = compute_nms(coords_all, types_all, 3.0, NMS_RADIUS)

    # 3-6. DBSCAN clustering
    endo_clusters, endo_count, endo_size = compute_cluster_metrics(coords_endo, EPSILON, MIN_SAMPLES)
    meso_clusters, meso_count, meso_size = compute_cluster_metrics(coords_meso, EPSILON, MIN_SAMPLES)

    result['Cluster_Count_Endo'] = endo_count
    result['Cluster_Count_Meso'] = meso_count
    result['Cluster_Size_Endo'] = endo_size
    result['Cluster_Size_Meso'] = meso_size

    # 7-8. Intra-cell-type cluster distance
    result['Intra_Distance_Endo'] = compute_intra_distance(endo_clusters)
    result['Intra_Distance_Meso'] = compute_intra_distance(meso_clusters)

    # 9. Inter-cell-type cluster distance
    result['Inter_Distance'] = compute_inter_distance(endo_clusters, meso_clusters)

    # 10. Adjacency %
    result['Adjacency_Pct'] = compute_adjacency(endo_clusters, meso_clusters, PROXIMITY_THRESHOLD)

    return result

# ==============================================================================
# FEATURE SIGNIFICANCE: KRUSKAL-WALLIS + SIGNAL/NOISE DECOMPOSITION
# ==============================================================================
def compute_feature_significance(df_clean, valid_features, experiments):
    """
    Per experiment, per feature:
    - Kruskal-Wallis H-test across dox levels (between-dox signal)
    - Within-dox variance / total variance (noise proportion)
    """
    records = []
    for exp in experiments:
        exp_data = df_clean[df_clean['Experiment'] == exp]
        for feat in valid_features:
            groups = [g[feat].dropna().values
                      for _, g in exp_data.groupby('Dox_Concentration')]
            groups = [g for g in groups if len(g) >= 2]

            if len(groups) >= 2:
                H, p = kruskal(*groups)
            else:
                H, p = np.nan, np.nan

            within_var = exp_data.groupby('Dox_Concentration')[feat].var().mean()
            total_var = exp_data[feat].var()
            noise_ratio = within_var / total_var if total_var > 0 else np.nan

            records.append({
                'Experiment': exp,
                'Feature': feat,
                'KW_H_statistic': H,
                'KW_p_value': p,
                'Noise_Ratio': noise_ratio,
                'Signal_Ratio': 1 - noise_ratio if not np.isnan(noise_ratio) else np.nan
            })

    return pd.DataFrame(records)


def compute_pca_group_separation(df_scores, pc_cols):
    """
    Decompose each PC variance into between-group fractions.
    Ratios are SS_between / SS_total and lie in [0, 1] when variance exists.
    """
    records = []

    def _between_ratio(values, groups):
        values = np.asarray(values, dtype=float)
        groups = np.asarray(groups)
        mask = np.isfinite(values)
        values = values[mask]
        groups = groups[mask]
        if len(values) < 2:
            return np.nan, np.nan

        grand_mean = float(np.mean(values))
        total_ss = float(np.sum((values - grand_mean) ** 2))
        if total_ss <= 0:
            return 0.0, 0.0

        df_tmp = pd.DataFrame({'v': values, 'g': groups})
        grp = df_tmp.groupby('g')['v'].agg(['mean', 'count'])
        between_ss = float(np.sum(grp['count'] * (grp['mean'] - grand_mean) ** 2))
        return between_ss / total_ss, total_ss / (len(values) - 1)

    for pc in pc_cols:
        vals = df_scores[pc].values
        ratio_exp, total_var = _between_ratio(vals, df_scores['Experiment'].values)
        ratio_dox, _ = _between_ratio(vals, df_scores['Dox_Concentration'].values)
        exp_dox_key = (df_scores['Experiment'].astype(str) + "|" +
                       df_scores['Dox_Concentration'].astype(str))
        ratio_exp_dox, _ = _between_ratio(vals, exp_dox_key.values)

        records.append({
            'PC': pc,
            'Total_Variance': total_var,
            'Between_Experiment_Ratio': ratio_exp,
            'Between_Dox_Ratio': ratio_dox,
            'Between_ExperimentDox_Ratio': ratio_exp_dox,
            'Within_Experiment_Ratio': (1 - ratio_exp) if pd.notna(ratio_exp) else np.nan
        })

    return pd.DataFrame(records)


def compute_consecutive_dox_distances(df_scores, experiments):
    """
    For each experiment, compute Euclidean distance between consecutive dox-level
    centroids in PCA space:
      - 2D: (PC1, PC2)
      - 3D: (PC1, PC2, PC3) when PC3 is available
    """
    has_pc3 = all(c in df_scores.columns for c in ['PC1', 'PC2', 'PC3'])
    records = []

    for exp in experiments:
        exp_data = df_scores[df_scores['Experiment'] == exp].copy()
        if exp_data.empty:
            continue

        centroid_cols = ['PC1', 'PC2'] + (['PC3'] if has_pc3 else [])
        centroids = (exp_data.groupby('Dox_Concentration', as_index=False)[centroid_cols]
                     .mean()
                     .sort_values('Dox_Concentration'))
        if len(centroids) < 2:
            continue

        for i in range(len(centroids) - 1):
            row_a = centroids.iloc[i]
            row_b = centroids.iloc[i + 1]
            dox_from = int(row_a['Dox_Concentration'])
            dox_to = int(row_b['Dox_Concentration'])

            dist_2d = float(np.linalg.norm(
                row_b[['PC1', 'PC2']].to_numpy(dtype=float) -
                row_a[['PC1', 'PC2']].to_numpy(dtype=float)
            ))
            dist_3d = (float(np.linalg.norm(
                row_b[['PC1', 'PC2', 'PC3']].to_numpy(dtype=float) -
                row_a[['PC1', 'PC2', 'PC3']].to_numpy(dtype=float)
            )) if has_pc3 else np.nan)

            records.append({
                'Experiment': exp,
                'Step_Index': i + 1,
                'Dox_From': dox_from,
                'Dox_To': dox_to,
                'Dox_Transition': f"{dox_from}->{dox_to}",
                'Distance_PC1_PC2': dist_2d,
                'Distance_PC1_PC2_PC3': dist_3d
            })

    return pd.DataFrame(records)


def plot_consecutive_dox_distances(df_dist, experiments, output_dir, label, mode_tag):
    if df_dist.empty:
        print(f"  [{mode_tag}] No consecutive dox transitions available for distance plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax2d, ax3d) = plt.subplots(1, 2, figsize=(16, 6), sharex=False)

    exp_present = [e for e in experiments if e in set(df_dist['Experiment'].unique())]
    cmap_exp = plt.cm.tab10
    exp_colors = {exp: cmap_exp(i % 10) for i, exp in enumerate(exp_present)}
    transition_pairs = sorted(
        {(int(r['Dox_From']), int(r['Dox_To'])) for _, r in df_dist[['Dox_From', 'Dox_To']].iterrows()},
        key=lambda x: (x[0], x[1])
    )
    transition_labels = [f"{a}->{b}" for a, b in transition_pairs]
    transition_index = {lab: i for i, lab in enumerate(transition_labels)}

    for exp in exp_present:
        exp_data = df_dist[df_dist['Experiment'] == exp].sort_values(['Dox_From', 'Dox_To'])
        if exp_data.empty:
            continue
        x_vals = exp_data['Dox_Transition'].map(transition_index).values
        exp_marker = EXP_MARKERS.get(exp, 'o')

        ax2d.plot(x_vals, exp_data['Distance_PC1_PC2'].values,
                  marker=exp_marker, markersize=7, linewidth=2.0,
                  color=exp_colors[exp], alpha=0.9, label=exp)

        if exp_data['Distance_PC1_PC2_PC3'].notna().any():
            ax3d.plot(x_vals, exp_data['Distance_PC1_PC2_PC3'].values,
                      marker=exp_marker, markersize=7, linewidth=2.0,
                      color=exp_colors[exp], alpha=0.9, label=exp)

    ax2d.set_xticks(np.arange(len(transition_labels)))
    ax2d.set_xticklabels(transition_labels, rotation=35, ha='right')

    ax2d.set_title("Consecutive Dox Distance in 2D PCA", fontweight='bold')
    ax2d.set_xlabel("Dox Transition (ng/mL)")
    ax2d.set_ylabel("Euclidean Distance")
    ax2d.legend(title="Experiment", fontsize=9, title_fontsize=10, loc='best')

    if df_dist['Distance_PC1_PC2_PC3'].notna().any():
        ax3d.set_xticks(np.arange(len(transition_labels)))
        ax3d.set_xticklabels(transition_labels, rotation=35, ha='right')
        ax3d.set_title("Consecutive Dox Distance in 3D PCA", fontweight='bold')
        ax3d.set_xlabel("Dox Transition (ng/mL)")
        ax3d.set_ylabel("Euclidean Distance")
        ax3d.legend(title="Experiment", fontsize=9, title_fontsize=10, loc='best')
    else:
        ax3d.axis('off')
        ax3d.text(0.5, 0.5, "PC3 not available\n(<3 PCA components)",
                  ha='center', va='center', fontsize=12)

    plt.suptitle(f"Across-Dox Step Distances in PCA Space [{mode_tag}]",
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_PCA_Consecutive_Dox_Distances.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Consecutive dox distance plot saved: {output_file}")
    plt.close()


def residualize_by_replicate_within_dox(df_features, feature_cols):
    """
    Residualize replicate effects while preserving dox-level means.
    For each feature x:
        x_adj = x - mean(x | experiment,dox,replicate) + mean(x | experiment,dox)
    """
    df_adj = df_features.copy()
    rep_group = ['Experiment', 'Dox_Concentration', 'Replicate']
    dox_group = ['Experiment', 'Dox_Concentration']

    for feat in feature_cols:
        # Keep residualized features float-typed to avoid integer assignment warnings.
        if feat in df_adj.columns:
            df_adj[feat] = pd.to_numeric(df_adj[feat], errors='coerce').astype(float)
        rep_mean = df_adj.groupby(rep_group)[feat].transform('mean')
        dox_mean = df_adj.groupby(dox_group)[feat].transform('mean')
        mask = df_adj[feat].notna() & rep_mean.notna() & dox_mean.notna()
        df_adj.loc[mask, feat] = df_adj.loc[mask, feat] - rep_mean[mask] + dox_mean[mask]

    return df_adj


def impute_missing_features(df_features, feature_cols):
    """
    Two-step imputation:
    1) median within (Experiment, Dox_Concentration)
    2) global feature median fallback
    """
    df_imp = df_features.copy()
    missing_before = df_imp[feature_cols].isna().sum()

    for feat in feature_cols:
        grouped_median = df_imp.groupby(['Experiment', 'Dox_Concentration'])[feat].transform('median')
        df_imp[feat] = df_imp[feat].fillna(grouped_median)
        df_imp[feat] = df_imp[feat].fillna(df_imp[feat].median())

    missing_after = df_imp[feature_cols].isna().sum()
    return df_imp, missing_before, missing_after


def normalize_composition_features(df_features, clip_fractions=True):
    """
    Normalize composition/size features before PCA/significance:
    - Total_Cells -> divide by max across all samples in current run-mode.
    - Pct_Endo/Pct_Meso expected as fractions (0..1); clip for safety.
    """
    df_norm = df_features.copy()

    if 'Total_Cells' in df_norm.columns:
        max_cells = df_norm['Total_Cells'].max(skipna=True)
        if pd.notna(max_cells) and max_cells > 0:
            df_norm['Total_Cells'] = df_norm['Total_Cells'] / max_cells

    if clip_fractions:
        for feat in ('Pct_Endo', 'Pct_Meso'):
            if feat in df_norm.columns:
                df_norm[feat] = df_norm[feat].clip(lower=0.0, upper=1.0)

    return df_norm


def build_pca_fit_matrix(df_pca, valid_features, pca_fit_basis):
    """
    Build the matrix used to fit scaler+PCA.
    Projection is always done for all organoids in df_pca.
    """
    if pca_fit_basis == 'exp_dox_centroids':
        group_cols = ['Experiment', 'Dox_Concentration']
        df_fit = (df_pca.groupby(group_cols, as_index=False)[valid_features]
                  .mean())
    elif pca_fit_basis == 'exp_centroids':
        group_cols = ['Experiment']
        df_fit = (df_pca.groupby(group_cols, as_index=False)[valid_features]
                  .mean())
    elif pca_fit_basis == 'all_organoids':
        df_fit = df_pca[valid_features].copy()
    else:
        raise ValueError(f"Unsupported PCA fit basis: {pca_fit_basis}")

    return df_fit[valid_features].values, len(df_fit)

# ==============================================================================
# VISUALIZATION: MAIN PCA FIGURE
# ==============================================================================
def plot_pca_pc1_pc2_per_experiment(df_scores, df_loadings, pca, experiments, valid_features,
                    n_components, output_dir, label, mode_tag):
    if 'PC2' not in df_scores.columns:
        print(f"\n  [{mode_tag}] Skipping PC1-PC2 figure (fewer than 2 PCs).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    n_exp = len(experiments)
    fig_width = max(9 * n_exp, 18)

    fig = plt.figure(figsize=(fig_width, 18))

    outer_gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.35)
    gs_top = GridSpecFromSubplotSpec(1, n_exp, subplot_spec=outer_gs[0], wspace=0.3)
    gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1],
                                        width_ratios=[2, 1], wspace=0.4)

    # Global dox color scale (union of all dox levels across experiments)
    all_dox = sorted(df_scores['Dox_Concentration'].unique())
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(len(all_dox) - 1, 1))
    dox_colors = {dox: cmap(norm(i)) for i, dox in enumerate(all_dox)}

    # Shared axis limits across experiment panels
    pc1_min, pc1_max = df_scores['PC1'].min(), df_scores['PC1'].max()
    pc2_min, pc2_max = df_scores['PC2'].min(), df_scores['PC2'].max()
    margin = 0.15 * max(pc1_max - pc1_min, pc2_max - pc2_min, 1)

    # --- Top row: PCA scatter per experiment ---
    for ei, exp in enumerate(experiments):
        ax = fig.add_subplot(gs_top[ei])
        exp_data = df_scores[df_scores['Experiment'] == exp]

        exp_dox = sorted(exp_data['Dox_Concentration'].unique())
        exp_marker = EXP_MARKERS.get(exp, 'o')

        # Individual organoids (color = dox, shape = experiment)
        for dox in exp_dox:
            subset = exp_data[exp_data['Dox_Concentration'] == dox]
            if subset.empty:
                continue
            ax.scatter(subset['PC1'], subset['PC2'],
                       c=[dox_colors[dox]], marker=exp_marker,
                       s=80, alpha=0.5, edgecolors='white',
                       linewidths=0.5, zorder=3)

        # Mean trajectory (connect centroids)
        centroids = exp_data.groupby('Dox_Concentration')[['PC1', 'PC2']].mean()
        centroids = centroids.reindex(exp_dox)
        ax.plot(centroids['PC1'].values, centroids['PC2'].values,
                'k-', linewidth=2.5, alpha=0.7, zorder=5)
        ax.scatter(centroids['PC1'].values, centroids['PC2'].values,
                   c=[dox_colors[d] for d in exp_dox],
                   s=220, edgecolors='black', linewidths=2, zorder=6, marker='D')

        # Annotate centroids with dox labels
        for dox in exp_dox:
            if dox in centroids.index:
                ax.annotate(f'{dox}', (centroids.loc[dox, 'PC1'], centroids.loc[dox, 'PC2']),
                            textcoords="offset points", xytext=(10, 8),
                            fontsize=8, fontweight='bold')

        ax.set_xlim(pc1_min - margin, pc1_max + margin)
        ax.set_ylim(pc2_min - margin, pc2_max + margin)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"{exp}", fontweight='bold')

        # Legend: dox colors
        dox_handles = [mlines.Line2D([], [], marker=exp_marker, color='w',
                                      markerfacecolor=dox_colors[d], markeredgecolor='gray',
                                      markersize=8,
                                      label=f'{d}') for d in exp_dox]
        ax.legend(handles=dox_handles, title="Dox (ng/mL)",
                  loc='upper right', fontsize=6, title_fontsize=7)

    # --- Bottom left: Loadings heatmap ---
    ax_load = fig.add_subplot(gs_bottom[0])
    n_show = min(4, n_components)
    loadings_show = df_loadings.iloc[:, :n_show]
    loadings_display = loadings_show.rename(index=DISPLAY_NAMES)

    sns.heatmap(loadings_display, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax_load, cbar_kws={'shrink': 0.8, 'label': 'Loading'},
                linewidths=0.5, linecolor='white')
    ax_load.set_title("PC Loadings", fontweight='bold')
    ax_load.set_ylabel("")

    # --- Bottom right: Scree plot ---
    ax_scree = fig.add_subplot(gs_bottom[1])
    pc_labels = [f'PC{i+1}' for i in range(n_components)]
    ax_scree.bar(pc_labels, pca.explained_variance_ratio_ * 100,
                  color='steelblue', alpha=0.7, edgecolor='navy')
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    ax_scree.plot(pc_labels, cumulative, 'ro-', linewidth=2, markersize=8)
    ax_scree.set_ylabel("Variance Explained (%)")
    ax_scree.set_title("Scree Plot", fontweight='bold')
    ax_scree.set_ylim(0, 105)

    for i, cum in enumerate(cumulative):
        ax_scree.annotate(f'{cum:.0f}%', (i, cum), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, color='red',
                           fontweight='bold')

    title_text = "Spatial State Trajectory Analysis (PCA)"
    if len(experiments) > 1:
        title_text += " — Shared PC Axes"
    title_text += f" [{mode_tag}]"
    plt.suptitle(title_text, fontsize=22, fontweight='bold', y=0.98)

    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Spatial_State_Trajectory_PCA.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  PCA figure saved: {output_file}")
    plt.close()


def plot_pca_pc3_projections_2d(df_scores, pca, experiments, output_dir, label, mode_tag):
    if 'PC3' not in df_scores.columns:
        print(f"  [{mode_tag}] Skipping PC3 figure (fewer than 3 PCs).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    n_exp = len(experiments)
    fig = plt.figure(figsize=(14, max(6, 5 * n_exp)))
    gs = GridSpec(n_exp, 2, figure=fig, wspace=0.30, hspace=0.35)

    all_dox = sorted(df_scores['Dox_Concentration'].unique())
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(len(all_dox) - 1, 1))
    dox_colors = {dox: cmap(norm(i)) for i, dox in enumerate(all_dox)}

    for ei, exp in enumerate(experiments):
        exp_data = df_scores[df_scores['Experiment'] == exp]
        exp_dox = sorted(exp_data['Dox_Concentration'].unique())
        centroids = exp_data.groupby('Dox_Concentration')[['PC1', 'PC2', 'PC3']].mean().reindex(exp_dox)
        exp_marker = EXP_MARKERS.get(exp, 'o')

        # --- PC1 vs PC3 ---
        ax13 = fig.add_subplot(gs[ei, 0])
        for dox in exp_dox:
            subset = exp_data[exp_data['Dox_Concentration'] == dox]
            if subset.empty:
                continue
            ax13.scatter(subset['PC1'], subset['PC3'],
                         c=[dox_colors[dox]], marker=exp_marker,
                         s=70, alpha=0.45, edgecolors='white', linewidths=0.5)
        ax13.plot(centroids['PC1'].values, centroids['PC3'].values, 'k-', linewidth=2.2, alpha=0.75)
        ax13.scatter(centroids['PC1'].values, centroids['PC3'].values,
                     c=[dox_colors[d] for d in exp_dox], marker='D', s=170,
                     edgecolors='black', linewidths=1.6)
        ax13.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax13.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax13.set_title(f"{exp}: PC1-PC3", fontweight='bold')

        # --- PC2 vs PC3 ---
        ax23 = fig.add_subplot(gs[ei, 1])
        for dox in exp_dox:
            subset = exp_data[exp_data['Dox_Concentration'] == dox]
            if subset.empty:
                continue
            ax23.scatter(subset['PC2'], subset['PC3'],
                         c=[dox_colors[dox]], marker=exp_marker,
                         s=70, alpha=0.45, edgecolors='white', linewidths=0.5)
        ax23.plot(centroids['PC2'].values, centroids['PC3'].values, 'k-', linewidth=2.2, alpha=0.75)
        ax23.scatter(centroids['PC2'].values, centroids['PC3'].values,
                     c=[dox_colors[d] for d in exp_dox], marker='D', s=170,
                     edgecolors='black', linewidths=1.6)
        ax23.set_xlabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax23.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax23.set_title(f"{exp}: PC2-PC3", fontweight='bold')

    plt.suptitle(f"PC3 Projection Views [{mode_tag}]", fontsize=18, fontweight='bold')
    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Spatial_State_Trajectory_PCA_PC3_2D.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  PC3 projection figure saved: {output_file}")
    plt.close()


def plot_pca_3d_scatter(df_scores, experiments, output_dir, label, mode_tag):
    if 'PC3' not in df_scores.columns:
        print(f"  [{mode_tag}] Skipping 3D trajectory figure (fewer than 3 PCs).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    n_exp = len(experiments)
    fig = plt.figure(figsize=(max(7 * n_exp, 8), 7))
    gs = GridSpec(1, n_exp, figure=fig, wspace=0.28)

    all_dox = sorted(df_scores['Dox_Concentration'].unique())
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(len(all_dox) - 1, 1))
    dox_colors = {dox: cmap(norm(i)) for i, dox in enumerate(all_dox)}

    for ei, exp in enumerate(experiments):
        exp_data = df_scores[df_scores['Experiment'] == exp]
        exp_dox = sorted(exp_data['Dox_Concentration'].unique())
        centroids = exp_data.groupby('Dox_Concentration')[['PC1', 'PC2', 'PC3']].mean().reindex(exp_dox)
        exp_marker = EXP_MARKERS.get(exp, 'o')

        ax3d = fig.add_subplot(gs[0, ei], projection='3d')
        for dox in exp_dox:
            subset = exp_data[exp_data['Dox_Concentration'] == dox]
            if subset.empty:
                continue
            ax3d.scatter(subset['PC1'], subset['PC3'], subset['PC2'],
                         c=[dox_colors[dox]], marker=exp_marker,
                         s=28, alpha=0.45, depthshade=True)

        ax3d.plot(centroids['PC1'].values, centroids['PC3'].values, centroids['PC2'].values,
                  color='black', linewidth=2.3, alpha=0.85)
        ax3d.scatter(centroids['PC1'].values, centroids['PC3'].values, centroids['PC2'].values,
                     c=[dox_colors[d] for d in exp_dox], marker='D', s=90,
                     edgecolors='black', linewidths=1.2, depthshade=False)
        ax3d.set_xlabel("PC1")
        ax3d.set_ylabel("PC3")
        ax3d.set_zlabel("PC2")
        ax3d.set_title(f"{exp}: 3D Trajectory", fontweight='bold')
        ax3d.view_init(elev=22, azim=-58)

    plt.suptitle(f"3D Trajectory Only [{mode_tag}]", fontsize=18, fontweight='bold')
    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Spatial_State_Trajectory_PCA_3D_only.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  3D trajectory figure saved: {output_file}")
    plt.close()


def plot_pca_pc1_pc2_cross_experiment(df_scores, df_loadings, pca, experiments, valid_features,
                                  n_components, output_dir, label, mode_tag):
    if 'PC2' not in df_scores.columns:
        print(f"\n  [{mode_tag}] Skipping PC1-PC2 figure (fewer than 2 PCs).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(18, 16))
    outer_gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.35)
    ax_top = fig.add_subplot(outer_gs[0])
    gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1],
                                        width_ratios=[2, 1], wspace=0.4)

    exp_present = [e for e in experiments if e in set(df_scores['Experiment'].unique())]
    cmap_exp = plt.cm.tab10
    exp_colors = {exp: cmap_exp(i % 10) for i, exp in enumerate(exp_present)}

    pc1_min, pc1_max = df_scores['PC1'].min(), df_scores['PC1'].max()
    pc2_min, pc2_max = df_scores['PC2'].min(), df_scores['PC2'].max()
    margin = 0.15 * max(pc1_max - pc1_min, pc2_max - pc2_min, 1)

    for exp in exp_present:
        exp_data = df_scores[df_scores['Experiment'] == exp]
        exp_marker = EXP_MARKERS.get(exp, 'o')
        subset = exp_data
        if subset.empty:
            continue
        ax_top.scatter(subset['PC1'], subset['PC2'],
                       c=[exp_colors[exp]], marker=exp_marker,
                       s=70, alpha=0.45, edgecolors='white', linewidths=0.5, zorder=3)

    centroids = df_scores.groupby('Experiment')[['PC1', 'PC2']].mean()
    centroids = centroids.reindex(exp_present)
    centroids = centroids.dropna()
    if len(centroids) >= 2:
        ax_top.plot(centroids['PC1'].values, centroids['PC2'].values,
                    'k-', linewidth=2.6, alpha=0.75, zorder=5)

    ax_top.scatter(centroids['PC1'].values, centroids['PC2'].values,
                   c=[exp_colors[e] for e in centroids.index],
                   s=240, edgecolors='black', linewidths=2, zorder=6, marker='D')

    for exp in centroids.index:
        ax_top.annotate(exp, (centroids.loc[exp, 'PC1'], centroids.loc[exp, 'PC2']),
                        textcoords="offset points", xytext=(10, 8), fontsize=9, fontweight='bold')

    ax_top.set_xlim(pc1_min - margin, pc1_max + margin)
    ax_top.set_ylim(pc2_min - margin, pc2_max + margin)
    ax_top.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_top.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_top.set_title("Cross-Experiment Trajectory (ordered by --experiment)", fontweight='bold')

    exp_handles = [mlines.Line2D([], [], marker=EXP_MARKERS.get(e, 'o'), color='w',
                                 markerfacecolor=exp_colors[e], markeredgecolor=exp_colors[e],
                                 markersize=8, label=e)
                   for e in exp_present]
    ax_top.legend(handles=exp_handles, title="Experiment", loc='upper right',
                  fontsize=7, title_fontsize=8)

    ax_load = fig.add_subplot(gs_bottom[0])
    n_show = min(4, n_components)
    loadings_show = df_loadings.iloc[:, :n_show]
    loadings_display = loadings_show.rename(index=DISPLAY_NAMES)
    sns.heatmap(loadings_display, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax_load, cbar_kws={'shrink': 0.8, 'label': 'Loading'},
                linewidths=0.5, linecolor='white')
    ax_load.set_title("PC Loadings", fontweight='bold')
    ax_load.set_ylabel("")

    ax_scree = fig.add_subplot(gs_bottom[1])
    pc_labels = [f'PC{i+1}' for i in range(n_components)]
    ax_scree.bar(pc_labels, pca.explained_variance_ratio_ * 100,
                 color='steelblue', alpha=0.7, edgecolor='navy')
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    ax_scree.plot(pc_labels, cumulative, 'ro-', linewidth=2, markersize=8)
    ax_scree.set_ylabel("Variance Explained (%)")
    ax_scree.set_title("Scree Plot", fontweight='bold')
    ax_scree.set_ylim(0, 105)
    for i, cum in enumerate(cumulative):
        ax_scree.annotate(f'{cum:.0f}%', (i, cum), textcoords="offset points",
                          xytext=(0, 10), ha='center', fontsize=9, color='red',
                          fontweight='bold')

    title_text = "Spatial State Trajectory Analysis (PCA) — Grouped by Experiment"
    title_text += f" [{mode_tag}]"
    plt.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98)

    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Spatial_State_Trajectory_PCA_by_experiment.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Cross-experiment PCA figure saved: {output_file}")
    plt.close()


def plot_pca_pc3_projections_2d_cross_experiment(df_scores, pca, experiments, output_dir, label, mode_tag):
    if 'PC3' not in df_scores.columns:
        print(f"  [{mode_tag}] Skipping PC3 figure (fewer than 3 PCs).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30, hspace=0.35)

    exp_present = [e for e in experiments if e in set(df_scores['Experiment'].unique())]
    cmap_exp = plt.cm.tab10
    exp_colors = {exp: cmap_exp(i % 10) for i, exp in enumerate(exp_present)}

    centroids = df_scores.groupby('Experiment')[['PC1', 'PC2', 'PC3']].mean().reindex(exp_present).dropna()

    ax13 = fig.add_subplot(gs[0, 0])
    ax23 = fig.add_subplot(gs[0, 1])
    for exp in exp_present:
        exp_data = df_scores[df_scores['Experiment'] == exp]
        exp_marker = EXP_MARKERS.get(exp, 'o')
        if exp_data.empty:
            continue
        ax13.scatter(exp_data['PC1'], exp_data['PC3'],
                     c=[exp_colors[exp]], marker=exp_marker,
                     s=65, alpha=0.45, edgecolors='white', linewidths=0.5)
        ax23.scatter(exp_data['PC2'], exp_data['PC3'],
                     c=[exp_colors[exp]], marker=exp_marker,
                     s=65, alpha=0.45, edgecolors='white', linewidths=0.5)

    if len(centroids) >= 2:
        ax13.plot(centroids['PC1'].values, centroids['PC3'].values, 'k-', linewidth=2.3, alpha=0.8)
        ax23.plot(centroids['PC2'].values, centroids['PC3'].values, 'k-', linewidth=2.3, alpha=0.8)
    ax13.scatter(centroids['PC1'].values, centroids['PC3'].values,
                 c=[exp_colors[e] for e in centroids.index], marker='D', s=190,
                 edgecolors='black', linewidths=1.5)
    ax23.scatter(centroids['PC2'].values, centroids['PC3'].values,
                 c=[exp_colors[e] for e in centroids.index], marker='D', s=190,
                 edgecolors='black', linewidths=1.5)

    ax13.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax13.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax13.set_title("PC1-PC3 (by experiment)", fontweight='bold')
    ax23.set_xlabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax23.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax23.set_title("PC2-PC3 (by experiment)", fontweight='bold')

    plt.suptitle(f"PC3 Projection Views — Grouped by Experiment [{mode_tag}]",
                 fontsize=18, fontweight='bold')
    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Spatial_State_Trajectory_PCA_PC3_2D_by_experiment.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Cross-experiment PC3 projection figure saved: {output_file}")
    plt.close()


def plot_pca_3d_scatter_cross_experiment(df_scores, experiments, output_dir, label, mode_tag):
    if 'PC3' not in df_scores.columns:
        print(f"  [{mode_tag}] Skipping 3D trajectory figure (fewer than 3 PCs).")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')

    exp_present = [e for e in experiments if e in set(df_scores['Experiment'].unique())]
    cmap_exp = plt.cm.tab10
    exp_colors = {exp: cmap_exp(i % 10) for i, exp in enumerate(exp_present)}

    centroids = df_scores.groupby('Experiment')[['PC1', 'PC2', 'PC3']].mean().reindex(exp_present).dropna()

    for exp in exp_present:
        exp_data = df_scores[df_scores['Experiment'] == exp]
        exp_marker = EXP_MARKERS.get(exp, 'o')
        if exp_data.empty:
            continue
        ax3d.scatter(exp_data['PC1'], exp_data['PC3'], exp_data['PC2'],
                     c=[exp_colors[exp]], marker=exp_marker,
                     s=26, alpha=0.45, depthshade=True)

    if len(centroids) >= 2:
        ax3d.plot(centroids['PC1'].values, centroids['PC3'].values, centroids['PC2'].values,
                  color='black', linewidth=2.4, alpha=0.85)
    ax3d.scatter(centroids['PC1'].values, centroids['PC3'].values, centroids['PC2'].values,
                 c=[exp_colors[e] for e in centroids.index], marker='D', s=110,
                 edgecolors='black', linewidths=1.2, depthshade=False)

    for exp in centroids.index:
        ax3d.text(centroids.loc[exp, 'PC1'], centroids.loc[exp, 'PC3'], centroids.loc[exp, 'PC2'],
                  exp, fontsize=8)

    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC3")
    ax3d.set_zlabel("PC2")
    ax3d.set_title("3D Trajectory (grouped by experiment)", fontweight='bold')
    ax3d.view_init(elev=22, azim=-58)

    plt.suptitle(f"3D Trajectory Only — Grouped by Experiment [{mode_tag}]",
                 fontsize=18, fontweight='bold')
    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Spatial_State_Trajectory_PCA_3D_only_by_experiment.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Cross-experiment 3D trajectory figure saved: {output_file}")
    plt.close()

# ==============================================================================
# VISUALIZATION: FEATURE SIGNIFICANCE FIGURE
# ==============================================================================
def plot_significance_figure(df_sig, experiments, output_dir, label, mode_tag):
    sns.set_theme(style="whitegrid", context="talk")
    df_sig = df_sig.copy()
    df_sig['Feature_Display'] = df_sig['Feature'].map(DISPLAY_NAMES).fillna(df_sig['Feature'])

    # Sort features by mean signal ratio (most informative first at top)
    feat_order = (df_sig.groupby('Feature_Display')['Signal_Ratio']
                  .mean().sort_values(ascending=True).index.tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # --- Panel A: Signal Ratio heatmap ---
    pivot_signal = df_sig.pivot(index='Feature_Display', columns='Experiment',
                                values='Signal_Ratio')
    pivot_signal = pivot_signal.reindex(index=feat_order, columns=experiments)

    sns.heatmap(pivot_signal, annot=True, fmt='.2f', cmap='YlGn', vmin=0, vmax=1,
                ax=ax1, linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.7, 'label': 'Signal Ratio'})
    ax1.set_title("Signal Ratio\n(between-dox / total variance)", fontweight='bold')
    ax1.set_ylabel("")

    # --- Panel B: -log10(p) heatmap with significance stars ---
    pivot_p = df_sig.pivot(index='Feature_Display', columns='Experiment',
                           values='KW_p_value')
    pivot_p = pivot_p.reindex(index=feat_order, columns=experiments)
    pivot_neglogp = -np.log10(pivot_p.clip(lower=1e-10))

    # Build annotation array with significance stars
    annot_arr = np.empty(pivot_p.shape, dtype=object)
    for i in range(pivot_p.shape[0]):
        for j in range(pivot_p.shape[1]):
            p = pivot_p.iloc[i, j]
            if pd.isna(p):
                annot_arr[i, j] = 'na'
            elif p < 0.001:
                annot_arr[i, j] = '***'
            elif p < 0.01:
                annot_arr[i, j] = '**'
            elif p < 0.05:
                annot_arr[i, j] = '*'
            else:
                annot_arr[i, j] = 'ns'

    sns.heatmap(pivot_neglogp, annot=annot_arr, fmt='', cmap='YlOrRd', vmin=0,
                ax=ax2, linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.7, 'label': r'$-\log_{10}(p)$'})
    ax2.set_title("Kruskal-Wallis Significance\n(across dox levels)", fontweight='bold')
    ax2.set_ylabel("")

    plt.suptitle(f"Feature Significance Analysis [{mode_tag}]", fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{label}_{mode_tag}_Feature_Significance.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Significance figure saved: {output_file}")
    plt.close()

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_pca_mode(df_features, valid_features, meta_cols, experiments, output_dir, label,
                 mode_tag, trajectory_group_by='dox', pca_fit_basis='auto'):
    if mode_tag == 'raw':
        df_mode = normalize_composition_features(df_features, clip_fractions=True)
    elif mode_tag == 'residualized':
        # Normalize first, then batch-correct on normalized features.
        df_mode_norm = normalize_composition_features(df_features, clip_fractions=True)
        df_mode = residualize_by_replicate_within_dox(df_mode_norm, valid_features)
    else:
        raise ValueError(f"Unsupported mode: {mode_tag}")

    print(f"\n{'-'*80}")
    print(f"PCA MODE: {mode_tag}")
    print(f"{'-'*80}")

    df_mode_imp, missing_before, missing_after = impute_missing_features(df_mode, valid_features)
    n_missing_before = int(missing_before.sum())
    n_missing_after = int(missing_after.sum())
    n_imputed = n_missing_before - n_missing_after
    print(f"  Missing values before imputation: {n_missing_before}")
    print(f"  Missing values imputed: {n_imputed}")
    print(f"  Missing values remaining: {n_missing_after}")

    df_pca = df_mode_imp.dropna(subset=valid_features).copy()
    n_dropped_after_impute = len(df_mode_imp) - len(df_pca)
    print(f"  Organoids in PCA: {len(df_pca)}")
    print(f"  Dropped after imputation: {n_dropped_after_impute}")

    if len(df_pca) < 5:
        print("  Not enough organoids for PCA in this mode. Skipping.")
        return

    X = df_pca[valid_features].values

    if pca_fit_basis == 'auto':
        pca_fit_basis_eff = ('exp_dox_centroids'
                             if trajectory_group_by == 'experiment' and len(experiments) > 1
                             else 'all_organoids')
    else:
        pca_fit_basis_eff = pca_fit_basis

    X_fit, n_fit_samples = build_pca_fit_matrix(df_pca, valid_features, pca_fit_basis_eff)
    if n_fit_samples < 2:
        print(f"  Not enough PCA fit samples for basis '{pca_fit_basis_eff}' (n={n_fit_samples}). Skipping.")
        return

    print(f"  PCA fit basis: {pca_fit_basis_eff} (fit n={n_fit_samples}, project n={len(df_pca)})")

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_scaled = scaler.transform(X)

    n_components = min(len(valid_features), n_fit_samples - 1, 5)
    pca = PCA(n_components=n_components)
    pca.fit(X_fit_scaled)
    scores = pca.transform(X_scaled)

    df_scores = df_pca[meta_cols].copy().reset_index(drop=True)
    for i in range(n_components):
        df_scores[f'PC{i+1}'] = scores[:, i]

    df_loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_features,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    df_variance = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_components)],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_),
        'PCA_Fit_Basis': pca_fit_basis_eff,
        'PCA_Fit_Samples': n_fit_samples,
        'Projected_Samples': len(df_pca)
    })

    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_group_sep = compute_pca_group_separation(df_scores, pc_cols)
    df_dox_step_dist = compute_consecutive_dox_distances(df_scores, experiments)
    if not df_dox_step_dist.empty:
        df_dox_step_dist_summary = (
            df_dox_step_dist.groupby('Dox_Transition', as_index=False)
            .agg(
                Mean_Distance_PC1_PC2=('Distance_PC1_PC2', 'mean'),
                SD_Distance_PC1_PC2=('Distance_PC1_PC2', 'std'),
                Mean_Distance_PC1_PC2_PC3=('Distance_PC1_PC2_PC3', 'mean'),
                SD_Distance_PC1_PC2_PC3=('Distance_PC1_PC2_PC3', 'std'),
                N_Experiments=('Experiment', 'nunique')
            )
        )
    else:
        df_dox_step_dist_summary = pd.DataFrame(columns=[
            'Dox_Transition',
            'Mean_Distance_PC1_PC2', 'SD_Distance_PC1_PC2',
            'Mean_Distance_PC1_PC2_PC3', 'SD_Distance_PC1_PC2_PC3',
            'N_Experiments'
        ])

    out_prefix = f"{label}_{mode_tag}"
    df_scores.to_csv(os.path.join(output_dir, f"{out_prefix}_pca_scores.csv"), index=False)
    df_loadings.to_csv(os.path.join(output_dir, f"{out_prefix}_pca_loadings.csv"))
    df_variance.to_csv(os.path.join(output_dir, f"{out_prefix}_pca_explained_variance.csv"), index=False)
    df_group_sep.to_csv(os.path.join(output_dir, f"{out_prefix}_pca_group_separation.csv"), index=False)
    df_dox_step_dist.to_csv(os.path.join(output_dir, f"{out_prefix}_pca_consecutive_dox_distances.csv"), index=False)
    df_dox_step_dist_summary.to_csv(
        os.path.join(output_dir, f"{out_prefix}_pca_consecutive_dox_distances_summary.csv"),
        index=False
    )
    print(f"\n  PCA CSVs saved with prefix: {out_prefix}")

    print(f"\n  Explained variance:")
    for i in range(n_components):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% "
              f"(cumulative: {np.cumsum(pca.explained_variance_ratio_)[i]*100:.1f}%)")

    print(f"\n  Top loadings per PC:")
    for i in range(min(3, n_components)):
        pc_col = f'PC{i+1}'
        sorted_loadings = df_loadings[pc_col].abs().sort_values(ascending=False)
        top3 = sorted_loadings.head(3)
        signs = ['+' if df_loadings.loc[feat, pc_col] > 0 else '-' for feat in top3.index]
        desc = ', '.join([f"{signs[j]}{feat}({top3.values[j]:.2f})" for j, feat in enumerate(top3.index)])
        print(f"    {pc_col}: {desc}")

    print(f"\n  PC variance decomposition (group separation ratios):")
    for _, row in df_group_sep.iterrows():
        print(f"    {row['PC']}: exp={row['Between_Experiment_Ratio']:.2f}, "
              f"dox={row['Between_Dox_Ratio']:.2f}, exp+dox={row['Between_ExperimentDox_Ratio']:.2f}")

    # Significance uses all available rows per feature (not complete-case only).
    df_sig = compute_feature_significance(df_mode, valid_features, experiments)
    sig_csv = os.path.join(output_dir, f"{out_prefix}_feature_significance.csv")
    df_sig.to_csv(sig_csv, index=False)
    print(f"\n  Feature significance saved: {sig_csv}")
    if trajectory_group_by == 'experiment' and len(experiments) > 1:
        print("  Note: feature significance is between-dox within each experiment (not between experiments).")

    print(f"\n  Feature Significance (Kruskal-Wallis across dox levels):")
    for exp in experiments:
        exp_sig = df_sig[df_sig['Experiment'] == exp].sort_values('KW_p_value')
        print(f"\n    [{exp}]:")
        for _, row in exp_sig.iterrows():
            stars = ('***' if row['KW_p_value'] < 0.001
                     else '**' if row['KW_p_value'] < 0.01
                     else '*' if row['KW_p_value'] < 0.05
                     else 'ns')
            print(f"      {row['Feature']:25s}  H={row['KW_H_statistic']:7.2f}  "
                  f"p={row['KW_p_value']:.4f} {stars}  signal={row['Signal_Ratio']:.2f}")

    if trajectory_group_by == 'experiment' and len(experiments) > 1:
        plot_pca_pc1_pc2_cross_experiment(df_scores, df_loadings, pca, experiments, valid_features,
                                      n_components, output_dir, label, mode_tag)
        plot_pca_pc3_projections_2d_cross_experiment(df_scores, pca, experiments, output_dir, label, mode_tag)
        plot_pca_3d_scatter_cross_experiment(df_scores, experiments, output_dir, label, mode_tag)
    else:
        if trajectory_group_by == 'experiment' and len(experiments) <= 1:
            print(f"  [{mode_tag}] trajectory-group-by=experiment requires >=2 experiments; using dox grouping.")
        plot_pca_pc1_pc2_per_experiment(df_scores, df_loadings, pca, experiments, valid_features,
                        n_components, output_dir, label, mode_tag)
        plot_pca_pc3_projections_2d(df_scores, pca, experiments, output_dir, label, mode_tag)
        plot_pca_3d_scatter(df_scores, experiments, output_dir, label, mode_tag)
    plot_consecutive_dox_distances(df_dox_step_dist, experiments, output_dir, label, mode_tag)
    plot_significance_figure(df_sig, experiments, output_dir, label, mode_tag)


def run_trajectory_analysis(experiments, output_dir, replicate_adjust_mode='both',
                            organoid_limit=3, trajectory_group_by='dox',
                            pca_fit_basis='auto'):
    all_records = []

    for exp_label in experiments:
        mapped_path = DATASET_MAP[exp_label]
        base_path = mapped_path if os.path.isabs(mapped_path) else os.path.join(SCRIPT_DIR, mapped_path)
        if not os.path.isdir(base_path):
            print(f"\n  [{exp_label}] Dataset folder not found: {base_path}")
            continue

        all_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
        file_list = filter_first_n_organoids(all_files, organoid_limit)

        print(f"\n  [{exp_label}] CSV files found: {len(all_files)}")
        print(f"  [{exp_label}] Processing {len(file_list)} organoids...")

        for fpath in file_list:
            fname = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath)
                dox = int(fname.split('dox')[0])
                replicate = os.path.basename(os.path.dirname(fpath))

                metrics = compute_all_metrics(df)
                if metrics is None:
                    continue

                metrics['Dox_Concentration'] = dox
                metrics['Replicate'] = replicate
                metrics['Experiment'] = exp_label
                metrics['File'] = fname
                all_records.append(metrics)

                n_total_features = sum(1 for k in metrics.keys()
                                       if k not in ('Dox_Concentration', 'Replicate', 'Experiment', 'File'))
                n_valid = sum(1 for k, v in metrics.items()
                              if k not in ('Dox_Concentration', 'Replicate', 'Experiment', 'File')
                              and isinstance(v, (int, float)) and not np.isnan(v))
                print(f"    {fname}: {n_valid}/{n_total_features} features")

            except Exception as e:
                print(f"    Error: {fname}: {e}")
                continue

    df_features = pd.DataFrame(all_records)

    if df_features.empty:
        print("No data!")
        return

    # File label
    label = '_'.join(experiments)

    # Save raw feature matrix
    feat_csv = os.path.join(output_dir, f"{label}_feature_matrix.csv")
    df_features.to_csv(feat_csv, index=False)
    print(f"\n  Feature matrix saved: {feat_csv}")

    # Identify feature columns
    meta_cols = ['Dox_Concentration', 'Replicate', 'Experiment', 'File']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    valid_features = [c for c in feature_cols if df_features[c].notna().sum() > 0]
    print(f"  Features: {valid_features}")
    if len(valid_features) == 0:
        print("No valid features with non-missing values.")
        return

    modes = ['raw', 'residualized'] if replicate_adjust_mode == 'both' else [replicate_adjust_mode]
    for mode_tag in modes:
        run_pca_mode(df_features, valid_features, meta_cols, experiments,
                     output_dir, label, mode_tag, trajectory_group_by, pca_fit_basis)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    run_start_epoch = time.time()
    run_start_dt = datetime.now()

    parser = argparse.ArgumentParser(description='Spatial State Trajectory PCA Analysis')
    parser.add_argument('--experiment', type=str, required=True, nargs='+',
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Experiment(s) to analyze. Multiple for cross-comparison.')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')
    parser.add_argument('--replicate-adjust', type=str, default='both',
                        choices=['none', 'raw', 'residualized', 'both'],
                        help='Replicate correction mode. "none" maps to raw.')
    parser.add_argument('--organoid-limit', type=int, default=3,
                        help='Organoids per (replicate,dox) to include. <=0 uses all organoids.')
    parser.add_argument('--trajectory-group-by', type=str, default='dox',
                        choices=['dox', 'experiment'],
                        help='Trajectory centroid grouping: by dox or by experiment order.')
    parser.add_argument('--pca-fit-basis', type=str, default='auto',
                        choices=['auto', 'all_organoids', 'exp_dox_centroids', 'exp_centroids'],
                        help='How scaler/PCA are fit: all organoids, experiment+dox centroids, experiment centroids, or auto.')
    parser.add_argument('--group-by-runtime', type=str, default='yes',
                        choices=['yes', 'no'],
                        help='Store outputs in timestamped run subfolders (default: yes).')

    args = parser.parse_args()

    experiments = args.experiment
    replicate_adjust_mode = 'raw' if args.replicate_adjust == 'none' else args.replicate_adjust
    organoid_limit = args.organoid_limit
    trajectory_group_by = args.trajectory_group_by
    pca_fit_basis = args.pca_fit_basis
    group_by_runtime = args.group_by_runtime == 'yes'
    label = '_'.join(experiments)
    output_subdir = f"{label}_spatial_trajectory"
    base_output_path = os.path.join(args.output_dir, output_subdir)
    run_stamp = run_start_dt.strftime("%Y%m%d_%H%M%S")
    output_path = (os.path.join(base_output_path, f"run_{run_stamp}")
                   if group_by_runtime else base_output_path)
    os.makedirs(output_path, exist_ok=True)

    mode = 'Cross-comparison (shared PCA axes)' if len(experiments) > 1 else 'Single experiment'

    print(f"\n{'='*80}")
    print(f"SPATIAL STATE TRAJECTORY ANALYSIS (PCA)")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Mode: {mode}")
    print(f"Features: Total Cells (max-norm), Endo/Meso Fraction, Radial Mean/Std (Endo/Meso), NMS, Cluster Count/Size, Intra/Inter Distance, Adjacency %")
    print(f"Replicate adjust: {replicate_adjust_mode}")
    print(f"Organoid limit per (replicate,dox): {organoid_limit}")
    print(f"Trajectory grouped by: {trajectory_group_by}")
    print(f"PCA fit basis: {pca_fit_basis}")
    print(f"Group outputs by runtime: {args.group_by_runtime}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    run_trajectory_analysis(experiments, output_path, replicate_adjust_mode,
                            organoid_limit, trajectory_group_by, pca_fit_basis)

    run_end_dt = datetime.now()
    run_elapsed_sec = time.time() - run_start_epoch
    run_meta_file = os.path.join(output_path, "run_metadata.txt")
    with open(run_meta_file, "w") as fh:
        fh.write(f"Run_Start: {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Run_End: {run_end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Runtime_Seconds: {run_elapsed_sec:.2f}\n")
        fh.write(f"Experiments: {', '.join(experiments)}\n")
        fh.write(f"Replicate_Adjust: {replicate_adjust_mode}\n")
        fh.write(f"Organoid_Limit: {organoid_limit}\n")
        fh.write(f"Trajectory_Grouped_By: {trajectory_group_by}\n")
        fh.write(f"PCA_Fit_Basis: {pca_fit_basis}\n")
        fh.write(f"Group_By_Runtime: {args.group_by_runtime}\n")

    print(f"  Run metadata saved: {run_meta_file}")
    print(f"  Runtime: {run_elapsed_sec:.2f} sec")
    print(f"\n  All outputs saved to: {output_path}")
