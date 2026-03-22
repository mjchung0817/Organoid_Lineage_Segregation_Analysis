import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import argparse
from esda.moran import Moran, Moran_Local
from libpysal.weights import DistanceBand
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ==============================================================================
# DATASET MAPPING
# ==============================================================================
DATASET_MAP = {
    'exp1': 'GATA-HA_Rep1-3_Ex1',
    'exp2_high_cn': 'GATA-HA-Ex2_high_copy_num',
    'exp2_low_cn': 'GATA-HA-Ex2_low_copy_num',
    'exp3': 'GATA-HA-BMP4+Wnt5a_Ex3'
}

CHANNELS = {'Endo': 'log1p_normed_546', 'Meso': 'log1p_normed_647'}
EPSILON = 30.0
MIN_SAMPLES = 20

# ==============================================================================
# HELPER: FILTER TO FIRST 3 ORGANOIDS PER REPLICATE PER CONDITION
# ==============================================================================
def filter_first_3_organoids(file_list):
    """
    Group files by replicate and condition, keep only first 3 organoids numerically.
    """
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
        filtered.extend([fpath for _, fpath in files_sorted[:3]])

    return filtered

# ==============================================================================
# MATH ENGINE: SPATIAL POWER (sPCA)
# ==============================================================================
def get_spca_loadings(df, channels_list):
    data_std = (df[channels_list] - df[channels_list].mean()) / df[channels_list].std()
    coords = df[['X', 'Y', 'Z']].values
    w = DistanceBand.from_array(coords, threshold=EPSILON, binary=True)
    lagged_data = np.array([np.mean(data_std.values[w.neighbors[i]], axis=0) if len(w.neighbors[i]) > 0
                           else data_std.values[i] for i in range(len(data_std))])
    pca = PCA(n_components=1)
    pca.fit(lagged_data)
    return dict(zip(channels_list, np.abs(pca.components_[0])))

# ==============================================================================
# DATA AGGREGATION
# ==============================================================================
def run_spatial_analysis(base_path, exp_label, output_dir):
    all_stats = []
    all_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    file_list = filter_first_3_organoids(all_files)

    print(f"Processing {len(file_list)} organoids (first 3 per replicate/condition from {len(all_files)} total)...")

    for file_path in file_list:
        fname = os.path.basename(file_path)
        dose_match = re.search(r"(\d+)dox", fname)
        cond_match = re.search(r"\+(.+?)_", fname)

        dose = int(dose_match.group(1)) if dose_match else 0
        condition = cond_match.group(1) if cond_match else "Control"
        rep_name = os.path.basename(os.path.dirname(file_path))

        try:
            df = pd.read_csv(file_path)
            w = DistanceBand.from_array(df[['X', 'Y', 'Z']].values, threshold=EPSILON, binary=True)
            db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(df[['X', 'Y', 'Z']].values)
            df['cluster'] = db.labels_

            spca_results = get_spca_loadings(df, list(CHANNELS.values()))

            for label, channel in CHANNELS.items():
                if channel not in df.columns: continue
                vals = df[channel].values
                if len(vals) < 2 or np.all(vals == vals[0]): continue

                mi = Moran(vals, w)
                lisa = Moran_Local(vals, w)
                df['lisa_val'] = lisa.Is
                avg_cluster_lisa = df[df['cluster'] != -1]['lisa_val'].mean()

                all_stats.append({
                    'Dose': dose, 'Condition': condition, 'Rep': rep_name, 'Lineage': label,
                    'Morans_I': mi.I, 'Morans_I_pval': mi.p_sim,
                    'Avg_Cluster_LISA': avg_cluster_lisa, 'Spatial_Power': spca_results[channel]
                })
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    stats_df = pd.DataFrame(all_stats)

    # Save raw stats (includes p-values) to CSV
    csv_path = os.path.join(output_dir, f"{exp_label}_moransI_sPCA_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"  ✓ Stats saved to: {csv_path}")

    # ==============================================================================
    # VISUALIZATION: PER-CONDITION SEPARATED FILES
    # ==============================================================================
    sns.set_context("paper", font_scale=1.5)
    metrics = [('Morans_I', "Global Moran's I"),
               ('Avg_Cluster_LISA', "Avg LISA (Within Clusters)"),
               ('Spatial_Power', "Lineage Spatial Power (sPCA)")]

    palette = {'Endo': '#d62728', 'Meso': '#1fb471'}
    rep_alphas = {"GATA6-HA_Rep1": 0.15, "GATA6-HA_Rep2": 0.35, "GATA6-HA_Rep3": 0.55}

    for cond in stats_df['Condition'].unique():
        cond_df = stats_df[stats_df['Condition'] == cond]

        for col, ylabel in metrics:
            plt.figure(figsize=(10, 7))

            # 1. Plot individual replicates (Faint dimmer lines)
            for rep in cond_df['Rep'].unique():
                rep_data = cond_df[cond_df['Rep'] == rep]
                sns.lineplot(data=rep_data, x='Dose', y=col, hue='Lineage', ax=plt.gca(),
                             legend=False, alpha=rep_alphas.get(rep, 0.3), linewidth=1.5, palette=palette)

            # 2. Plot mean across org and rep with SE bars
            sns.lineplot(data=cond_df, x='Dose', y=col, hue='Lineage', ax=plt.gca(),
                         marker='o', markersize=10, linewidth=4, errorbar='se',
                         palette=palette, err_style='bars')

            plt.xscale('symlog', linthresh=10)
            plt.xticks([0, 10, 25, 50, 100, 250, 500, 1000])
            plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
            plt.title(f"{exp_label} | {cond}\n{ylabel}", fontweight='bold')
            plt.xlim(-1, 1100)
            plt.xlabel("Dox Concentration (ng/mL)")

            plt.tight_layout()
            output_file = os.path.join(output_dir, f"{exp_label}_{cond}_{col}.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"  ✓ Saved: {output_file}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spatial Cluster Analysis: Moran\'s I and sPCA')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Which experiment dataset to analyze')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')

    args = parser.parse_args()

    # Get dataset path
    mapped_path = DATASET_MAP[args.experiment]
    base_path = mapped_path if os.path.isabs(mapped_path) else os.path.join(PROJECT_ROOT, mapped_path)

    # Create output directory
    output_subdir = f"{args.experiment}_moransI_sPCA"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"SPATIAL CLUSTER ANALYSIS: MORAN'S I & sPCA")
    print(f"Dataset: {args.experiment} ({base_path})")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    run_spatial_analysis(base_path, args.experiment, output_path)

    print(f"\n✓ All plots saved to: {output_path}")
