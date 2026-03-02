import pandas as pd
import numpy as np
import os
import glob
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

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

# Calibration Constants
EPSILON = 30.0
MIN_SAMPLES = 20

# ==============================================================================
# HELPER: FILTER TO FIRST 3 ORGANOIDS PER REPLICATE PER CONDITION
# ==============================================================================
def filter_first_3_organoids(file_list):
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
# ANALYSIS ENGINE
# ==============================================================================
def compute_distances(df):
    """
    Computes:
    1. Inter-cell-type cluster distance: edge-to-edge distances between
       every Endo cluster and every Meso cluster, averaged per organoid.
    2. Intra-cell-type cluster distance: edge-to-edge distances between
       clusters of the SAME lineage (same logic as 20260120 script),
       separately for Endo and Meso.
    """
    target_col = next((c for c in df.columns if 'cell_type' in c.lower()), None)
    if not target_col:
        return None, None

    endo_subset = df[df[target_col] == 2.0]
    meso_subset = df[df[target_col] == 3.0]

    # Run DBSCAN on each lineage
    cluster_data = {}
    for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
        if len(subset) < MIN_SAMPLES:
            cluster_data[name] = []
            continue

        coords = subset[['X', 'Y', 'Z']].values
        db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(coords)
        labels = db.labels_
        unique_labels = [l for l in np.unique(labels) if l != -1]

        clusters = [coords[labels == l] for l in unique_labels]
        cluster_data[name] = clusters

    # --- Inter-cell-type: cross-lineage cluster distance (Endo-Meso) ---
    inter_dist = None
    endo_clusters = cluster_data.get('Endo', [])
    meso_clusters = cluster_data.get('Meso', [])

    if len(endo_clusters) > 0 and len(meso_clusters) > 0:
        pair_distances = []
        for endo_cl in endo_clusters:
            tree = cKDTree(endo_cl)
            for meso_cl in meso_clusters:
                dists, _ = tree.query(meso_cl, k=1)
                pair_distances.append(np.min(dists))
        inter_dist = np.mean(pair_distances)

    # --- Intra-cell-type: same-lineage inter-cluster distance (edge-to-edge) ---
    intra_results = {}
    for name, clusters in cluster_data.items():
        if len(clusters) > 1:
            pair_distances = []
            for i in range(len(clusters)):
                tree_i = cKDTree(clusters[i])
                for j in range(i + 1, len(clusters)):
                    dist, _ = tree_i.query(clusters[j], k=1)
                    pair_distances.append(np.min(dist))
            intra_results[name] = np.mean(pair_distances)

    return inter_dist, intra_results

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_distance_analysis(base_path, exp_label, output_dir):
    inter_records = []
    intra_records = []

    all_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    csv_files = filter_first_3_organoids(all_files)
    print(f"Processing {len(csv_files)} organoids (first 3 per replicate/condition from {len(all_files)} total)...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            dox = int(file_name.split('dox')[0])
            replicate = os.path.basename(os.path.dirname(file_path))

            inter_dist, intra_results = compute_distances(df)

            if inter_dist is None and intra_results is None:
                continue

            if inter_dist is not None:
                inter_records.append({
                    'Dox_Concentration': dox,
                    'Inter_Cluster_Distance': inter_dist,
                    'Pair': 'Endo-Meso',
                    'Experiment': exp_label,
                    'Replicate': replicate,
                    'File': file_name
                })

            for lineage, dist in (intra_results or {}).items():
                intra_records.append({
                    'Dox_Concentration': dox,
                    'Intra_CellType_Cluster_Distance': dist,
                    'Cell_Type': lineage,
                    'Experiment': exp_label,
                    'Replicate': replicate,
                    'File': file_name
                })

        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            continue

    df_inter = pd.DataFrame(inter_records)
    df_intra = pd.DataFrame(intra_records)

    if df_inter.empty and df_intra.empty:
        print("No data to plot!")
        return

    # ==============================================================================
    # VISUALIZATION: SIDE-BY-SIDE
    # ==============================================================================
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    lineage_palette = {'Endo': '#d62728', 'Meso': '#1fb471'}

    # Panel 1: Cross-lineage inter-cluster distance
    if not df_inter.empty:
        sns.lineplot(data=df_inter, x='Dox_Concentration', y='Inter_Cluster_Distance',
                     marker='o', color='#7570b3', errorbar='se', linewidth=2.5, ax=ax1)
        ax1.set_title("Inter-Cell-Type Cluster Distance (Endo-Meso)\nEdge-to-Edge", fontweight='bold')
        ax1.set_xscale('symlog', linthresh=10)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Dox Concentration (ng/mL)")
        ax1.set_ylabel("Distance (μm)")

    # Panel 2: Intra-cluster distance (per lineage)
    if not df_intra.empty:
        sns.lineplot(data=df_intra, x='Dox_Concentration', y='Intra_CellType_Cluster_Distance',
                     hue='Cell_Type', palette=lineage_palette,
                     marker='o', errorbar='se', linewidth=2.5, ax=ax2)
        ax2.set_title("Intra-Cell-Type Cluster Distance\n(Same-Lineage Edge-to-Edge)", fontweight='bold')
        ax2.set_xscale('symlog', linthresh=10)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel("Dox Concentration (ng/mL)")
        ax2.set_ylabel("Distance (μm)")

    plt.suptitle(f"{exp_label} - Inter & Intra Cluster Distance Analysis",
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_file = os.path.join(output_dir, f"{exp_label}_Inter_Intra_CellType_Cluster_Distance.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_file}")
    plt.close()

    # Save raw and plot-ready data tables
    inter_csv = os.path.join(output_dir, f"{exp_label}_Inter_Cluster_Distance_Organoid_Level.csv")
    intra_csv = os.path.join(output_dir, f"{exp_label}_Intra_Cluster_Distance_Organoid_Level.csv")

    plot_chunks = []
    if not df_inter.empty:
        inter_plot = df_inter.rename(columns={'Inter_Cluster_Distance': 'Value'}).copy()
        inter_plot['Metric'] = 'Inter_Cluster_Distance'
        inter_plot['Lineage_or_Pair'] = inter_plot['Pair']
        inter_plot = inter_plot[['Experiment', 'Dox_Concentration', 'Replicate', 'File', 'Metric', 'Lineage_or_Pair', 'Value']]
        plot_chunks.append(inter_plot)

    if not df_intra.empty:
        intra_plot = df_intra.rename(columns={'Intra_CellType_Cluster_Distance': 'Value'}).copy()
        intra_plot['Metric'] = 'Intra_CellType_Cluster_Distance'
        intra_plot['Lineage_or_Pair'] = intra_plot['Cell_Type']
        intra_plot = intra_plot[['Experiment', 'Dox_Concentration', 'Replicate', 'File', 'Metric', 'Lineage_or_Pair', 'Value']]
        plot_chunks.append(intra_plot)

    plot_data = pd.concat(plot_chunks, ignore_index=True) if plot_chunks else pd.DataFrame()
    plot_csv = os.path.join(output_dir, f"{exp_label}_Inter_Intra_Plot_Data.csv")

    if not df_inter.empty:
        df_inter.to_csv(inter_csv, index=False)
        print(f"✓ Data saved: {inter_csv}")
    if not df_intra.empty:
        df_intra.to_csv(intra_csv, index=False)
        print(f"✓ Data saved: {intra_csv}")
    if not plot_data.empty:
        plot_data.to_csv(plot_csv, index=False)
        print(f"✓ Data saved: {plot_csv}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inter & Intra Cluster Distance Analysis')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Which experiment dataset to analyze')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')

    args = parser.parse_args()

    mapped_path = DATASET_MAP[args.experiment]
    base_path = mapped_path if os.path.isabs(mapped_path) else os.path.join(SCRIPT_DIR, mapped_path)

    output_subdir = f"{args.experiment}_inter_intra_distance"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"INTER & INTRA CLUSTER DISTANCE ANALYSIS")
    print(f"Dataset: {args.experiment} ({base_path})")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    run_distance_analysis(base_path, args.experiment, output_path)
