import pandas as pd
import numpy as np
import os
import glob
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from scipy.spatial import cKDTree, KDTree

try:
    from sklearn.cluster import DBSCAN  # type: ignore
except ImportError:
    DBSCAN = None

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
PARAMS = {
    2.0: {'eps': 30, 'ms': 20, 'name': 'Endo'},
    3.0: {'eps': 30, 'ms': 20, 'name': 'Meso'}
}
TOTAL_CLUSTER_NAME = 'Total'


def dbscan_labels(coords, eps, min_samples):
    """
    Return DBSCAN labels for coords.
    Uses sklearn when available; otherwise uses a lightweight KDTree-based fallback.
    """
    if DBSCAN is not None:
        return DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_

    n = len(coords)
    if n == 0:
        return np.array([], dtype=int)

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=eps)
    labels = np.full(n, -1, dtype=int)  # -1 = noise/unassigned
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh_i = neighbors[i]
        if len(neigh_i) < min_samples:
            continue

        labels[i] = cluster_id
        queue = deque(neigh_i)
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors[j]
                if len(neigh_j) >= min_samples:
                    queue.extend(neigh_j)
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels

# ==============================================================================
# HELPER: FILTER TO FIRST 3 ORGANOIDS PER REPLICATE PER CONDITION
# ==============================================================================
def filter_first_3_organoids(file_list):
    """
    Group files by replicate and condition, keep only first 3 organoids numerically.
    Filename pattern: {dox}dox_{sample}_{organoid_num}.csv
    """
    # Group by (replicate_dir, dox_concentration)
    grouped = {}
    for fpath in file_list:
        fname = os.path.basename(fpath)
        replicate = os.path.basename(os.path.dirname(fpath))

        # Extract dox concentration
        dox_match = re.search(r'(\d+)dox', fname)
        if not dox_match:
            continue
        dox = int(dox_match.group(1))

        # Extract organoid number (last _XXX before .csv)
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

    # Sort each group numerically and take first 3
    filtered = []
    for key, files in grouped.items():
        files_sorted = sorted(files, key=lambda x: x[0])  # Sort by organoid number
        filtered.extend([fpath for _, fpath in files_sorted[:3]])  # Take first 3

    return filtered

# ==============================================================================
# ENGINE: PER-ORGANOID SPATIAL METRICS
# ==============================================================================
def run_pipeline(base_path, exp_label):
    organoid_records = []
    distance_records = []
    count_records = []
    nms_records = []

    # Recursive search to catch all Rep folders
    all_csv_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    csv_files = filter_first_3_organoids(all_csv_files)
    print(f"Scanning {len(csv_files)} organoids (first 3 per replicate/condition from {len(all_csv_files)} total)...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            # Extract Dox from filename (e.g., '100dox_...')
            dox = int(file_name.split('dox')[0])
            replicate = os.path.basename(os.path.dirname(file_path))

            # Safe column search for 'adusted' vs 'adjusted'
            target_col = next((c for c in df.columns if 'cell_type' in c.lower()), None)
            if not target_col: continue

            # Restrict analysis to Endo/Meso only (no triple-negative/pluripotent tracking)
            lineage_df = df[df[target_col].isin([2.0, 3.0])].copy()

            # --- PER-LINEAGE METRICS ---
            lineage_cluster_counts = {p['name']: 0 for p in PARAMS.values()}
            for ct_id, p in PARAMS.items():
                subset = lineage_df[lineage_df[target_col] == ct_id]
                cluster_count = 0
                cluster_coords_list = []

                if len(subset) >= p['ms']:
                    coords = subset[['X', 'Y', 'Z']].values
                    labels = dbscan_labels(coords, eps=p['eps'], min_samples=p['ms'])
                    unique_labels = [l for l in np.unique(labels) if l != -1]
                    cluster_count = len(unique_labels)

                    if cluster_count > 0:
                        # (A) Cluster Size Logic
                        sizes = [len(coords[labels == l]) for l in unique_labels]
                        organoid_records.append({
                            'Experiment': exp_label,
                            'Dox_Concentration': dox,
                            'Replicate': replicate,
                            'Cell_Type': p['name'],
                            'Avg_Cluster_Size': np.mean(sizes),
                            'File': file_name
                        })

                        # (B) Edge-to-Edge Distance Logic (KDTree)
                        for l in unique_labels:
                            cluster_coords_list.append(coords[labels == l])

                        if len(cluster_coords_list) > 1:
                            pair_distances = []
                            for i in range(len(cluster_coords_list)):
                                tree_i = cKDTree(cluster_coords_list[i])
                                for j in range(i + 1, len(cluster_coords_list)):
                                    dist, _ = tree_i.query(cluster_coords_list[j], k=1)
                                    pair_distances.append(np.min(dist))

                            distance_records.append({
                                'Experiment': exp_label,
                                'Dox_Concentration': dox,
                                'Replicate': replicate,
                                'Cell_Type': p['name'],
                                'Edge_to_Edge_Distance_um': np.mean(pair_distances),
                                'File': file_name
                            })

                lineage_cluster_counts[p['name']] = cluster_count
                # (C) Cluster Count Logic (always record, including zero)
                count_records.append({
                    'Experiment': exp_label,
                    'Dox_Concentration': dox,
                    'Replicate': replicate,
                    'Cell_Type': p['name'],
                    'Cluster_Count': cluster_count,
                    'File': file_name
                })

            # --- TOTAL CLUSTER COUNT (additive: Endo + Meso) ---
            count_records.append({
                'Experiment': exp_label,
                'Dox_Concentration': dox,
                'Replicate': replicate,
                'Cell_Type': TOTAL_CLUSTER_NAME,
                'Cluster_Count': int(
                    lineage_cluster_counts.get('Endo', 0) +
                    lineage_cluster_counts.get('Meso', 0)
                ),
                'File': file_name
            })

            # --- NMS: Endo vs Meso only ---
            RADIUS = 100.0  # Same as in calc_nms script

            endo_subset = lineage_df[lineage_df[target_col] == 2.0]
            meso_subset = lineage_df[lineage_df[target_col] == 3.0]
            non_endo_subset = lineage_df[lineage_df[target_col] != 2.0]  # Meso only in filtered set
            non_meso_subset = lineage_df[lineage_df[target_col] != 3.0]  # Endo only in filtered set

            nms_e = None
            nms_m = None

            # NMS (Ref: Endoderm vs Mesoderm)
            if len(endo_subset) > 10 and len(non_endo_subset) > 10:
                e_coords = endo_subset[['X', 'Y', 'Z']].values
                non_e_coords = non_endo_subset[['X', 'Y', 'Z']].values
                e_tree = KDTree(e_coords)
                non_e_tree = KDTree(non_e_coords)

                n_non_e_around_e = sum([len(n) for n in non_e_tree.query_ball_point(e_coords, r=RADIUS)])
                n_e_around_e = sum([len(n)-1 for n in e_tree.query_ball_point(e_coords, r=RADIUS)])
                nms_e = (n_non_e_around_e / n_e_around_e) / (len(non_endo_subset) / len(endo_subset)) if n_e_around_e > 0 else 0

            # NMS (Ref: Mesoderm vs Endoderm)
            if len(meso_subset) > 10 and len(non_meso_subset) > 10:
                m_coords = meso_subset[['X', 'Y', 'Z']].values
                non_m_coords = non_meso_subset[['X', 'Y', 'Z']].values
                m_tree = KDTree(m_coords)
                non_m_tree = KDTree(non_m_coords)

                n_non_m_around_m = sum([len(n) for n in non_m_tree.query_ball_point(m_coords, r=RADIUS)])
                n_m_around_m = sum([len(n)-1 for n in m_tree.query_ball_point(m_coords, r=RADIUS)])
                nms_m = (n_non_m_around_m / n_m_around_m) / (len(non_meso_subset) / len(meso_subset)) if n_m_around_m > 0 else 0

            # Only record if at least one NMS value was calculated
            if nms_e is not None or nms_m is not None:
                record = {
                    'Experiment': exp_label,
                    'Dox_Concentration': dox,
                    'Replicate': replicate,
                    'File': file_name
                }
                if nms_e is not None:
                    record['NMS_Endo'] = nms_e
                if nms_m is not None:
                    record['NMS_Meso'] = nms_m
                nms_records.append(record)

        except Exception as e:
            continue

    return (pd.DataFrame(organoid_records), pd.DataFrame(distance_records),
            pd.DataFrame(count_records), pd.DataFrame(nms_records))

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBSCAN Cluster Size and Inter-Distance Analysis')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Which experiment dataset to analyze')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')
    parser.add_argument('--append-only', action='store_true',
                        help='Only append new 3-label cluster-count outputs without overwriting existing exports.')

    args = parser.parse_args()

    # Get dataset path
    mapped_path = DATASET_MAP[args.experiment]
    base_path = mapped_path if os.path.isabs(mapped_path) else os.path.join(SCRIPT_DIR, mapped_path)

    # Create output directory
    output_subdir = f"{args.experiment}_dbscan_cluster_analysis"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DBSCAN CLUSTER ANALYSIS")
    print(f"Dataset: {args.experiment} ({base_path})")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    # Run pipeline
    df_size, df_dist, df_count, df_nms = run_pipeline(base_path, args.experiment)

    # Save organoid-level raw tables for downstream figure cleanup
    csv_size = os.path.join(output_path, f"{args.experiment}_Cluster_Size_Organoid_Level.csv")
    csv_dist = os.path.join(output_path, f"{args.experiment}_InterCluster_Separation_Organoid_Level.csv")
    csv_count = os.path.join(output_path, f"{args.experiment}_Cluster_Count_Organoid_Level.csv")
    csv_nms = os.path.join(output_path, f"{args.experiment}_NMS_Organoid_Level.csv")

    # Always save append-only cluster count artifacts with unique names
    if not df_count.empty:
        df_count_3label = df_count[df_count['Cell_Type'].isin(['Endo', 'Meso', 'Total'])].copy()
        csv_count_3label = os.path.join(output_path, f"{args.experiment}_Cluster_Count_Organoid_Level_3Labels.csv")
        df_count_3label.to_csv(csv_count_3label, index=False)
        print(f"✓ Data saved: {csv_count_3label}")

        plt.figure(figsize=(8, 5))
        count_palette = {'Endo': '#d62728', 'Meso': '#1fb471', 'Total': '#4c78a8'}
        sns.lineplot(
            data=df_count_3label,
            x='Dox_Concentration',
            y='Cluster_Count',
            hue='Cell_Type',
            palette=count_palette,
            marker='^',
            errorbar='sd',
            linewidth=2.2
        )
        plt.xscale('symlog', linthresh=10)
        plt.xlim(left=0)
        plt.xlabel("Dox Concentration (ng/mL)")
        plt.ylabel("Number of Clusters")
        plt.title("Cluster Count vs Dox (Endo / Meso / Total=Endo+Meso)")
        plt.tight_layout()
        count_plot_3label = os.path.join(output_path, f"{args.experiment}_Cluster_Count_3Labels_vs_Dox.png")
        plt.savefig(count_plot_3label, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure saved to: {count_plot_3label}")

    if args.append_only:
        print("\nAppend-only mode complete (existing legacy outputs preserved).")
        raise SystemExit(0)

    if not df_size.empty:
        df_size.to_csv(csv_size, index=False)
        print(f"✓ Data saved: {csv_size}")
    if not df_dist.empty:
        df_dist.to_csv(csv_dist, index=False)
        print(f"✓ Data saved: {csv_dist}")
    if not df_count.empty:
        df_count.to_csv(csv_count, index=False)
        print(f"✓ Data saved: {csv_count}")
    if not df_nms.empty:
        df_nms.to_csv(csv_nms, index=False)
        print(f"✓ Data saved: {csv_nms}")

    # Visualization
    if not df_size.empty:
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        sns.set_theme(style="whitegrid", context="talk")

        # Consistent lineage colors: Endo=Red, Meso=Green
        lineage_palette = {'Endo': '#d62728', 'Meso': '#1fb471'}

        # Consistent y-axis scales across experiments
        CLUSTER_SIZE_YLIM = (0, 2000)
        CLUSTER_COUNT_YLIM = (0, 80)

        # Check if we need a broken axis: only when group means exceed limit by >20%
        mean_per_group = df_size.groupby(['Dox_Concentration', 'Cell_Type'])['Avg_Cluster_Size'].mean()
        data_max = df_size['Avg_Cluster_Size'].max()
        needs_break = mean_per_group.max() > CLUSTER_SIZE_YLIM[1] * 1.2  # Trigger at >2400

        fig = plt.figure(figsize=(24, 16))
        outer_gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        # --- Panel 1: Cluster Size (broken axis if needed) ---
        if needs_break:
            # Split top-left cell into two: small top window (1) + large bottom (4)
            inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, 0],
                                              height_ratios=[1, 4], hspace=0.08)
            ax_top = fig.add_subplot(inner_gs[0])
            ax_bot = fig.add_subplot(inner_gs[1])

            # Determine top window range based on per-dox GROUP MEANS (not individual max)
            mean_per_dox = df_size.groupby(['Dox_Concentration', 'Cell_Type'])['Avg_Cluster_Size'].mean()
            mean_max = mean_per_dox.max()
            top_lo = max(CLUSTER_SIZE_YLIM[1] + 200, mean_max - 1500)
            top_hi = mean_max * 1.15

            # Bottom axis: full CI bands for the main data range
            sns.lineplot(data=df_size, x='Dox_Concentration', y='Avg_Cluster_Size', hue='Cell_Type',
                         palette=lineage_palette, marker='o', errorbar='sd', ax=ax_bot)
            ax_bot.set_xscale('symlog', linthresh=10)
            ax_bot.set_xlim(left=0)
            ax_bot.get_legend().remove()

            # Top axis: means only (no CI bands) so extreme points are clearly visible
            sns.lineplot(data=df_size, x='Dox_Concentration', y='Avg_Cluster_Size', hue='Cell_Type',
                         palette=lineage_palette, marker='o', markersize=8, errorbar=None, ax=ax_top)
            ax_top.set_xscale('symlog', linthresh=10)
            ax_top.set_xlim(left=0)
            ax_top.get_legend().remove()

            # Set y ranges
            ax_top.set_ylim(top_lo, top_hi)
            ax_bot.set_ylim(*CLUSTER_SIZE_YLIM)

            # Hide spines between the two axes
            ax_top.spines['bottom'].set_visible(False)
            ax_bot.spines['top'].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_top.set_xlabel("")
            ax_bot.set_xlabel("Dox Concentration (ng/mL)")
            ax_bot.set_ylabel("Avg Cluster Size (cells)")

            # Draw break marks (diagonal lines)
            d = 0.015
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.5)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_bot.transAxes)
            ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            ax_top.set_title("Cluster Size (Avg Cells per Cluster)", fontweight='bold')
            ax_top.legend(loc='upper right', fontsize=9)
        else:
            ax_size = fig.add_subplot(outer_gs[0, 0])
            sns.lineplot(data=df_size, x='Dox_Concentration', y='Avg_Cluster_Size', hue='Cell_Type',
                         palette=lineage_palette, marker='o', errorbar='sd', ax=ax_size)
            ax_size.set_title("Cluster Size (Avg Cells per Cluster)", fontweight='bold')
            ax_size.set_xscale('symlog', linthresh=10)
            ax_size.set_xlim(left=0)
            ax_size.set_ylim(*CLUSTER_SIZE_YLIM)
            ax_size.set_xlabel("Dox Concentration (ng/mL)")
            ax_size.set_ylabel("Avg Cluster Size (cells)")

        # --- Panel 2: Inter-Cluster Separation (Edge-to-Edge) ---
        ax_dist = fig.add_subplot(outer_gs[0, 1])
        if not df_dist.empty:
            sns.lineplot(data=df_dist, x='Dox_Concentration', y='Edge_to_Edge_Distance_um', hue='Cell_Type',
                         palette=lineage_palette, marker='s', errorbar='sd', ax=ax_dist)
            ax_dist.set_title("Inter-Cluster Separation (Edge-to-Edge)", fontweight='bold')
            ax_dist.set_xscale('symlog', linthresh=10)
            ax_dist.set_xlim(left=0)
            ax_dist.set_ylim(bottom=0)
            ax_dist.set_xlabel("Dox Concentration (ng/mL)")
            ax_dist.set_ylabel("Separation Distance (μm)")
        else:
            ax_dist.set_axis_off()

        # --- Panel 3: Cluster Count ---
        if not df_count.empty:
            ax_count = fig.add_subplot(outer_gs[1, 0])
            count_palette = {'Endo': '#d62728', 'Meso': '#1fb471', 'Total': '#4c78a8'}
            count_ymax = max(CLUSTER_COUNT_YLIM[1], int(np.ceil(df_count['Cluster_Count'].max() * 1.1)))
            sns.lineplot(data=df_count, x='Dox_Concentration', y='Cluster_Count', hue='Cell_Type',
                         palette=count_palette, marker='^', errorbar='sd', ax=ax_count)
            ax_count.set_title("Cluster Count (# of Clusters, Total=Endo+Meso)", fontweight='bold')
            ax_count.set_xscale('symlog', linthresh=10)
            ax_count.set_xlim(left=0)
            ax_count.set_ylim(0, count_ymax)
            ax_count.set_xlabel("Dox Concentration (ng/mL)")
            ax_count.set_ylabel("Number of Clusters")

        # --- Panel 4: NMS (Mixing Score) ---
        if not df_nms.empty:
            ax_nms = fig.add_subplot(outer_gs[1, 1])
            nms_value_vars = [col for col in df_nms.columns if col.startswith('NMS_')]
            nms_long = df_nms.melt(id_vars=['Experiment', 'Dox_Concentration', 'Replicate', 'File'],
                                   value_vars=nms_value_vars,
                                   var_name='Lineage', value_name='NMS')
            nms_long['Lineage'] = nms_long['Lineage'].str.replace('NMS_', '')
            nms_long = nms_long.dropna(subset=['NMS'])

            nms_palette = {'Endo': '#d62728', 'Meso': '#1fb471'}

            sns.lineplot(data=nms_long, x='Dox_Concentration', y='NMS', hue='Lineage',
                         palette=nms_palette, marker='D', errorbar='sd', linewidth=2.5, ax=ax_nms)
            ax_nms.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random Mixing')
            ax_nms.set_title("Normalized Mixing Score (NMS)", fontweight='bold')
            ax_nms.set_xscale('symlog', linthresh=10)
            ax_nms.set_xlim(left=0)
            ax_nms.set_xlabel("Dox Concentration (ng/mL)")
            ax_nms.set_ylabel("NMS Value")

            nms_plot_csv = os.path.join(output_path, f"{args.experiment}_NMS_Plot_Data_Long.csv")
            nms_long.to_csv(nms_plot_csv, index=False)
            print(f"✓ Data saved: {nms_plot_csv}")

        plt.suptitle(f"{args.experiment} - Spatial Analysis (Eps: 30, MinSamples: 20)",
                     fontsize=22, fontweight='bold', y=0.995)

        # Save figure
        output_file = os.path.join(output_path, f"{args.experiment}_4Panel_Spatial_Analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {output_file}")
        plt.close()
    else:
        print("No data to plot!")
