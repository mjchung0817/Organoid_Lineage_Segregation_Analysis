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
EPSILON = 30.0  # DBSCAN clustering threshold
MIN_SAMPLES = 20  # DBSCAN min samples
PROXIMITY_THRESHOLD = 30.0  # Distance threshold for "touching" (μm)

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

        key = (replicate, dox)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((org_num, fpath))

    filtered = []
    for key, files in grouped.items():
        files_sorted = sorted(files, key=lambda x: x[0])
        filtered.extend([fpath for _, fpath in files_sorted[:3]])

    return filtered

# ==============================================================================
# CLUSTER PROXIMITY ANALYSIS ENGINE
# ==============================================================================
def calculate_cluster_kissing(df):
    """
    Calculate three touching metrics for Endo/Meso cluster interfaces:
    1) Minority clusters touching majority / total minority.
    2) Majority clusters touching minority / total majority.
    3) Sum(A_endo_meso) / (dim(endo) * dim(meso)) where A is cluster-touch adjacency.
    """
    target_col = next((c for c in df.columns if 'cell_type' in c.lower()), None)
    if not target_col:
        return None

    endo_subset = df[df[target_col] == 2.0]
    meso_subset = df[df[target_col] == 3.0]

    # Run DBSCAN on each lineage
    results = {}
    for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
        if len(subset) < MIN_SAMPLES:
            results[name] = {'clusters': [], 'n_clusters': 0}
            continue

        coords = subset[['X', 'Y', 'Z']].values
        db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(coords)
        labels = db.labels_
        unique_labels = [l for l in np.unique(labels) if l != -1]

        # Store coordinates for each cluster
        cluster_list = []
        for label in unique_labels:
            cluster_points = coords[labels == label]
            cluster_list.append(cluster_points)

        results[name] = {
            'clusters': cluster_list,
            'n_clusters': len(cluster_list)
        }

    # Determine majority/minority by cluster count
    n_endo_clusters = results['Endo']['n_clusters']
    n_meso_clusters = results['Meso']['n_clusters']

    if n_endo_clusters == 0 and n_meso_clusters == 0:
        return None

    if n_endo_clusters >= n_meso_clusters:
        majority = 'Endo'
        minority = 'Meso'
    else:
        majority = 'Meso'
        minority = 'Endo'

    majority_clusters = results[majority]['clusters']
    minority_clusters = results[minority]['clusters']

    n_endo = len(results['Endo']['clusters'])
    n_meso = len(results['Meso']['clusters'])

    touching_pairs = 0
    touching_endo = 0
    touching_meso = 0

    if len(majority_clusters) == 0 or len(minority_clusters) == 0:
        return {
            'Majority_Lineage': majority,
            'Minority_Lineage': minority,
            'Minority_Touching_Pct': 0.0,
            'Majority_Touching_Pct': 0.0,
            'Adjacency_Density_Pct': 0.0,
            'Kissing_Percentage': 0.0,
            'Total_Majority_Clusters': len(majority_clusters),
            'Total_Minority_Clusters': len(minority_clusters),
            'Touching_Minority_Clusters': 0,
            'Touching_Majority_Clusters': 0,
            'Touching_Cluster_Pairs': 0,
            'Total_Endo_Clusters': n_endo,
            'Total_Meso_Clusters': n_meso,
            'Total_Possible_Endo_Meso_Pairs': n_endo * n_meso
        }

    # Build bipartite touch adjacency between Endo clusters and Meso clusters.
    endo_clusters = results['Endo']['clusters']
    meso_clusters = results['Meso']['clusters']
    adjacency = np.zeros((n_endo, n_meso), dtype=np.uint8)

    for i, endo_cluster in enumerate(endo_clusters):
        endo_tree = cKDTree(endo_cluster)
        for j, meso_cluster in enumerate(meso_clusters):
            distances, _ = endo_tree.query(meso_cluster, k=1)
            min_dist = np.min(distances)
            if min_dist <= PROXIMITY_THRESHOLD:
                adjacency[i, j] = 1

    touching_pairs = int(adjacency.sum())
    if n_endo > 0:
        touching_endo = int(np.sum(np.any(adjacency == 1, axis=1)))
    if n_meso > 0:
        touching_meso = int(np.sum(np.any(adjacency == 1, axis=0)))

    touching_by_lineage = {'Endo': touching_endo, 'Meso': touching_meso}
    touching_minority = touching_by_lineage.get(minority, 0)
    touching_majority = touching_by_lineage.get(majority, 0)

    minority_touching_pct = (touching_minority / len(minority_clusters)) * 100.0 if len(minority_clusters) > 0 else 0.0
    majority_touching_pct = (touching_majority / len(majority_clusters)) * 100.0 if len(majority_clusters) > 0 else 0.0

    possible_pairs = n_endo * n_meso
    adjacency_density_pct = (touching_pairs / possible_pairs) * 100.0 if possible_pairs > 0 else 0.0

    return {
        'Majority_Lineage': majority,
        'Minority_Lineage': minority,
        'Minority_Touching_Pct': minority_touching_pct,
        'Majority_Touching_Pct': majority_touching_pct,
        'Adjacency_Density_Pct': adjacency_density_pct,
        'Kissing_Percentage': minority_touching_pct,
        'Total_Majority_Clusters': len(majority_clusters),
        'Total_Minority_Clusters': len(minority_clusters),
        'Touching_Minority_Clusters': touching_minority,
        'Touching_Majority_Clusters': touching_majority,
        'Touching_Cluster_Pairs': touching_pairs,
        'Total_Endo_Clusters': n_endo,
        'Total_Meso_Clusters': n_meso,
        'Total_Possible_Endo_Meso_Pairs': possible_pairs
    }

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_proximity_analysis(base_path, exp_label, output_dir):
    records = []

    all_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    csv_files = filter_first_3_organoids(all_files)
    print(f"Processing {len(csv_files)} organoids (first 3 per replicate/condition from {len(all_files)} total)...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            dox = int(file_name.split('dox')[0])
            replicate = os.path.basename(os.path.dirname(file_path))

            result = calculate_cluster_kissing(df)
            if result:
                result.update({
                    'Dox_Concentration': dox,
                    'Replicate': replicate,
                    'File': file_name
                })
                records.append(result)
                print(
                    f"  {file_name}: minority {result['Touching_Minority_Clusters']}/{result['Total_Minority_Clusters']} "
                    f"({result['Minority_Touching_Pct']:.1f}%), majority {result['Touching_Majority_Clusters']}/{result['Total_Majority_Clusters']} "
                    f"({result['Majority_Touching_Pct']:.1f}%), adjacency density {result['Adjacency_Density_Pct']:.1f}%"
                )

        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            continue

    df_results = pd.DataFrame(records)

    if df_results.empty:
        print("No data to plot!")
        return

    # ==============================================================================
    # VISUALIZATION
    # ==============================================================================
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Color palette for lineages
    lineage_palette = {'Endo': '#d62728', 'Meso': '#1fb471'}

    # Panel 1: Three touching-method definitions
    panel1_cols = {
        'Minority_Touching_Pct': 'Minority in contact / total minority',
        'Majority_Touching_Pct': 'Majority in contact / total majority',
        'Adjacency_Density_Pct': 'sum(A_endo_meso) / (dim(endo)*dim(meso))'
    }
    df_panel1 = df_results.melt(
        id_vars=['Dox_Concentration', 'Replicate', 'File', 'Minority_Lineage', 'Majority_Lineage'],
        value_vars=list(panel1_cols.keys()),
        var_name='Method_Key',
        value_name='Touching_Percentage'
    )
    df_panel1['Method'] = df_panel1['Method_Key'].map(panel1_cols)

    sns.lineplot(data=df_panel1, x='Dox_Concentration', y='Touching_Percentage',
                 hue='Method',
                 marker='o', errorbar=('ci', 95), linewidth=2.5, ax=ax1)
    ax1.set_title("Cluster Interface Touching %\n(Three Definitions)",
                  fontweight='bold')
    ax1.set_xscale('symlog', linthresh=10)
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Dox Concentration (ng/mL)")
    ax1.set_ylabel("Touching Percentage (%)")
    ax1.legend(title="Method", loc='best')

    # Panel 2: Which lineage is minority (by cluster count)
    # Count how often each lineage has fewer clusters per organoid
    summary_data = []
    for dox in df_results['Dox_Concentration'].unique():
        dox_data = df_results[df_results['Dox_Concentration'] == dox]

        endo_minority = len(dox_data[dox_data['Minority_Lineage'] == 'Endo'])
        meso_minority = len(dox_data[dox_data['Minority_Lineage'] == 'Meso'])

        total = endo_minority + meso_minority
        if total > 0:
            summary_data.append({
                'Dox_Concentration': dox,
                'Lineage': 'Endo',
                'Minority_Frequency': (endo_minority / total) * 100
            })
            summary_data.append({
                'Dox_Concentration': dox,
                'Lineage': 'Meso',
                'Minority_Frequency': (meso_minority / total) * 100
            })

    df_summary = pd.DataFrame(summary_data)

    if not df_summary.empty:
        sns.lineplot(data=df_summary, x='Dox_Concentration', y='Minority_Frequency',
                     hue='Lineage', palette=lineage_palette,
                     marker='s', linewidth=2.5, ax=ax2)
        ax2.set_title("Cluster Count Minority: % of Organoids Where\nLineage Has Fewer Clusters",
                      fontweight='bold')
        ax2.set_xscale('symlog', linthresh=10)
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Dox Concentration (ng/mL)")
        ax2.set_ylabel("% Organoids Where Lineage is Minority")
        ax2.legend(title="Lineage", loc='best')

    plt.suptitle(f"{exp_label} - Cluster Proximity Analysis (Threshold: {PROXIMITY_THRESHOLD}μm)",
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save figure
    output_file = os.path.join(output_dir, f"{exp_label}_Cluster_Proximity_Kissing_Analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_file}")
    plt.close()

    # Save data to CSVs (raw + plot-ready)
    csv_raw = os.path.join(output_dir, f"{exp_label}_Cluster_Proximity_Organoid_Level.csv")
    csv_panel1 = os.path.join(output_dir, f"{exp_label}_Cluster_Proximity_Panel1_Methods.csv")
    csv_panel2 = os.path.join(output_dir, f"{exp_label}_Cluster_Proximity_Panel2_Minority_Frequency.csv")
    csv_legacy = os.path.join(output_dir, f"{exp_label}_Cluster_Proximity_Data.csv")

    df_results.to_csv(csv_raw, index=False)
    df_panel1.to_csv(csv_panel1, index=False)
    df_summary.to_csv(csv_panel2, index=False)
    df_results.to_csv(csv_legacy, index=False)  # Backward-compatible filename

    print(f"✓ Data saved: {csv_raw}")
    print(f"✓ Data saved: {csv_panel1}")
    print(f"✓ Data saved: {csv_panel2}")
    print(f"✓ Data saved (legacy): {csv_legacy}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster Proximity ("Kissing") Analysis')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Which experiment dataset to analyze')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')

    args = parser.parse_args()

    # Get dataset path
    mapped_path = DATASET_MAP[args.experiment]
    base_path = mapped_path if os.path.isabs(mapped_path) else os.path.join(SCRIPT_DIR, mapped_path)

    # Create output directory
    output_subdir = f"{args.experiment}_cluster_proximity"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"CLUSTER PROXIMITY (KISSING) ANALYSIS")
    print(f"Dataset: {args.experiment} ({base_path})")
    print(f"Proximity Threshold: {PROXIMITY_THRESHOLD} μm")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    run_proximity_analysis(base_path, args.experiment, output_path)
