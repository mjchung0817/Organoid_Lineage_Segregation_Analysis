import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import re
from scipy.spatial import ConvexHull, KDTree

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

RADIUS = 50.0
VIEW_WINDOW = 800
Z_RADIUS = 20.0  # +/- 20 microns (Depth Cueing)
DOT_SIZE = 4

# ==============================================================================
# HELPER: FILTER TO FIRST 3 ORGANOIDS PER REPLICATE PER CONDITION
# ==============================================================================
def filter_first_3_organoids(file_list):
    """
    Group files by replicate and condition, keep only first 3 organoids numerically.
    Filename pattern: {dox}dox_{sample}_{organoid_num}.csv
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
# HELPER: COLUMN STANDARDIZATION
# ==============================================================================
def standardize_columns(df):
    renames = {col: 'Global ' + col for col in ['X', 'Y', 'Z'] if col in df.columns}
    return df.rename(columns=renames)

# ==============================================================================
# HELPER: CALCULATE LOCAL MIXING
# ==============================================================================
def calculate_local_mixing(df, radius=50.0):
    coords = df[['Global X', 'Global Y']].values
    if len(coords) == 0: return np.array([])
    tree = KDTree(coords)
    indices_list = tree.query_ball_point(coords, r=radius)
    # Detect lineage column typo automatically
    target_col = 'cell_type_dapi_adusted' if 'cell_type_dapi_adusted' in df.columns else 'cell_type_dapi_adjusted'
    cell_types = df[target_col].values

    local_scores = []
    for i, neighbor_indices in enumerate(indices_list):
        if len(neighbor_indices) < 2:
            local_scores.append(0.0); continue
        my_type = cell_types[i]
        neighbor_types = cell_types[neighbor_indices]
        foreign_count = np.sum(neighbor_types != my_type)
        total_neighbors = len(neighbor_indices) - 1
        local_scores.append(foreign_count / total_neighbors if total_neighbors > 0 else 0.0)
    return np.array(local_scores)

# ==============================================================================
# MAIN VISUALIZATION FUNCTION
# ==============================================================================
def visualize_organoids(base_path, output_dir, exp_label):
    """Create 3-panel visualizations for each dox level"""
    all_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    files_list = filter_first_3_organoids(all_files)
    export_records = []

    # Group by dox concentration
    dox_groups = {}
    for fpath in files_list:
        fname = os.path.basename(fpath)
        dox_match = re.search(r'(\d+)dox', fname)
        if dox_match:
            dox = int(dox_match.group(1))
            if dox not in dox_groups:
                dox_groups[dox] = []
            dox_groups[dox].append(fpath)

    base_palette = {2.0: '#d62728', 3.0: "#1fb471"}  # Endo=Red, Meso=Green

    for dox in sorted(dox_groups.keys()):
        print(f"Processing {dox}ng/mL...")
        for fpath in dox_groups[dox]:
            fname = os.path.basename(fpath)
            replicate = os.path.basename(os.path.dirname(fpath))
            df = pd.read_csv(fpath)
            df = standardize_columns(df)

            # Simple Z-Slice Selection for Baseline Visualization
            z_center = df['Global Z'].median()
            df_slice = df[(df['Global Z'] >= z_center - Z_RADIUS) &
                          (df['Global Z'] <= z_center + Z_RADIUS)].copy()

            if not df_slice.empty:
                # Center the organoid for the VIEW_WINDOW
                df_slice['Global X'] -= df_slice['Global X'].mean()
                df_slice['Global Y'] -= df_slice['Global Y'].mean()
                target_col = 'cell_type_dapi_adusted' if 'cell_type_dapi_adusted' in df_slice.columns else 'cell_type_dapi_adjusted'
                df_slice = df_slice[df_slice[target_col].isin([2.0, 3.0])].copy()
                if df_slice.empty:
                    continue
                df_slice['local_mixing'] = calculate_local_mixing(df_slice, radius=RADIUS)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Anatomy Panel
                colors = df_slice[target_col].map(lambda x: base_palette.get(x, '#d3d3d3'))
                axes[0].scatter(df_slice['Global X'], df_slice['Global Y'], c=colors, s=DOT_SIZE, alpha=0.6)
                axes[0].set_title(f"Anatomy ({dox} ng/mL)")

                # Mixing Panels (Meso and Endo)
                # Use YlOrRd colormap: Yellow (low NMS) -> Red (high NMS)
                for i, (lin_id, lin_name) in enumerate([(3.0, "Mesoderm"), (2.0, "Endoderm")]):
                    sub = df_slice[df_slice[target_col] == lin_id]
                    norm = plt.Normalize(0, 1.0)
                    sc = axes[i+1].scatter(sub['Global X'], sub['Global Y'], c=sub['local_mixing'],
                                           cmap='YlOrRd', norm=norm, s=DOT_SIZE)
                    plt.colorbar(sc, ax=axes[i+1], label='Mixing Score')
                    axes[i+1].set_title(f"{lin_name} Mixing")

                for ax in axes:
                    ax.set_xlim(-VIEW_WINDOW, VIEW_WINDOW); ax.set_ylim(-VIEW_WINDOW, VIEW_WINDOW)
                    ax.set_aspect('equal')

                plt.tight_layout()

                # Save with unique filename
                output_file = os.path.join(output_dir, f"{exp_label}_{dox}ng_{fname.replace('.csv', '.png')}")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {output_file}")

                export_df = df_slice[['Global X', 'Global Y', 'Global Z', target_col, 'local_mixing']].copy()
                export_df = export_df.rename(columns={target_col: 'Cell_Type_ID'})
                export_df['Cell_Type'] = export_df['Cell_Type_ID'].map({2.0: 'Endo', 3.0: 'Meso'})
                export_df['Experiment'] = exp_label
                export_df['Dox_Concentration'] = dox
                export_df['Replicate'] = replicate
                export_df['File'] = fname
                export_df['Z_Center'] = z_center
                export_records.append(export_df)

    if export_records:
        export_all = pd.concat(export_records, ignore_index=True)
        csv_output = os.path.join(output_dir, f"{exp_label}_Z_Biopsy_LocalMixing_Plot_Data.csv")
        export_all.to_csv(csv_output, index=False)
        print(f"✓ Data saved: {csv_output}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Z-Biopsy Visualization with NMS Heatmap')
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
    output_subdir = f"{args.experiment}_z_biopsy_visualization"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Z-BIOPSY VISUALIZATION WITH NMS HEATMAP")
    print(f"Dataset: {args.experiment} ({base_path})")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    visualize_organoids(base_path, output_path, args.experiment)

    print(f"\n✓ All visualizations saved to: {output_path}")
