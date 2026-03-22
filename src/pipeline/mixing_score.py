import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import argparse
from scipy.spatial import KDTree

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

# Multi-radius analysis parameters
RADII = [30.0, 50.0, 100.0]

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
# METRIC ENGINE (UPDATED: Each lineage vs ALL other cell types)
# ==============================================================================
def calculate_metrics(df, radius=100.0):
    target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
    if target_col not in df.columns: return None

    # Restrict analysis to Endo/Meso only (no triple-negative/pluripotent tracking)
    lineage_df = df[df[target_col].isin([2.0, 3.0])].copy()
    endo_subset = lineage_df[lineage_df[target_col] == 2.0]           # Endoderm
    meso_subset = lineage_df[lineage_df[target_col] == 3.0]           # Mesoderm
    non_endo_subset = lineage_df[lineage_df[target_col] != 2.0]       # Meso only after filtering
    non_meso_subset = lineage_df[lineage_df[target_col] != 3.0]       # Endo only after filtering

    if len(endo_subset) < 10 or len(non_endo_subset) < 10:
        nms_e = None
    else:
        nms_e = 0

    if len(meso_subset) < 10 or len(non_meso_subset) < 10:
        nms_m = None
    else:
        nms_m = 0

    # Calculate NMS (Ref: Endoderm vs ALL other cell types)
    if len(endo_subset) >= 10 and len(non_endo_subset) >= 10:
        e_coords = endo_subset[['X', 'Y', 'Z']].values
        non_e_coords = non_endo_subset[['X', 'Y', 'Z']].values
        e_tree = KDTree(e_coords)
        non_e_tree = KDTree(non_e_coords)

        n_non_e_around_e = sum([len(n) for n in non_e_tree.query_ball_point(e_coords, r=radius)])
        n_e_around_e = sum([len(n)-1 for n in e_tree.query_ball_point(e_coords, r=radius)])
        nms_e = (n_non_e_around_e / n_e_around_e) / (len(non_endo_subset) / len(endo_subset)) if n_e_around_e > 0 else 0

    # Calculate NMS (Ref: Mesoderm vs ALL other cell types)
    if len(meso_subset) >= 10 and len(non_meso_subset) >= 10:
        m_coords = meso_subset[['X', 'Y', 'Z']].values
        non_m_coords = non_meso_subset[['X', 'Y', 'Z']].values
        m_tree = KDTree(m_coords)
        non_m_tree = KDTree(non_m_coords)

        n_non_m_around_m = sum([len(n) for n in non_m_tree.query_ball_point(m_coords, r=radius)])
        n_m_around_m = sum([len(n)-1 for n in m_tree.query_ball_point(m_coords, r=radius)])
        nms_m = (n_non_m_around_m / n_m_around_m) / (len(non_meso_subset) / len(meso_subset)) if n_m_around_m > 0 else 0

    # Return None if either metric couldn't be calculated
    if nms_e is None or nms_m is None:
        return None

    return {'NMS_Endo': nms_e, 'NMS_Meso': nms_m}

# ==============================================================================
# CROSS-EXPERIMENT DATA RETRIEVAL (Multi-Radius)
# ==============================================================================
def run_comparison(baseline_path, baseline_label, treatment_path, treatment_label, output_dir):
    # Process data for each radius value
    for RADIUS in RADII:
        print(f"\n{'='*80}")
        print(f"PROCESSING RADIUS: {RADIUS} μm")
        print(f"{'='*80}\n")

        results = []

        # Process baseline experiment
        baseline_files = glob.glob(os.path.join(baseline_path, "**/*.csv"), recursive=True)
        baseline_files = filter_first_3_organoids(baseline_files)
        print(f"Baseline ({baseline_label}): {len(baseline_files)} organoids")

        for f in baseline_files:
            fname = os.path.basename(f)
            dose = int(re.search(r"(\d+)dox", fname).group(1)) if re.search(r"(\d+)dox", fname) else 0

            # We only need these specific doses
            if dose in [0, 100, 1000]:
                m = calculate_metrics(pd.read_csv(f), radius=RADIUS)
                if m:
                    m.update({'Dose': dose, 'Condition': 'CTRL', 'Replicate': os.path.basename(os.path.dirname(f))})
                    results.append(m)

        # Process treatment experiment
        treatment_files = glob.glob(os.path.join(treatment_path, "**/*.csv"), recursive=True)
        treatment_files = filter_first_3_organoids(treatment_files)
        print(f"Treatment ({treatment_label}): {len(treatment_files)} organoids")

        for f in treatment_files:
            fname = os.path.basename(f)
            dose = int(re.search(r"(\d+)dox", fname).group(1)) if re.search(r"(\d+)dox", fname) else 0

            # Determine Condition from filename
            cond_match = re.search(r"\+(.+?)_", fname)
            cond = cond_match.group(1).upper() if cond_match else "BASAL"

            # We only need these specific doses
            if dose in [0, 100, 1000]:
                m = calculate_metrics(pd.read_csv(f), radius=RADIUS)
                if m:
                    m.update({'Dose': dose, 'Condition': cond, 'Replicate': os.path.basename(os.path.dirname(f))})
                    results.append(m)

        df_plot = pd.DataFrame(results)
        if df_plot.empty:
            print(f"No valid Endo/Meso NMS data at radius {RADIUS} μm; skipping plots.")
            continue

        # Save organoid-level datapoints for this radius
        raw_csv = os.path.join(output_dir, f"NMS_Radius_{int(RADIUS)}um_Organoid_Level.csv")
        df_plot.to_csv(raw_csv, index=False)
        print(f"✓ Data saved: {raw_csv}")

        # Determine category order based on available conditions
        all_conditions = df_plot['Condition'].unique()
        category_order = ['CTRL']
        for cond in ['BMP4', 'WNT5A', 'BMP4+WNT5A', 'BASAL']:
            if cond in all_conditions:
                category_order.append(cond)

        # ==============================================================================
        # DOT PLOT VISUALIZATION (DISTINCT COLORS) - For this radius
        # ==============================================================================
        sns.set_context("talk", font_scale=1.1)
        # High-contrast palette: Black (0), Blue (100), Orange/Yellow (1000)
        custom_palette = {0: "#222222", 100: "#1f77b4", 1000: "#ffcc00"}

        for metric in ["NMS_Endo", "NMS_Meso"]:
            fig, ax = plt.subplots(figsize=(15, 8))

            # 1. Stripplot for individual organoids (with jitter)
            sns.stripplot(data=df_plot, x='Condition', y=metric, hue='Dose', palette=custom_palette,
                          order=category_order, dodge=True, alpha=0.6, size=9, ax=ax)

            # 2. Pointplot for Mean + Standard Error (SE)
            sns.pointplot(data=df_plot, x='Condition', y=metric, hue='Dose', palette=custom_palette,
                          order=category_order, dodge=0.55, join=False, errorbar='se',
                          markers="D", scale=1.2, ax=ax)

            # --- FORMATTING ---
            ax.set_title(f"Spatial Segregation: {metric.replace('_', ' ')} (Radius: {RADIUS}μm)\n{treatment_label} vs {baseline_label} Baseline (Each Lineage vs ALL Other Cell Types)", fontweight='bold')
            ax.set_ylabel("Normalized Mixing Score (NMS)")
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random Mixing')

            # Consolidate Legend
            handles, labels = ax.get_legend_handles_labels()
            # Ensure we only grab unique dose labels
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys(), title="Dox (ng/mL)", loc='upper right', frameon=False)

            sns.despine()
            plt.tight_layout()

            # Save with radius in filename
            output_filename = os.path.join(output_dir, f"NMS_{metric}_Radius_{int(RADIUS)}um.png")
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_filename}")
            plt.close()

        # Save long-form table used directly by plotting
        plot_long = df_plot.melt(
            id_vars=['Dose', 'Condition', 'Replicate'],
            value_vars=['NMS_Endo', 'NMS_Meso'],
            var_name='Metric',
            value_name='NMS'
        )
        plot_long_csv = os.path.join(output_dir, f"NMS_Radius_{int(RADIUS)}um_Plot_Data_Long.csv")
        plot_long.to_csv(plot_long_csv, index=False)
        print(f"✓ Data saved: {plot_long_csv}")

    print(f"\n{'='*80}")
    print(f"ALL RADII PROCESSED SUCCESSFULLY")
    print(f"{'='*80}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NMS Replicates Comparison (Multi-Radius)')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Baseline experiment dataset')
    parser.add_argument('--treatment', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Treatment experiment dataset')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')

    args = parser.parse_args()

    # Get dataset paths
    baseline_mapped = DATASET_MAP[args.baseline]
    treatment_mapped = DATASET_MAP[args.treatment]
    baseline_path = baseline_mapped if os.path.isabs(baseline_mapped) else os.path.join(PROJECT_ROOT, baseline_mapped)
    treatment_path = treatment_mapped if os.path.isabs(treatment_mapped) else os.path.join(PROJECT_ROOT, treatment_mapped)

    # Create output directory
    output_subdir = f"{args.baseline}_vs_{args.treatment}_nms_comparison"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"NMS REPLICATES COMPARISON (MULTI-RADIUS)")
    print(f"Baseline: {args.baseline} ({baseline_path})")
    print(f"Treatment: {args.treatment} ({treatment_path})")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    run_comparison(baseline_path, args.baseline, treatment_path, args.treatment, output_path)
