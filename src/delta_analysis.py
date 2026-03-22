import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import argparse
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

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
RADIUS, EPSILON, MIN_SAMPLES = 100.0, 30.0, 20

ERRORBAR_LABELS = {
    'sd': 'SD',
    'se': 'SE',
    'ci95': '95% CI',
    'none': 'No Error Bar'
}

# ==============================================================================
# HELPER: FILTER TO FIRST 3 ORGANOIDS PER REPLICATE PER CONDITION
# ==============================================================================
def filter_first_3_organoids(file_list):
    """
    Group files by replicate, dox, and condition, then keep only the first
    3 organoids numerically within each group.
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
# ANALYSIS ENGINE (Captures Count, Size, Separation, and NMS)
# ==============================================================================
def get_spatial_metrics(df):
    target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
    if target_col not in df.columns: return None

    # Restrict analysis to Endo/Meso only (no triple-negative/pluripotent tracking)
    lineage_df = df[df[target_col].isin([2.0, 3.0])].copy()
    endo_subset = lineage_df[lineage_df[target_col] == 2.0]
    meso_subset = lineage_df[lineage_df[target_col] == 3.0]
    results = {}

    # --- Metric A: DBSCAN Clusters (Count, Size, & Separation) ---
    for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
        if len(subset) >= MIN_SAMPLES:
            db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(subset[['X', 'Y', 'Z']])
            labels = db.labels_
            unique_clusters = [l for l in np.unique(labels) if l != -1]

            results[f'count_{name}'] = len(unique_clusters)
            if len(unique_clusters) > 0:
                results[f'avg_size_{name}'] = np.mean([len(subset[labels == l]) for l in unique_clusters])

                # Calculate inter-cluster separation (distance between cluster centroids)
                if len(unique_clusters) > 1:
                    coords = subset[['X', 'Y', 'Z']].values
                    centroids = np.array([coords[labels == l].mean(axis=0) for l in unique_clusters])
                    pairwise_dists = pdist(centroids)
                    results[f'separation_{name}'] = np.mean(pairwise_dists)
                else:
                    # Only one cluster, no separation to measure
                    results[f'separation_{name}'] = 0
            else:
                results[f'avg_size_{name}'] = 0
                results[f'separation_{name}'] = 0
        else:
            results[f'count_{name}'], results[f'avg_size_{name}'], results[f'separation_{name}'] = 0, 0, 0

    # --- Metric B: NMS (Endo/Meso only) ---
    # NMS (Ref: Endoderm vs Mesoderm)
    non_endo_subset = lineage_df[lineage_df[target_col] != 2.0]  # Meso only after filtering
    if len(endo_subset) > 10 and len(non_endo_subset) > 10:
        e_coords = endo_subset[['X', 'Y', 'Z']].values
        non_e_coords = non_endo_subset[['X', 'Y', 'Z']].values
        e_tree = KDTree(e_coords)
        non_e_tree = KDTree(non_e_coords)

        n_non_e_around_e = sum([len(n) for n in non_e_tree.query_ball_point(e_coords, r=RADIUS)])
        n_e_around_e = sum([len(n)-1 for n in e_tree.query_ball_point(e_coords, r=RADIUS)])
        if n_e_around_e > 0:
            results['nms_Endo'] = (n_non_e_around_e / n_e_around_e) / (len(non_endo_subset) / len(endo_subset))

    # NMS (Ref: Mesoderm vs Endoderm)
    non_meso_subset = lineage_df[lineage_df[target_col] != 3.0]  # Endo only after filtering
    if len(meso_subset) > 10 and len(non_meso_subset) > 10:
        m_coords = meso_subset[['X', 'Y', 'Z']].values
        non_m_coords = non_meso_subset[['X', 'Y', 'Z']].values
        m_tree = KDTree(m_coords)
        non_m_tree = KDTree(non_m_coords)

        n_non_m_around_m = sum([len(n) for n in non_m_tree.query_ball_point(m_coords, r=RADIUS)])
        n_m_around_m = sum([len(n)-1 for n in m_tree.query_ball_point(m_coords, r=RADIUS)])
        if n_m_around_m > 0:
            results['nms_Meso'] = (n_non_m_around_m / n_m_around_m) / (len(non_meso_subset) / len(meso_subset))

    return results


def resolve_errorbar(errorbar_mode):
    if errorbar_mode == 'sd':
        return 'sd'
    if errorbar_mode == 'se':
        return 'se'
    if errorbar_mode == 'ci95':
        return ('ci', 95)
    if errorbar_mode == 'none':
        return None
    raise ValueError(f"Unsupported errorbar mode: {errorbar_mode}")

# ==============================================================================
# MAIN COMPARISON FUNCTION
# ==============================================================================
def run_delta_analysis(baseline_path, baseline_label, treatment_path, treatment_label, output_dir,
                       errorbar_mode='sd', save_nms_only=True):
    # Data scraping & audit
    all_data = []
    for exp_tag, base_path in [(baseline_label, baseline_path), (treatment_label, treatment_path)]:
        all_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
        files = filter_first_3_organoids(all_files)
        print(f"{exp_tag}: Processing {len(files)} organoids (first 3 per replicate/condition from {len(all_files)} total)")

        for f in files:
            fname = os.path.basename(f)
            dose = int(re.search(r"(\d+)dox", fname).group(1)) if re.search(r"(\d+)dox", fname) else 0
            if dose in [100, 1000]:
                cond = re.search(r"\+(.+?)_", fname).group(1).upper() if re.search(r"\+(.+?)_", fname) else "BASAL"
                m = get_spatial_metrics(pd.read_csv(f))
                if m:
                    m.update({'Dose': dose, 'Condition': cond, 'Exp': exp_tag, 'Rep': os.path.basename(os.path.dirname(f))})
                    all_data.append(m)

    raw_df = pd.DataFrame(all_data)
    if raw_df.empty:
        print("No data found for selected baseline/treatment and dose filters.")
        return

    # Calculation: Organoid-Level Deltas
    final_comparison_list = []
    metrics_to_plot = ['nms_Endo', 'nms_Meso', 'avg_size_Endo', 'avg_size_Meso', 'count_Endo', 'count_Meso', 'separation_Endo', 'separation_Meso']

    for dose in [100, 1000]:
        baseline_ref = raw_df[(raw_df['Exp'] == baseline_label) & (raw_df['Dose'] == dose)].mean(numeric_only=True)
        treatment_data = raw_df[(raw_df['Exp'] == treatment_label) & (raw_df['Dose'] == dose)]

        for _, row in treatment_data.iterrows():
            entry = {'Dose': dose, 'Condition': row['Condition'], 'Rep': row['Rep']}
            for m in metrics_to_plot:
                v_ref = baseline_ref.get(m, 0)
                # For NMS metrics, calculate ABSOLUTE difference
                if 'nms' in m.lower():
                    entry[f'Abs_{m}'] = row[m] - v_ref
                # For other metrics, calculate RELATIVE % change
                else:
                    entry[f'Rel_{m}'] = ((row[m] - v_ref) / v_ref * 100) if v_ref > 0 else 0
            final_comparison_list.append(entry)

    comp_df = pd.DataFrame(final_comparison_list)
    if comp_df.empty:
        print("No treatment organoids available to compute deltas.")
        return

    # ==============================================================================
    # DIAGNOSTIC OUTPUT
    # ==============================================================================
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC: {baseline_label} vs {treatment_label}")
    print(f"{'='*80}\n")
    print(f"Baseline ({baseline_label}) Average Values:")
    for dose in [100, 1000]:
        baseline_ref = raw_df[(raw_df['Exp'] == baseline_label) & (raw_df['Dose'] == dose)].mean(numeric_only=True)
        print(f"\n  {dose}ng/mL:")
        print(f"    Endo - Count: {baseline_ref.get('count_Endo', 0):.1f}, Size: {baseline_ref.get('avg_size_Endo', 0):.1f}, Sep: {baseline_ref.get('separation_Endo', 0):.1f}")
        print(f"    Meso - Count: {baseline_ref.get('count_Meso', 0):.1f}, Size: {baseline_ref.get('avg_size_Meso', 0):.1f}, Sep: {baseline_ref.get('separation_Meso', 0):.1f}")

    # ==============================================================================
    # VISUALIZATION: 4 PANELS (NMS, SIZE, COUNT, SEPARATION)
    # ==============================================================================
    # Transform to long format
    long_df = pd.melt(comp_df,
                      id_vars=['Dose', 'Condition', 'Rep'],
                      value_vars=[col for col in comp_df.columns if col.startswith(('Abs_', 'Rel_'))],
                      var_name='Metric', value_name='value')

    # Categorize metrics properly
    long_df['Lineage'] = long_df['Metric'].str.split('_').str[-1]
    long_df['Type'] = long_df['Metric'].apply(lambda x: 'nms' if 'nms' in x else
                                                        'avg_size' if 'avg_size' in x else
                                                        'count' if 'count' in x else 'separation')

    # Save data for external figure curation
    raw_csv = os.path.join(output_dir, f"{baseline_label}_vs_{treatment_label}_raw_metrics_organoid_level.csv")
    comp_csv = os.path.join(output_dir, f"{baseline_label}_vs_{treatment_label}_delta_metrics_organoid_level.csv")
    long_csv = os.path.join(output_dir, f"{baseline_label}_vs_{treatment_label}_delta_plot_data_long.csv")
    raw_df.to_csv(raw_csv, index=False)
    comp_df.to_csv(comp_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    print(f"✓ Data saved: {raw_csv}")
    print(f"✓ Data saved: {comp_csv}")
    print(f"✓ Data saved: {long_csv}")

    # Visualization Setup
    sns.set_context("talk")
    palette = {'Endo': '#d62728', 'Meso': '#1fb471'}
    errorbar_spec = resolve_errorbar(errorbar_mode)
    errorbar_label = ERRORBAR_LABELS[errorbar_mode]

    # Create a marker style mapping for replicates (for the stripplot overlay)
    unique_reps = sorted(comp_df['Rep'].unique())
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X']  # circle, square, triangle, diamond, etc.
    rep_markers = dict(zip(unique_reps, marker_styles[:len(unique_reps)]))

    # Print replicate marker mapping
    print(f"\n{'='*80}")
    print("REPLICATE MARKER MAPPING")
    print(f"{'='*80}")
    for rep, marker in rep_markers.items():
        print(f"{rep}: {marker}")
    print(f"{'='*80}\n")

    for dose in [100, 1000]:
        # Expanded to (1, 4) to accommodate the new Inter-Cluster Separation panel
        fig, axes = plt.subplots(1, 4, figsize=(40, 10))
        dose_df = comp_df[comp_df['Dose'] == dose]
        dose_long_df = long_df[long_df['Dose'] == dose]

        # Iterate through the four main analysis themes
        for i, m_type in enumerate(['nms', 'avg_size', 'count', 'separation']):
            label_map = {
                'nms': 'NMS (Absolute Δ)',
                'avg_size': 'Cluster Size (% Change)',
                'count': 'Cluster Count (% Change)',
                'separation': 'Inter-Cluster Separation (% Change)'
            }

            metric_df = dose_long_df[dose_long_df['Type'] == m_type].copy()
            metric_df = metric_df.dropna(subset=['value'])
            if metric_df.empty:
                axes[i].set_visible(False)
                continue

            conditions = list(metric_df['Condition'].dropna().unique())

            # 1. Barplot for mean % change
            sns.barplot(data=metric_df, x='Condition', y='value', hue='Lineage',
                        palette=palette, ax=axes[i], capsize=.1, errorbar=errorbar_spec)

            # 1.5. Add mean markers (diamonds) to verify bar heights
            # Position diamonds to match barplot: Endo LEFT (-0.2), Meso RIGHT (+0.2)
            for condition_idx, condition in enumerate(conditions):
                for lineage in ['Endo', 'Meso']:
                    subset = metric_df[(metric_df['Condition'] == condition) & (metric_df['Lineage'] == lineage)]
                    if not subset.empty:
                        mean_val = subset['value'].mean()
                        x_pos = condition_idx + (-0.2 if lineage == 'Endo' else 0.2)  # Correct: Endo=-0.2, Meso=+0.2
                        axes[i].scatter([x_pos], [mean_val], marker='D', s=120,
                                      color='black', edgecolors='white', linewidths=2,
                                      zorder=10, label='_nolegend_')

            # 2. Stripplot to show individual organoids - SHAPE-CODED BY REPLICATE
            # Plot Endo and Meso separately to maintain dodge positioning
            # Barplot puts Endo on LEFT (negative offset) and Meso on RIGHT (positive offset)
            for lineage_idx, lineage in enumerate(['Endo', 'Meso']):
                lineage_df = metric_df[metric_df['Lineage'] == lineage]
                for rep in unique_reps:
                    rep_df = lineage_df[lineage_df['Rep'] == rep]
                    if not rep_df.empty:
                        # Calculate dodge offset to MATCH barplot positions
                        x_vals = [conditions.index(cond) for cond in rep_df['Condition']]
                        dodge_offset = -0.2 if lineage == 'Endo' else 0.2  # Flipped to match barplot
                        x_jittered = [x + dodge_offset + np.random.uniform(-0.08, 0.08) for x in x_vals]
                        axes[i].scatter(x_jittered, rep_df['value'].values,
                                      color=palette[lineage], marker=rep_markers[rep],
                                      s=80, alpha=0.7,
                                      edgecolors='white', linewidths=0.5, zorder=3)

            # Panel Formatting
            axes[i].set_title(f"Metric: {label_map[m_type]}", fontweight='bold', pad=20)
            axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
            axes[i].set_ylabel("Δ from Baseline" if m_type == 'nms' else "% Change from Baseline")
            axes[i].set_xlabel("")

        # Add replicate marker legend to the last panel
        from matplotlib.lines import Line2D
        rep_handles = [Line2D([0], [0], marker=rep_markers[rep], color='gray', linestyle='None',
                              markersize=8, label=rep) for rep in unique_reps]
        axes[-1].legend(handles=axes[-1].get_legend_handles_labels()[0] + rep_handles,
                        loc='upper right', fontsize=9)

        plt.suptitle(f"{treatment_label} vs {baseline_label} | {dose}ng/mL Dox | Bars = mean ± {errorbar_label}",
                     fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        # Save figure
        output_file = os.path.join(output_dir, f"{baseline_label}_vs_{treatment_label}_{dose}ng_4Panel_Delta_Analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()

        if save_nms_only:
            nms_df = dose_long_df[dose_long_df['Type'] == 'nms'].copy()
            nms_df = nms_df.dropna(subset=['value'])
            if not nms_df.empty:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                conditions = list(nms_df['Condition'].dropna().unique())

                sns.barplot(data=nms_df, x='Condition', y='value', hue='Lineage',
                            palette=palette, ax=ax, capsize=.1, errorbar=errorbar_spec)

                for condition_idx, condition in enumerate(conditions):
                    for lineage in ['Endo', 'Meso']:
                        subset = nms_df[(nms_df['Condition'] == condition) & (nms_df['Lineage'] == lineage)]
                        if not subset.empty:
                            mean_val = subset['value'].mean()
                            x_pos = condition_idx + (-0.2 if lineage == 'Endo' else 0.2)
                            ax.scatter([x_pos], [mean_val], marker='D', s=120,
                                       color='black', edgecolors='white', linewidths=2,
                                       zorder=10, label='_nolegend_')

                for lineage in ['Endo', 'Meso']:
                    lineage_df = nms_df[nms_df['Lineage'] == lineage]
                    for rep in unique_reps:
                        rep_df = lineage_df[lineage_df['Rep'] == rep]
                        if not rep_df.empty:
                            x_vals = [conditions.index(cond) for cond in rep_df['Condition']]
                            dodge_offset = -0.2 if lineage == 'Endo' else 0.2
                            x_jittered = [x + dodge_offset + np.random.uniform(-0.08, 0.08) for x in x_vals]
                            ax.scatter(x_jittered, rep_df['value'].values,
                                       color=palette[lineage], marker=rep_markers[rep],
                                       s=80, alpha=0.7, edgecolors='white',
                                       linewidths=0.5, zorder=3)

                rep_handles = [Line2D([0], [0], marker=rep_markers[rep], color='gray',
                                      linestyle='None', markersize=8, label=rep)
                               for rep in unique_reps]
                lineage_handles, lineage_labels = ax.get_legend_handles_labels()
                ax.legend(handles=lineage_handles + rep_handles,
                          labels=lineage_labels + list(unique_reps),
                          loc='upper right', fontsize=9)
                ax.set_title("Metric: NMS (Absolute Δ)", fontweight='bold', pad=20)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.set_ylabel("Δ from Baseline")
                ax.set_xlabel("")
                plt.suptitle(f"{treatment_label} vs {baseline_label} | {dose}ng/mL Dox | NMS Only | Bars = mean ± {errorbar_label}",
                             fontsize=18, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0.02, 1, 0.95])
                nms_output_file = os.path.join(output_dir, f"{baseline_label}_vs_{treatment_label}_{dose}ng_NMS_Delta_Analysis.png")
                plt.savefig(nms_output_file, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {nms_output_file}")
                plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delta Analysis: % Change Comparison Between Experiments')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Baseline experiment dataset')
    parser.add_argument('--treatment', type=str, required=True,
                        choices=['exp1', 'exp2_high_cn', 'exp2_low_cn', 'exp3'],
                        help='Treatment experiment dataset')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results)')
    parser.add_argument('--errorbar-mode', type=str, default='sd',
                        choices=['sd', 'se', 'ci95', 'none'],
                        help='Bar error bar type: sd, se, ci95, or none (default: sd)')
    parser.add_argument('--no-save-nms-only', action='store_true',
                        help='Disable saving the dedicated NMS-only delta plot')

    args = parser.parse_args()

    # Get dataset paths
    baseline_mapped = DATASET_MAP[args.baseline]
    treatment_mapped = DATASET_MAP[args.treatment]
    baseline_path = baseline_mapped if os.path.isabs(baseline_mapped) else os.path.join(SCRIPT_DIR, baseline_mapped)
    treatment_path = treatment_mapped if os.path.isabs(treatment_mapped) else os.path.join(SCRIPT_DIR, treatment_mapped)

    # Create output directory
    output_subdir = f"{args.baseline}_vs_{args.treatment}_delta_analysis"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DELTA ANALYSIS: % CHANGE COMPARISON")
    print(f"Baseline: {args.baseline} ({baseline_path})")
    print(f"Treatment: {args.treatment} ({treatment_path})")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    run_delta_analysis(
        baseline_path,
        args.baseline,
        treatment_path,
        args.treatment,
        output_path,
        errorbar_mode=args.errorbar_mode,
        save_nms_only=not args.no_save_nms_only
    )

    print(f"\n✓ All delta analysis plots saved to: {output_path}")
