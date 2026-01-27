# 100 dox vs 1000 dox across different additive conditions in Exp 3

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# import re
# from sklearn.cluster import DBSCAN
# from scipy.spatial import KDTree

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================
# BASE_DIR = "data/GATA-HA-BMP4+Wnt5a_Ex3"
# EXP_NAME = os.path.basename(BASE_DIR)
# OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "Ex3_Final_SE_Plots")
# if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# # Use existing constants from your calibration
# RADIUS, EPSILON, MIN_SAMPLES = 100.0, 30.0, 20

# # ==============================================================================
# # ENGINE: DUAL-LINEAGE METRICS
# # ==============================================================================
# def get_spatial_metrics(df):
#     # Flexible column selection for the 'adjusted' vs 'adusted' typo
#     target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
#     if target_col not in df.columns: return None

#     endo_subset = df[df[target_col] == 2.0]
#     meso_subset = df[df[target_col] == 3.0]
#     results = {}

#     # 1. Cluster Counts (DBSCAN)
#     for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
#         if len(subset) >= MIN_SAMPLES:
#             db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(subset[['X', 'Y', 'Z']])
#             results[f'count_{name}'] = len([l for l in np.unique(db.labels_) if l != -1])
#         else:
#             results[f'count_{name}'] = 0

#     # 2. Dual-Lineage NMS
#     if len(endo_subset) > 10 and len(meso_subset) > 10:
#         e_coords, m_coords = endo_subset[['X', 'Y', 'Z']].values, meso_subset[['X', 'Y', 'Z']].values
#         e_tree, m_tree = KDTree(e_coords), KDTree(m_coords)

#         for ref, t_tree, r_coords, r_sub, t_sub in [('Endo', m_tree, e_coords, endo_subset, meso_subset),
#                                                    ('Meso', e_tree, m_coords, meso_subset, endo_subset)]:
#             n_target = sum([len(n) for n in t_tree.query_ball_point(r_coords, r=RADIUS)])
#             n_ref = sum([len(n)-1 for n in KDTree(r_coords).query_ball_point(r_coords, r=RADIUS)])
#             if n_ref > 0:
#                 results[f'nms_{ref}'] = (n_target / n_ref) / (len(t_sub) / len(r_sub))
#     return results

# # ==============================================================================
# # DATA AGGREGATION & RELATIVE CALCULATION
# # ==============================================================================
# all_raw_data = []
# file_list = glob.glob(os.path.join(BASE_DIR, "**/*.csv"), recursive=True)

# for f in file_list:
#     fname = os.path.basename(f)
#     dose_match = re.search(r"(\d+)dox", fname)
#     cond_match = re.search(r"\+(.+?)_", fname)
    
#     dose = int(dose_match.group(1)) if dose_match else 0
#     # Fix case-sensitivity issues
#     cond = cond_match.group(1).upper() if cond_match else "BASAL"
    
#     if dose in [100, 1000]:
#         m = get_spatial_metrics(pd.read_csv(f))
#         if m:
#             m.update({'Dose': dose, 'Condition': cond, 'Rep': os.path.basename(os.path.dirname(f))})
#             all_raw_data.append(m)

# raw_df = pd.DataFrame(all_raw_data)

# # Calculate Relative % Change per Replicate
# delta_list = []
# for (cond, rep), group in raw_df.groupby(['Condition', 'Rep']):
#     d100 = group[group['Dose'] == 100]
#     d1000 = group[group['Dose'] == 1000]
    
#     if not d100.empty and not d1000.empty:
#         v100_s, v1000_s = d100.iloc[0], d1000.iloc[0]
#         entry = {'Condition': cond, 'Rep': rep}
#         for metric in ['nms_Endo', 'nms_Meso', 'count_Endo', 'count_Meso']:
#             v1, v2 = v100_s.get(metric, 0), v1000_s.get(metric, 0)
#             entry[f'Rel_Delta_{metric}'] = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
#         delta_list.append(entry)

# final_delta_df = pd.DataFrame(delta_list)

# # ==============================================================================
# # VISUALIZATION: BAR PLOTS WITH SE BARS
# # ==============================================================================
# fig, axes = plt.subplots(1, 2, figsize=(22, 8))
# palette = {'Endo': '#d62728', 'Meso': '#1f77b4'}

# for i, m_type in enumerate(['nms', 'count']):
#     label = "Mixing Score (NMS)" if m_type == 'nms' else "Number of Clusters"
#     long_df = final_delta_df.melt(id_vars=['Condition', 'Rep'], 
#                                  value_vars=[f'Rel_Delta_{m_type}_Endo', f'Rel_Delta_{m_type}_Meso'])
#     long_df['Lineage'] = long_df['variable'].str.extract(r'([^_]+)$')
    
#     # Plotting with Standard Error (se)
#     sns.barplot(data=long_df, x='Condition', y='value', hue='Lineage', 
#                 palette=palette, ax=axes[i], capsize=.1, errorbar='se')
    
#     axes[i].set_title(f"Relative % Change in {label} (100ng vs 1000ng)", fontweight='bold')
#     axes[i].set_ylabel("% Change from Baseline (100ng)")
#     axes[i].axhline(0, color='black', linewidth=1)

# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, f"{EXP_NAME}_Relative_Deltas_SE.png"), dpi=300)
# plt.show()
# # ==============================================================================
# # ==============================================================================
# # ==============================================================================
# # ==============================================================================
# # ==============================================================================
# # ==============================================================================
# # ==============================================================================
# # ==============================================================================


# #Exp1 100 dox vs Exp3 100 dox and Exp 1 1000 dox vs Exp 3 1000 dox
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# import re
# from sklearn.cluster import DBSCAN
# from scipy.spatial import KDTree

# # ==============================================================================
# # 1. CONFIGURATION
# # ==============================================================================
# # Paths to both Experiment 1 and Experiment 3
# BASE_DIR_EX1 = "data/GATA-HA_Rep1-3_Ex1"
# BASE_DIR_EX3 = "data/GATA-HA-BMP4+Wnt5a_Ex3"

# OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR_EX3), "Ex1_vs_Ex3_Comparison")
# if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# # Analysis Constants
# RADIUS, EPSILON, MIN_SAMPLES = 100.0, 30.0, 20

# # ==============================================================================
# # 2. METRIC ENGINE
# # ==============================================================================
# def get_spatial_metrics(df):
#     target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
#     if target_col not in df.columns: return None

#     endo_subset = df[df[target_col] == 2.0]
#     meso_subset = df[df[target_col] == 3.0]
#     results = {}

#     # Cluster Counts
#     for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
#         if len(subset) >= MIN_SAMPLES:
#             db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(subset[['X', 'Y', 'Z']])
#             results[f'count_{name}'] = len([l for l in np.unique(db.labels_) if l != -1])
#         else:
#             results[f'count_{name}'] = 0
#             print(f"      [FILTER] {name} failed DBSCAN: only {len(endo_subset)} cells.")

#     # NMS (Ref: Endo and Meso)
#     if len(endo_subset) > 10 and len(meso_subset) > 10:
#         e_coords, m_coords = endo_subset[['X', 'Y', 'Z']].values, meso_subset[['X', 'Y', 'Z']].values
#         e_tree, m_tree = KDTree(e_coords), KDTree(m_coords)

#         for ref, t_tree, r_coords, r_sub, t_sub in [('Endo', m_tree, e_coords, endo_subset, meso_subset),
#                                                    ('Meso', e_tree, m_coords, meso_subset, endo_subset)]:
#             n_target = sum([len(n) for n in t_tree.query_ball_point(r_coords, r=RADIUS)])
#             n_ref = sum([len(n)-1 for n in KDTree(r_coords).query_ball_point(r_coords, r=RADIUS)])
#             if n_ref > 0:
#                 results[f'nms_{ref}'] = (n_target / n_ref) / (len(t_sub) / len(r_sub))
#     return results

# # ==============================================================================
# # 3. DUAL-FOLDER DATA AGGREGATION
# # ==============================================================================
# all_data = []

# # Scan Ex1 and Ex3
# for exp_tag, base_path in [('Ex1', BASE_DIR_EX1), ('Ex3', BASE_DIR_EX3)]:
#     files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
#     for f in files:
#         fname = os.path.basename(f)
#         dose = int(re.search(r"(\d+)dox", fname).group(1)) if re.search(r"(\d+)dox", fname) else 0
        
#         # Only process the requested doses
#         if dose in [100, 1000]:
#             cond = re.search(r"\+(.+?)_", fname).group(1).upper() if re.search(r"\+(.+?)_", fname) else "BASAL"
#             m = get_spatial_metrics(pd.read_csv(f))
#             if m:
#                 m.update({'Dose': dose, 'Condition': cond, 'Exp': exp_tag, 'Rep': os.path.basename(os.path.dirname(f))})
#                 all_data.append(m)

# raw_df = pd.DataFrame(all_data)

# # ==============================================================================
# # 4. CROSS-EXPERIMENT DELTA CALCULATION
# # ==============================================================================
# final_comparison_list = []

# for dose in [100, 1000]:
#     # Use Ex1 (Basal) at this specific dose as the reference
#     ex1_ref = raw_df[(raw_df['Exp'] == 'Ex1') & (raw_df['Dose'] == dose)].mean(numeric_only=True)
    
#     # Compare each Ex3 condition at the SAME dose to the Ex1 baseline
#     ex3_data = raw_df[(raw_df['Exp'] == 'Ex3') & (raw_df['Dose'] == dose)]
    
#     for (cond, rep), group in ex3_data.groupby(['Condition', 'Rep']):
#         entry = {'Dose': dose, 'Condition': cond, 'Rep': rep}
#         for metric in ['nms_Endo', 'nms_Meso', 'count_Endo', 'count_Meso']:
#             v_ref = ex1_ref.get(metric, 0)
#             v_ex3 = group[metric].mean()
#             # Relative % Change: (Ex3 - Ex1) / Ex1
#             entry[f'Rel_Delta_{metric}'] = ((v_ex3 - v_ref) / v_ref * 100) if v_ref != 0 else 0
#         final_comparison_list.append(entry)

# comp_df = pd.DataFrame(final_comparison_list)

# # ==============================================================================
# # 5. VISUALIZATION
# # ==============================================================================
# sns.set_context("talk")
# palette = {'Endo': '#d62728', 'Meso': '#1f77b4'}

# for dose in [100, 1000]:
#     fig, axes = plt.subplots(1, 2, figsize=(20, 7))
#     dose_df = comp_df[comp_df['Dose'] == dose]
    
#     for i, m_type in enumerate(['nms', 'count']):
#         label = "NMS" if m_type == 'nms' else "DBSCAN Cluster Count"
#         long_df = dose_df.melt(id_vars=['Condition', 'Rep'], 
#                                value_vars=[f'Rel_Delta_{m_type}_Endo', f'Rel_Delta_{m_type}_Meso'])
#         long_df['Lineage'] = long_df['variable'].str.extract(r'([^_]+)$')
        
#         sns.barplot(data=long_df, x='Condition', y='value', hue='Lineage', 
#                     palette=palette, ax=axes[i], capsize=.1, errorbar='sd')
        
#         axes[i].set_title(f"{dose}ng Dox: Ex3 vs Ex1 Baseline\n% Change in {label}", fontweight='bold')
#         axes[i].set_ylabel("% Change relative to Ex1")
#         axes[i].axhline(0, color='black', linewidth=1)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, f"Comparison_{dose}ng_Ex1_vs_Ex3.png"), dpi=300)
#     plt.show()






################################################################################################################SANITY CHECK################################RUN IF YOU ARE UNSURE IF THE DATA ACQUISITION IS PROPER################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# import re
# from sklearn.cluster import DBSCAN
# from scipy.spatial import KDTree

# # ==============================================================================
# # 1. CONFIGURATION
# # ==============================================================================
# BASE_DIR_EX1 = "data/GATA-HA_Rep1-3_Ex1"
# BASE_DIR_EX3 = "data/GATA-HA-BMP4+Wnt5a_Ex3"

# OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR_EX3), "Ex1_vs_Ex3_Deep_Troubleshooting")
# if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# RADIUS, EPSILON, MIN_SAMPLES = 100.0, 30.0, 20

# # ==============================================================================
# # 2. ANALYSIS ENGINE WITH DIAGNOSTICS
# # ==============================================================================
# def get_spatial_metrics(df, filename):
#     target_col = None
#     for col in df.columns:
#         if 'cell_type' in col.lower() and 'dapi' in col.lower():
#             target_col = col
#             break
            
#     if not target_col:
#         print(f"      [!!!] REJECTED {filename}: No lineage column found. (Found: {list(df.columns)})")
#         return None
#     # 1. Label Count Diagnostic
#     counts = df[target_col].value_counts().to_dict()
#     n_pluri = counts.get(1.0, 0)
#     n_endo = counts.get(2.0, 0)
#     n_meso = counts.get(3.0, 0)
    
#     # Troubleshooting Print
#     print(f"  [SCAN] {filename}: Pluri={n_pluri}, Endo={n_endo}, Meso={n_meso}")

#     # Rejection Logic Check
#     reasons = []
#     if n_endo < 10: reasons.append(f"Endo < 10 ({n_endo})")
#     if n_meso < 10: reasons.append(f"Meso < 10 ({n_meso})")
    
#     if reasons:
#         print(f"      ---> REJECTED for NMS: {', '.join(reasons)}")
#         return None

#     # Processing Metrics...
#     endo_subset = df[df[target_col] == 2.0]
#     meso_subset = df[df[target_col] == 3.0]
#     results = {'n_endo': n_endo, 'n_meso': n_meso}

#     # DBSCAN Cluster Logic
#     for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
#         if len(subset) >= MIN_SAMPLES:
#             db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(subset[['X', 'Y', 'Z']])
#             unique_clusters = [l for l in np.unique(db.labels_) if l != -1]
#             results[f'count_{name}'] = len(unique_clusters)
#             if unique_clusters:
#                 sizes = [len(subset[db.labels_ == l]) for l in unique_clusters]
#                 results[f'avg_size_{name}'] = np.mean(sizes)
#         else:
#             results[f'count_{name}'], results[f'avg_size_{name}'] = 0, 0

#     # NMS Logic
#     e_coords, m_coords = endo_subset[['X', 'Y', 'Z']].values, meso_subset[['X', 'Y', 'Z']].values
#     e_tree, m_tree = KDTree(e_coords), KDTree(m_coords)
#     for ref, t_tree, r_coords, r_sub, t_sub in [('Endo', m_tree, e_coords, endo_subset, meso_subset),
#                                                ('Meso', e_tree, m_coords, meso_subset, endo_subset)]:
#         n_target = sum([len(n) for n in t_tree.query_ball_point(r_coords, r=RADIUS)])
#         n_ref = sum([len(n)-1 for n in KDTree(r_coords).query_ball_point(r_coords, r=RADIUS)])
#         if n_ref > 0:
#             results[f'nms_{ref}'] = (n_target / n_ref) / (len(t_sub) / len(r_sub))
    
#     print(f"      ---> ACCEPTED. Clusters: E={results.get('count_Endo',0)}, M={results.get('count_Meso',0)}")
#     return results

# # ==============================================================================
# # 3. DATA AGGREGATION & AUDIT
# # ==============================================================================
# all_data = []
# total_found = 0

# for exp_tag, base_path in [('Ex1', BASE_DIR_EX1), ('Ex3', BASE_DIR_EX3)]:
#     files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
#     print(f"\n--- SCRAPING {exp_tag} ({len(files)} potential files) ---")
    
#     for f in files:
#         fname = os.path.basename(f)
#         dose_match = re.search(r"(\d+)dox", fname)
#         if dose_match and int(dose_match.group(1)) in [100, 1000]:
#             total_found += 1
#             cond = re.search(r"\+(.+?)_", fname).group(1).upper() if re.search(r"\+(.+?)_", fname) else "BASAL"
#             m = get_spatial_metrics(pd.read_csv(f), fname)
#             if m:
#                 m.update({'Dose': int(dose_match.group(1)), 'Condition': cond, 'Exp': exp_tag, 'Rep': os.path.basename(os.path.dirname(f))})
#                 all_data.append(m)

# raw_df = pd.DataFrame(all_data)
# print(f"\nAUDIT SUMMARY: Out of {total_found} detected 100/1000ng files, {len(raw_df)} were accepted.")

# # ==============================================================================
# # 4. CROSS-EXPERIMENT CALCULATION
# # ==============================================================================
# final_results = []
# metrics_to_plot = ['nms_Endo', 'nms_Meso', 'avg_size_Endo', 'avg_size_Meso']

# for dose in [100, 1000]:
#     ex1_ref = raw_df[(raw_df['Exp'] == 'Ex1') & (raw_df['Dose'] == dose)].mean(numeric_only=True)
#     ex3_data = raw_df[(raw_df['Exp'] == 'Ex3') & (raw_df['Dose'] == dose)]
    
#     for (cond, rep), group in ex3_data.groupby(['Condition', 'Rep']):
#         entry = {'Dose': dose, 'Condition': cond, 'Rep': rep, 'N_count': len(group)}
#         for m in metrics_to_plot:
#             v_ref = ex1_ref.get(m, 0)
#             v_ex3 = group[m].mean()
#             entry[f'Rel_{m}'] = ((v_ex3 - v_ref) / v_ref * 100) if v_ref > 0 else 0
#         final_results.append(entry)

# Visualization code remains the same as previous (Section 5)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BASE_DIR_EX1 = "data/GATA-HA_Rep1-3_Ex1"
BASE_DIR_EX3 = "data/GATA-HA-BMP4+Wnt5a_Ex3"

# Metadata and Parameters
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR_EX3), "Ex1_vs_Ex3_3Panel_Analysis")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Calibration Constants
RADIUS, EPSILON, MIN_SAMPLES = 100.0, 30.0, 20

# ==============================================================================
# 2. ANALYSIS ENGINE (Captures both Count and Size)
# ==============================================================================
def get_spatial_metrics(df):
    target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
    if target_col not in df.columns: return None

    endo_subset = df[df[target_col] == 2.0]
    meso_subset = df[df[target_col] == 3.0]
    results = {}

    # --- Metric A: DBSCAN Clusters (Count & Size) ---
    for name, subset in [('Endo', endo_subset), ('Meso', meso_subset)]:
        if len(subset) >= MIN_SAMPLES:
            db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(subset[['X', 'Y', 'Z']])
            labels = db.labels_
            unique_clusters = [l for l in np.unique(labels) if l != -1]
            
            results[f'count_{name}'] = len(unique_clusters)
            if len(unique_clusters) > 0:
                results[f'avg_size_{name}'] = np.mean([len(subset[labels == l]) for l in unique_clusters])
            else:
                results[f'avg_size_{name}'] = 0
        else:
            results[f'count_{name}'], results[f'avg_size_{name}'] = 0, 0

    # --- Metric B: Dual-Lineage NMS ---
    if len(endo_subset) > 10 and len(meso_subset) > 10:
        e_coords, m_coords = endo_subset[['X', 'Y', 'Z']].values, meso_subset[['X', 'Y', 'Z']].values
        e_tree, m_tree = KDTree(e_coords), KDTree(m_coords)
        for ref, t_tree, r_coords, r_sub, t_sub in [('Endo', m_tree, e_coords, endo_subset, meso_subset),
                                                   ('Meso', e_tree, m_coords, meso_subset, endo_subset)]:
            n_target = sum([len(n) for n in t_tree.query_ball_point(r_coords, r=RADIUS)])
            n_ref = sum([len(n)-1 for n in KDTree(r_coords).query_ball_point(r_coords, r=RADIUS)])
            if n_ref > 0:
                results[f'nms_{ref}'] = (n_target / n_ref) / (len(t_sub) / len(r_sub))
    return results

# ==============================================================================
# 3. DATA SCRAPING & AUDIT
# ==============================================================================
all_data = []
for exp_tag, base_path in [('Ex1', BASE_DIR_EX1), ('Ex3', BASE_DIR_EX3)]:
    files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
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

# ==============================================================================
# 4. CALCULATION: Organoid-Level Deltas (Includes Counts)
# ==============================================================================
final_comparison_list = []
# Added count_Endo and count_Meso to the tracked metrics
metrics_to_plot = ['nms_Endo', 'nms_Meso', 'avg_size_Endo', 'avg_size_Meso', 'count_Endo', 'count_Meso']

for dose in [100, 1000]:
    ex1_ref = raw_df[(raw_df['Exp'] == 'Ex1') & (raw_df['Dose'] == dose)].mean(numeric_only=True)
    ex3_data = raw_df[(raw_df['Exp'] == 'Ex3') & (raw_df['Dose'] == dose)]
    
    for _, row in ex3_data.iterrows():
        entry = {'Dose': dose, 'Condition': row['Condition'], 'Rep': row['Rep']}
        for m in metrics_to_plot:
            v_ref = ex1_ref.get(m, 0)
            entry[f'Rel_{m}'] = ((row[m] - v_ref) / v_ref * 100) if v_ref > 0 else 0
        final_comparison_list.append(entry)

comp_df = pd.DataFrame(final_comparison_list)

# ==============================================================================
# 5. VISUALIZATION: 3 PANELS (NMS, SIZE, COUNT)
# ==============================================================================
sns.set_context("talk")
palette = {'Endo': '#d62728', 'Meso': '#1f77b4'}

for dose in [100, 1000]:
    # Expanded to (1, 3) to accommodate the new Cluster Count panel
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    dose_df = comp_df[comp_df['Dose'] == dose]
    
    # Iterate through the three main analysis themes
    for i, m_type in enumerate(['nms', 'avg_size', 'count']):
        label_map = {
            'nms': "Mixing Score (NMS)", 
            'avg_size': "Cluster Size (Cells/Cluster)", 
            'count': "Cluster Count (# Clusters)"
        }
        
        long_df = dose_df.melt(id_vars=['Condition', 'Rep'], 
                               value_vars=[f'Rel_{m_type}_Endo', f'Rel_{m_type}_Meso'])
        long_df['Lineage'] = long_df['variable'].str.extract(r'([^_]+)$')
        
        # 1. Barplot for mean % change
        sns.barplot(data=long_df, x='Condition', y='value', hue='Lineage', 
                    palette=palette, ax=axes[i], capsize=.1, errorbar='se')
        
        # 2. Stripplot to show individual organoids (the 9s and 10s)
        sns.stripplot(data=long_df, x='Condition', y='value', hue='Lineage', 
                      dodge=True, alpha=0.4, jitter=True, palette=['black', 'black'], 
                      ax=axes[i], legend=False)

        # Panel Formatting
        axes[i].set_title(f"Metric: {label_map[m_type]}", fontweight='bold', pad=20)
        axes[i].set_ylabel("% Change from Ex1 Baseline")
        axes[i].axhline(0, color='black', linewidth=1.5)
        
        # AXIS FIX: Set limit to prevent +800% outliers from squishing bars
        axes[i].set_ylim(-110, 250) 

    plt.suptitle(f"Spatial Analysis (Individual Organoid Delta): {dose}ng/mL Dox Comparison", 
                 fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 6. FIXED AUDIT PRINT (Size-based count)
# ==============================================================================
print("\n--- SAMPLE SIZE AUDIT (Total Organoids per Group) ---")
# Count rows in each category to verify the 9s and 10s
print(comp_df.groupby(['Condition', 'Dose']).size())