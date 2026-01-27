# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial import KDTree

# # ==============================================================================
# # 1. CONFIGURATION
# # ==============================================================================
# root_folder = "data/GATA6-HA_Rep1-3/"
# RADIUS = 100.0

# # ==============================================================================
# # 2. MATH FUNCTION
# # ==============================================================================
# def calculate_metrics(df, radius=30.0):
#     if 'cell_type_dapi_adusted' not in df.columns:
#         return None, None, None, None
        
#     ref_df = df[df['cell_type_dapi_adusted'] == 2.0]    # Endoderm
#     target_df = df[df['cell_type_dapi_adusted'] == 3.0] # Mesoderm
    
#     if len(ref_df) < 10 or len(target_df) < 10:
#         return None, None, None, None

#     # Calculate Counts
#     n_endo = len(ref_df)
#     n_meso = len(target_df)

#     # --- ABUNDANCE METRIC ---
#     # Formula: Endo / (Endo + Meso)
#     pct_endo = n_endo / (n_endo + n_meso)

#     # --- SPATIAL PREP ---
#     endo_coords = ref_df[['X', 'Y', 'Z']].values
#     meso_coords = target_df[['X', 'Y', 'Z']].values
#     endo_tree = KDTree(endo_coords)
#     meso_tree = KDTree(meso_coords)

#     # ==========================================
#     # CALCULATION 1: NMS with respect to ENDODERM
#     # (How many Meso neighbors does an Endo cell have?)
#     # ==========================================
#     global_ratio_endo = n_meso / n_endo
    
#     neighbors_meso_around_endo = meso_tree.query_ball_point(endo_coords, r=radius)
#     count_meso_around_endo = sum([len(n) for n in neighbors_meso_around_endo])

#     neighbors_endo_around_endo = endo_tree.query_ball_point(endo_coords, r=radius)
#     count_endo_around_endo = sum([len(n) - 1 for n in neighbors_endo_around_endo]) # -1 to exclude self

#     if count_endo_around_endo == 0:
#         nms_endo = 0.0
#     else:
#         raw_score = count_meso_around_endo / count_endo_around_endo
#         nms_endo = raw_score / global_ratio_endo

#     # ==========================================
#     # CALCULATION 2: NMS with respect to MESODERM
#     # (How many Endo neighbors does a Meso cell have?)
#     # ==========================================
#     global_ratio_meso = n_endo / n_meso

#     neighbors_endo_around_meso = endo_tree.query_ball_point(meso_coords, r=radius)
#     count_endo_around_meso = sum([len(n) for n in neighbors_endo_around_meso])

#     neighbors_meso_around_meso = meso_tree.query_ball_point(meso_coords, r=radius)
#     count_meso_around_meso = sum([len(n) - 1 for n in neighbors_meso_around_meso])

#     if count_meso_around_meso == 0:
#         nms_meso = 0.0
#     else:
#         raw_score_meso = count_endo_around_meso / count_meso_around_meso
#         nms_meso = raw_score_meso / global_ratio_meso

#     return nms_endo, nms_meso, pct_endo, global_ratio_endo

# # ==============================================================================
# # 3. BATCH PROCESSING
# # ==============================================================================
# results_list = []
# print(f"Scanning for files in: {os.path.abspath(root_folder)}...")

# for root, dirs, files in os.walk(root_folder):
#     for file in files:
#         if "dox" in file and file.endswith(".csv"):
#             try:
#                 dox_str = file.split('dox')[0]
#                 dox_conc = int(dox_str)
#             except ValueError:
#                 continue

#             try:
#                 df = pd.read_csv(os.path.join(root, file))
#                 nms_e, nms_m, pct_endo, _ = calculate_metrics(df, radius=RADIUS)
                
#                 if nms_e is not None:
#                     results_list.append({
#                         'Dox Concentration': dox_conc,
#                         'NMS_Endo': nms_e,
#                         'NMS_Meso': nms_m,
#                         'Pct_Endoderm': pct_endo,
#                         'Replicate': file
#                     })
#                     print(f"  [OK] {file}: NMS(E)={nms_e:.2f}, NMS(M)={nms_m:.2f}, %Endo={pct_endo:.2%}")
#                 else:
#                     print(f"  [SKIP] {file}: Not enough cells.")
#             except Exception as e:
#                 print(f"  [ERROR] {file}: {e}")

# # ==============================================================================
# # 4. VISUALIZATION (FIXED OVERLAP)
# # ==============================================================================
# if len(results_list) > 0:
#     final_df = pd.DataFrame(results_list)
#     final_df = final_df.sort_values('Dox Concentration')
    
#     final_df.to_csv("summary_spatial_dual_nms.csv", index=False)
#     print("\nData saved to 'summary_spatial_dual_nms.csv'")

#     # Create Figure
#     fig, ax1 = plt.subplots(figsize=(12, 8)) # Increased Height for Bottom Legend

#     # --- LEFT AXIS (Blue/Cyan): Spatial Segregation ---
#     sns.lineplot(
#         data=final_df, x='Dox Concentration', y='NMS_Endo',
#         marker='o', color='#1f77b4', linewidth=2.5, 
#         err_style='bars', errorbar='se', ax=ax1,
#         label='NMS (Ref: Endoderm)'
#     )
    
#     sns.lineplot(
#         data=final_df, x='Dox Concentration', y='NMS_Meso',
#         marker='^', color='#00bfff', linewidth=2.5, linestyle='--',
#         err_style='bars', errorbar='se', ax=ax1,
#         label='NMS (Ref: Mesoderm)'
#     )

#     ax1.set_ylabel('Normalized Mixing Score (NMS)', color='#1f77b4', fontsize=12, fontweight='bold')
#     ax1.tick_params(axis='y', labelcolor='#1f77b4')
#     ax1.set_xlabel('Dox Concentration (ng/mL)', fontsize=12)
    
#     # Random Mixing Line
#     ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.6, label='Random Mixing (1.0)')

#     # --- RIGHT AXIS (Orange): Cell Fate Abundance ---
#     ax2 = ax1.twinx() 
#     sns.lineplot(
#         data=final_df, x='Dox Concentration', y='Pct_Endoderm',
#         marker='s', color='#ff7f0e', linewidth=2.5,
#         err_style='bars', errorbar='se', ax=ax2,
#         label='% Endoderm Abundance'
#     )
    
#     # Formula in title
#     ax2.set_ylabel('Proportion of Endoderm Cells\n(Endo / [Endo + Meso])', 
#                    color='#ff7f0e', fontsize=12, fontweight='bold')
#     ax2.tick_params(axis='y', labelcolor='#ff7f0e')
#     ax2.set_ylim(0, 1.0) 

#     # --- FORMATTING ---
#     ax1.set_xscale('symlog', linthresh=10)
#     ax1.set_xlim(left=-5)
    
#     # --- FIX LEGEND OVERLAP (Move to Bottom) ---
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
    
#     # Position: Upper Center relative to the ANCHOR, which is placed BELOW the plot (-0.15)
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
#                loc='upper center', 
#                bbox_to_anchor=(0.5, -0.12), # Moves legend below the X-axis
#                ncol=3, # Spreads items horizontally
#                frameon=False, fontsize=11)
    
#     if ax2.get_legend(): ax2.get_legend().remove()

#     plt.title(f'Spatial Patterning vs. Cell Fate Abundance (Radius {RADIUS}um)', fontsize=16, pad=20)
#     plt.grid(True, alpha=0.3)
    
#     # Use constrained_layout to automatically make space for the bottom legend
#     fig.set_layout_engine('constrained')
    
#     plt.show()

# else:
#     print("No valid data found.")






# ##########################################
# ##Experiment 3 data
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# from scipy.spatial import KDTree

# # ==============================================================================
# # 1. CONFIGURATION
# # ==============================================================================
# # Point this to your Experiment 3 folder
# root_folder = "data/GATA-HA-BMP4+Wnt5a_Ex3"
# RADIUS = 100.0

# # ==============================================================================
# # 2. METRIC ENGINE (Dual-NMS + % Endo)
# # ==============================================================================
# def calculate_metrics(df, radius=100.0):
#     # Flexible typo handling
#     target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
#     if target_col not in df.columns: return None, None, None

#     ref_df = df[df[target_col] == 2.0]    # Endoderm
#     target_df = df[df[target_col] == 3.0] # Mesoderm
    
#     if len(ref_df) < 10 or len(target_df) < 10:
#         return None, None, None

#     n_endo, n_meso = len(ref_df), len(target_df)
#     pct_endo = n_endo / (n_endo + n_meso)

#     e_coords, m_coords = ref_df[['X', 'Y', 'Z']].values, target_df[['X', 'Y', 'Z']].values
#     e_tree, m_tree = KDTree(e_coords), KDTree(m_coords)

#     # NMS (Ref: Endoderm)
#     n_m_around_e = sum([len(n) for n in m_tree.query_ball_point(e_coords, r=radius)])
#     n_e_around_e = sum([len(n)-1 for n in e_tree.query_ball_point(e_coords, r=radius)])
#     nms_e = (n_m_around_e / n_e_around_e) / (n_meso / n_endo) if n_e_around_e > 0 else 0

#     # NMS (Ref: Mesoderm)
#     n_e_around_m = sum([len(n) for n in e_tree.query_ball_point(m_coords, r=radius)])
#     n_m_around_m = sum([len(n)-1 for n in m_tree.query_ball_point(m_coords, r=radius)])
#     nms_m = (n_e_around_m / n_m_around_m) / (n_endo / n_meso) if n_m_around_m > 0 else 0

#     return nms_e, nms_m, pct_endo

# # ==============================================================================
# # 3. BATCH PROCESSING FOR EXP 3
# # ==============================================================================
# results_list = []
# for root, dirs, files in os.walk(root_folder):
#     for file in files:
#         if "dox" in file and file.endswith(".csv"):
#             # Parse Metadata
#             dose = int(re.search(r"(\d+)dox", file).group(1)) if re.search(r"(\d+)dox", file) else 0
#             # Normalize condition case (Wnt5a -> WNT5A)
#             cond = re.search(r"\+(.+?)_", file).group(1).upper() if re.search(r"\+(.+?)_", file) else "BASAL"
            
#             try:
#                 df = pd.read_csv(os.path.join(root, file))
#                 nms_e, nms_m, pct_e = calculate_metrics(df, radius=RADIUS)
#                 if nms_e is not None:
#                     results_list.append({
#                         'Dose': dose, 'Condition': cond,
#                         'NMS_Endo': nms_e, 'NMS_Meso': nms_m, 'Pct_Endo': pct_e,
#                         'Replicate': os.path.basename(root)
#                     })
#             except Exception as e: print(f"Error {file}: {e}")

# master_df = pd.DataFrame(results_list).sort_values('Dose')

# # 4. LINE PLOT VISUALIZATION (Condition vs. Basal Overlay)
# # ==============================================================================
# # Separate the Basal data to use as an overlay
# basal_subset = master_df[master_df['Condition'] == "BASAL"]
# signaling_conditions = [c for c in master_df['Condition'].unique() if c != "BASAL"]

# for cond in signaling_conditions:
#     cond_subset = master_df[master_df['Condition'] == cond]
#     fig, ax1 = plt.subplots(figsize=(12, 7))

#     # --- LEFT AXIS: Spatial Segregation (NMS) ---
#     # 1. Plot the Signaling Condition (Solid Lines)
#     sns.lineplot(data=cond_subset, x='Dose', y='NMS_Endo', marker='o', color='#d62728', 
#                  err_style='bars', errorbar='se', ax=ax1, label=f'NMS Endo ({cond})')
#     sns.lineplot(data=cond_subset, x='Dose', y='NMS_Meso', marker='^', color='#1f77b4', 
#                  err_style='bars', errorbar='se', ax=ax1, label=f'NMS Meso ({cond})')
    
#     # 2. Plot the Basal Control (Dashed Gray Lines)
#     sns.lineplot(data=basal_subset, x='Dose', y='NMS_Endo', color='gray', linestyle='--', 
#                  alpha=0.5, errorbar=None, ax=ax1, label='NMS Endo (Basal)')
#     sns.lineplot(data=basal_subset, x='Dose', y='NMS_Meso', color='darkgray', linestyle=':', 
#                  alpha=0.5, errorbar=None, ax=ax1, label='NMS Meso (Basal)')

#     ax1.set_ylabel('Mixing Score (NMS)', fontsize=12, fontweight='bold')
#     ax1.set_xlabel('Dox Concentration (ng/mL)', fontsize=12)
#     ax1.axhline(y=1.0, color='black', linestyle='-', alpha=0.2) # Random Mixing Baseline

#     # --- RIGHT AXIS: Abundance (% Endo) ---
#     ax2 = ax1.twinx()
#     # 1. Plot Signaling Abundance
#     sns.lineplot(data=cond_subset, x='Dose', y='Pct_Endo', marker='s', color='#ff7f0e', 
#                  err_style='bars', errorbar='se', ax=ax2, label=f'% Endo ({cond})')
#     # 2. Plot Basal Abundance
#     sns.lineplot(data=basal_subset, x='Dose', y='Pct_Endo', color='#ffcc80', linestyle='--', 
#                  alpha=0.5, errorbar=None, ax=ax2, label='% Endo (Basal)')
    
#     ax2.set_ylabel('Proportion of Endoderm Cells', color='#ff7f0e', fontsize=12, fontweight='bold')
#     ax2.set_ylim(0, 1.0)

#     # --- AXIS SCALING & LIMITS ---
#     ax1.set_xscale('symlog', linthresh=10) 
#     ax1.set_xlim(left=0) # Starts the plot exactly at 0 Dox
    
#     # Clean up legends
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
#                bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)
#     if ax2.get_legend(): ax2.get_legend().remove()

#     plt.title(f'Synergy Analysis: {cond} vs. Basal Control', fontsize=15, pad=20)
#     plt.tight_layout()
#     plt.show()



##EXP3: dot plot of NMS with the 4 categories (CTRL vs +BMP4 vs +Wnt5a vs +BMP4+Wnt5a) for each condition across dox concnetrations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from scipy.spatial import KDTree

# ==============================================================================
# 1. CONFIGURATION (Dual-Experiment Access)
# ==============================================================================
BASE_DIR_EX1 = "data/GATA-HA_Rep1-3_Ex1"
BASE_DIR_EX3 = "data/GATA-HA-BMP4+Wnt5a_Ex3"

RADIUS = 100.0

# ==============================================================================
# 2. METRIC ENGINE
# ==============================================================================
def calculate_metrics(df, radius=100.0):
    target_col = 'cell_type_dapi_adjusted' if 'cell_type_dapi_adjusted' in df.columns else 'cell_type_dapi_adusted'
    if target_col not in df.columns: return None

    ref_df = df[df[target_col] == 2.0]    # Endo
    target_df = df[df[target_col] == 3.0] # Meso
    
    if len(ref_df) < 10 or len(target_df) < 10: return None

    e_coords, m_coords = ref_df[['X', 'Y', 'Z']].values, target_df[['X', 'Y', 'Z']].values
    e_tree, m_tree = KDTree(e_coords), KDTree(m_coords)

    # NMS (Ref: Endoderm)
    n_m_around_e = sum([len(n) for n in m_tree.query_ball_point(e_coords, r=radius)])
    n_e_around_e = sum([len(n)-1 for n in e_tree.query_ball_point(e_coords, r=radius)])
    nms_e = (n_m_around_e / n_e_around_e) / (len(target_df) / len(ref_df)) if n_e_around_e > 0 else 0

    # NMS (Ref: Mesoderm)
    n_e_around_m = sum([len(n) for n in e_tree.query_ball_point(m_coords, r=radius)])
    n_m_around_m = sum([len(n)-1 for n in m_tree.query_ball_point(m_coords, r=radius)])
    nms_m = (n_e_around_m / n_m_around_m) / (len(ref_df) / len(target_df)) if n_m_around_m > 0 else 0

    return {'NMS_Endo': nms_e, 'NMS_Meso': nms_m}

# ==============================================================================
# 3. CROSS-EXPERIMENT DATA RETRIEVAL
# ==============================================================================
results = []

# Logic: Pull CTRL from Ex1, Pull Additives from Ex3
for exp_tag, base_path in [('Ex1', BASE_DIR_EX1), ('Ex3', BASE_DIR_EX3)]:
    files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    for f in files:
        fname = os.path.basename(f)
        dose = int(re.search(r"(\d+)dox", fname).group(1)) if re.search(r"(\d+)dox", fname) else 0
        
        # Determine Condition
        if exp_tag == 'Ex1':
            cond = "CTRL"
        else:
            cond_match = re.search(r"\+(.+?)_", fname)
            cond = cond_match.group(1).upper() if cond_match else "BASAL"
        
        # We only need these specific doses
        if dose in [0, 100, 1000]:
            m = calculate_metrics(pd.read_csv(f), radius=RADIUS)
            if m:
                m.update({'Dose': dose, 'Condition': cond, 'Replicate': os.path.basename(os.path.dirname(f))})
                results.append(m)

df_plot = pd.DataFrame(results)
category_order = ["CTRL", "BMP4", "WNT5A", "BMP4+WNT5A"]

# ==============================================================================
# 4. DOT PLOT VISUALIZATION (DISTINCT COLORS)
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
    ax.set_title(f"Spatial Segregation: {metric.replace('_', ' ')} (Ex3 vs Ex1 Baseline)", fontweight='bold')
    ax.set_ylabel("Normalized Mixing Score (NMS)")
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random Mixing')
    
    # Consolidate Legend
    handles, labels = ax.get_legend_handles_labels()
    # Ensure we only grab unique dose labels
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Dox (ng/mL)", loc='upper right', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.show()