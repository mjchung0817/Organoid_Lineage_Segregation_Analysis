# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial import KDTree

# # --- 1. CONFIGURATION ---
# # POINT THIS TO THE PARENT FOLDER CONTAINING REP1, REP2, REP3
# # Based on your screenshot, this is the likely path:
# root_folder = "data/GATA6-HA_Rep1-3/"

# # --- 2. MATH FUNCTION (Normalized Mixing Score) ---
# def calculate_normalized_mixing_score(df, radius=30.0):
#     # Filter for Clean Clusters (Endo=2.0, Meso=3.0)
#     # Using the misspelled column name 'adusted' as confirmed in your screenshot
#     if 'cell_type_dapi_adusted' not in df.columns:
#         return None, None
        
#     ref_df = df[df['cell_type_dapi_adusted'] == 2.0]
#     target_df = df[df['cell_type_dapi_adusted'] == 3.0]
    
#     # Skip if missing a population
#     if len(ref_df) < 10 or len(target_df) < 10:
#         return None, None

#     # Calculate Global Ratio (Baseline Randomness)
#     global_ratio = len(target_df) / len(ref_df)

#     # Build KD-Trees
#     ref_coords = ref_df[['X', 'Y', 'Z']].values
#     target_coords = target_df[['X', 'Y', 'Z']].values
#     ref_tree = KDTree(ref_coords)
#     target_tree = KDTree(target_coords)

#     # Count Neighbors (Fixed Radius 30um)
#     target_neighbors = target_tree.query_ball_point(ref_coords, r=radius)
#     count_target = sum([len(n) for n in target_neighbors])

#     ref_neighbors = ref_tree.query_ball_point(ref_coords, r=radius)
#     count_ref = sum([len(n) - 1 for n in ref_neighbors]) # -1 to exclude self

#     if count_ref == 0:
#         return 0.0, global_ratio

#     # Normalize: (Observed / Global Ratio)
#     raw_score = count_target / count_ref
#     return raw_score / global_ratio, global_ratio

# # --- 3. BATCH PROCESSING ---
# results_list = []
# print(f"Scanning for files in: {os.path.abspath(root_folder)}...")

# for root, dirs, files in os.walk(root_folder):
#     for file in files:
#         # Strict check to ensure we only grab data CSVs
#         if "dox" in file and file.endswith(".csv"):
            
#             # Parse Dox Concentration from filename (e.g., "50dox..." -> 50)
#             try:
#                 dox_str = file.split('dox')[0]
#                 dox_conc = int(dox_str)
#             except ValueError:
#                 continue

#             # Load Data
#             full_path = os.path.join(root, file)
#             try:
#                 df = pd.read_csv(full_path)
#                 nms, global_ratio= calculate_normalized_mixing_score(df, radius=30.0)
                
#                 if nms is not None:
#                     results_list.append({
#                         'Dox Concentration': dox_conc,
#                         'NMS': nms,
#                         'Replicate': file,
#                         'Folder': os.path.basename(root)
#                     })
#                     print(f"  [OK] {file}: Dox={dox_conc}, NMS={nms:.3f}")
#                     print(f"global ratio(meso/endo): {global_ratio}")
#                 else:
#                     print(f"  [SKIP] {file}: Not enough cells.")
            
#             except Exception as e:
#                 print(f"  [ERROR] {file}: {e}")

# # --- 4. VISUALIZATION ---
# if len(results_list) > 0:
#     final_df = pd.DataFrame(results_list)
#     final_df = final_df.sort_values('Dox Concentration')

#     # Save raw data
#     final_df.to_csv("summary_spatial_analysis.csv", index=False)
#     print("\nData saved to 'summary_spatial_analysis.csv'")

#     # Create Plot
#     plt.figure(figsize=(10, 6))
    
#     # Lineplot with Error Bars (95% Confidence Interval by default)
#     sns.lineplot(
#         data=final_df, 
#         x='Dox Concentration',    ####seaborn.lineplot(...) aggregates automatically by x
#         y='NMS', 
#         marker='o',
#         linewidth=2.5,
#         err_style='bars',
#         errorbar='se' # Shows Standard Error (variation between replicates), dont need to call the replicates. 
#     )

#     plt.axhline(y=1.0, color='red', linestyle='--', label='Random Mixing (1.0)')
    
#     # Log scale is often better for Dox (0, 10, 100, 1000)
#     # We use symlog to handle the '0' value gracefully
#     plt.xscale('symlog')
    
#     plt.title('Endoderm-Mesoderm Segregation vs. Dox', fontsize=14)
#     plt.ylabel('Normalized Mixing Score (NMS)', fontsize=12)
#     plt.xlabel('Dox Concentration (ng/mL)', fontsize=12)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# else:
#     print("No valid files found! Check the path.")





import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree

# --- 1. CONFIGURATION ---
root_folder = "data/GATA6-HA_Rep1-3/"
RADIUS = 50.0

# --- 2. MATH FUNCTION ---
def calculate_metrics(df, radius=30.0):
    if 'cell_type_dapi_adusted' not in df.columns:
        return None, None, None
        
    ref_df = df[df['cell_type_dapi_adusted'] == 2.0]    # Endoderm
    target_df = df[df['cell_type_dapi_adusted'] == 3.0] # Mesoderm
    
    if len(ref_df) < 10 or len(target_df) < 10:
        return None, None, None

    # Calculate Counts
    n_endo = len(ref_df)
    n_meso = len(target_df)

    # Metric 1: Global Ratio (Meso/Endo) for Normalization
    global_ratio = n_meso / n_endo
    
    # ### [NEW] Metric 2: Percent Endoderm Calculation
    # We calculate what % of the analyzed cells are Endoderm
    pct_endo = n_endo / (n_endo + n_meso)

    # Metric 3: Normalized Mixing Score (NMS)
    ref_coords = ref_df[['X', 'Y', 'Z']].values
    target_coords = target_df[['X', 'Y', 'Z']].values
    ref_tree = KDTree(ref_coords)
    target_tree = KDTree(target_coords)

    target_neighbors = target_tree.query_ball_point(ref_coords, r=radius)
    count_target = sum([len(n) for n in target_neighbors])

    ref_neighbors = ref_tree.query_ball_point(ref_coords, r=radius)
    count_ref = sum([len(n) - 1 for n in ref_neighbors])

    if count_ref == 0:
        nms = 0.0
    else:
        raw_score = count_target / count_ref
        nms = raw_score / global_ratio

    # ### [NEW] Return pct_endo along with nms
    return nms, pct_endo, global_ratio

# --- 3. BATCH PROCESSING ---
results_list = []
print(f"Scanning for files in: {os.path.abspath(root_folder)}...")

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if "dox" in file and file.endswith(".csv"):
            try:
                dox_str = file.split('dox')[0]
                dox_conc = int(dox_str)
            except ValueError:
                continue

            try:
                df = pd.read_csv(os.path.join(root, file))
                
                # ### [NEW] Unpack the new return value (pct_endo)
                nms, pct_endo, _ = calculate_metrics(df, radius=RADIUS)
                
                if nms is not None:
                    results_list.append({
                        'Dox Concentration': dox_conc,
                        'NMS': nms,
                        'Pct_Endoderm': pct_endo, # ### [NEW] Save abundance data
                        'Replicate': file
                    })
                    print(f"  [OK] {file}: NMS={nms:.3f}, %Endo={pct_endo:.2%}")
                else:
                    print(f"  [SKIP] {file}: Not enough cells.")
            except Exception as e:
                print(f"  [ERROR] {file}: {e}")

# --- 4. DUAL-AXIS VISUALIZATION ---
if len(results_list) > 0:
    final_df = pd.DataFrame(results_list)
    final_df = final_df.sort_values('Dox Concentration')
    
    final_df.to_csv("summary_spatial_vs_abundance.csv", index=False)
    print("\nData saved to 'summary_spatial_vs_abundance.csv'")

    # ### [NEW] Create Figure with Dual Axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ### [NEW] Plot 1: Mixing Score on LEFT Axis (Blue)
    sns.lineplot(
        data=final_df, x='Dox Concentration', y='NMS',
        marker='o', color='#1f77b4', err_style='bars', errorbar='se', ax=ax1,
        label='Spatial Segregation (NMS)'
    )
    ax1.set_ylabel('Normalized Mixing Score (NMS)', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xlabel('Dox Concentration (ng/mL)', fontsize=12)
    
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Random Mixing (1.0)')

    # ### [NEW] Plot 2: Abundance on RIGHT Axis (Orange)
    ax2 = ax1.twinx() # Create a second y-axis sharing the same x-axis
    sns.lineplot(
        data=final_df, x='Dox Concentration', y='Pct_Endoderm',
        marker='s', color='#ff7f0e', err_style='bars', errorbar='se', ax=ax2,
        label='% Endoderm Abundance'
    )
    ax2.set_ylabel('Proportion of Endoderm Cells', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 1.0) # Lock percentage scale from 0 to 100%

    # --- FORMATTING ---
    ax1.set_xscale('symlog', linthresh=10)
    ax1.set_xlim(left=-5)
    
    # ### [NEW] Combine Legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    plt.title(f'Spatial Patterning vs. Cell Fate Abundance (Radius {RADIUS}um)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
    print("No valid data found.")
    
    