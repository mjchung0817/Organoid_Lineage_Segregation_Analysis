# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from scipy.spatial import ConvexHull, KDTree

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================
# files_map = {
#     0:    "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/0dox_GATA6-HA_001.csv",
#     10:   "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/10dox_GATA6-HA_001.csv",
#     25:   "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/25dox_GATA6-HA_001.csv",
#     50:   "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/50dox_GATA6-HA_001.csv",
#     100:  "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/100dox_GATA6-HA_001.csv",
#     250:  "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/250dox_GATA6-HA_001.csv",
#     500:  "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/500dox_GATA6-HA_001.csv",
#     1000: "data/GATA-HA_Rep1-3/GATA6-HA_Rep3/1000dox_GATA6-HA_001.csv"
# }


# RADIUS = 50.0  
# VIEW_WINDOW = 800
# Z_RADIUS = 20.0  # +/- 20 microns (Depth Cueing)
# DOT_SIZE = 3     # Low clutter size

# # ==============================================================================
# # HELPER 1: ROBUST COLUMN RENAMING
# # ==============================================================================
# def standardize_columns(df):
#     cols = df.columns
#     renames = {}
#     if 'Global Z' not in cols:
#         if 'Position Z' in cols: renames['Position Z'] = 'Global Z'
#         elif 'Z' in cols: renames['Z'] = 'Global Z'
#     if 'Global X' not in cols:
#         if 'Position X' in cols: renames['Position X'] = 'Global X'
#         elif 'X' in cols: renames['X'] = 'Global X'
#     if 'Global Y' not in cols:
#         if 'Position Y' in cols: renames['Position Y'] = 'Global Y'
#         elif 'Y' in cols: renames['Y'] = 'Global Y'
#     if renames:
#         df = df.rename(columns=renames)
#     return df

# # ==============================================================================
# # HELPER 2: FIND WIDEST SLICE (Convex Hull)
# # ==============================================================================
# def get_widest_z_layer(dataframe, z_col='Global Z'):
#     if z_col not in dataframe.columns:
#         return dataframe.iloc[0].get(z_col, 0)
        
#     z_layers = dataframe[z_col].round(1).unique()
#     max_area = 0
#     best_z = z_layers[0] if len(z_layers) > 0 else 0

#     for z in z_layers:
#         slice_points = dataframe[dataframe[z_col].round(1) == z][['Global X', 'Global Y']].values
#         if len(slice_points) >= 3:
#             try:
#                 hull = ConvexHull(slice_points)
#                 if hull.volume > max_area:
#                     max_area = hull.volume
#                     best_z = z
#             except:
#                 continue
#     return best_z

# # ==============================================================================
# # HELPER 3: CALCULATE NMS (Mixing Score)
# # ==============================================================================
# def calculate_local_mixing(df, radius=50.0):
#     coords = df[['Global X', 'Global Y']].values
#     if len(coords) == 0: return np.array([])
        
#     tree = KDTree(coords)
#     indices_list = tree.query_ball_point(coords, r=radius)
#     cell_types = df['cell_type_dapi_adusted'].values
#     local_scores = []
    
#     for i, neighbor_indices in enumerate(indices_list):
#         if len(neighbor_indices) < 2:
#             local_scores.append(0.0)
#             continue
            
#         my_type = cell_types[i]
#         neighbor_types = cell_types[neighbor_indices]
#         # Count neighbors that are NOT my type
#         foreign_count = np.sum(neighbor_types != my_type)
#         total_neighbors = len(neighbor_indices) - 1
        
#         score = foreign_count / total_neighbors if total_neighbors > 0 else 0.0
#         local_scores.append(score)
        
#     return np.array(local_scores)

# # ==============================================================================
# # MAIN PROCESSING FUNCTION
# # ==============================================================================
# def process_organoid(file_path):
#     try:
#         df = pd.read_csv(file_path)
#     except Exception as e:
#         print(f"Failed to load {file_path}: {e}")
#         return None, None
    
#     df = standardize_columns(df)
#     target_z_value = get_widest_z_layer(df)
    
#     # Create Thicker Slice (+/- 20)
#     df_slice = df[(df['Global Z'] >= target_z_value - Z_RADIUS) & 
#                   (df['Global Z'] <= target_z_value + Z_RADIUS)].copy()
    
#     if df_slice.empty: return None, target_z_value

#     # --- CENTER THE ORGANOID ---
#     center_x = df_slice['Global X'].mean()
#     center_y = df_slice['Global Y'].mean()
#     df_slice['Global X'] = df_slice['Global X'] - center_x
#     df_slice['Global Y'] = df_slice['Global Y'] - center_y

#     # --- CALCULATE DEPTH ALPHA ---
#     z_distance = np.abs(df_slice['Global Z'] - target_z_value)
#     norm_dist = z_distance / Z_RADIUS 
#     alpha_values = 1.0 - (0.9 * norm_dist)
#     df_slice['alpha'] = np.clip(alpha_values, 0.1, 1.0)

#     # Mixing Calculation (NMS)
#     df_slice['local_mixing'] = calculate_local_mixing(df_slice, radius=RADIUS)
    
#     return df_slice, target_z_value

# # ==============================================================================
# # PLOTTING LOOP (3 PANELS: ANATOMY | MESO MIXING | ENDO MIXING)
# # ==============================================================================
# dox_levels = sorted(files_map.keys())

# # COLOR MAPPING: 2.0 = Red (Endo), 3.0 = Blue (Meso)
# base_palette = {2.0: '#d62728', 3.0: "#1fb471"} 

# for i, dox in enumerate(dox_levels):
#     df_slice, chosen_z = process_organoid(files_map[dox])
#     print(f"[{dox} ng/mL] Optimized Z-Slice: {chosen_z}")
    
#     if df_slice is not None:
#         # Increase figure size for 3 panels
#         fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
#         # --- PANEL 1: ANATOMY ---
#         colors = df_slice['cell_type_dapi_adusted'].map(
#             lambda x: base_palette.get(x, '#d3d3d3')
#         )
#         rgba_colors = mcolors.to_rgba_array(colors)
#         rgba_colors[:, 3] = df_slice['alpha'].values
        
#         axes[0].scatter(df_slice['Global X'], df_slice['Global Y'], 
#                         c=rgba_colors, s=DOT_SIZE)
#         axes[0].set_title(f"Anatomy ({dox} ng/mL)")

#         # --- PANEL 2: MESODERM MIXING (Ref = Endoderm) ---
#         # Plot ONLY Mesoderm cells (Type 3.0) that are mixing
#         meso_mixing = df_slice[(df_slice['local_mixing'] > 0.0) & 
#                                (df_slice['cell_type_dapi_adusted'] == 3.0)].copy()
        
#         if not meso_mixing.empty:
#             norm = plt.Normalize(vmin=0, vmax=1.0)
#             cmap = plt.cm.viridis
#             heatmap_colors = cmap(norm(meso_mixing['local_mixing'].values))
#             heatmap_colors[:, 3] = meso_mixing['alpha'].values # Apply Alpha
            
#             sc1 = axes[1].scatter(meso_mixing['Global X'], meso_mixing['Global Y'], 
#                                  c=heatmap_colors, s=DOT_SIZE)
#             plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[1], label='Mixing Score')
            
#         axes[1].set_title(f"Mesoderm Mixing\n(Ref: Endoderm)")

#         # --- PANEL 3: ENDODERM MIXING (Ref = Mesoderm) ---
#         # Plot ONLY Endoderm cells (Type 2.0) that are mixing
#         endo_mixing = df_slice[(df_slice['local_mixing'] > 0.0) & 
#                                (df_slice['cell_type_dapi_adusted'] == 2.0)].copy()
        
#         if not endo_mixing.empty:
#             norm = plt.Normalize(vmin=0, vmax=1.0)
#             cmap = plt.cm.viridis
#             heatmap_colors = cmap(norm(endo_mixing['local_mixing'].values))
#             heatmap_colors[:, 3] = endo_mixing['alpha'].values # Apply Alpha
            
#             sc2 = axes[2].scatter(endo_mixing['Global X'], endo_mixing['Global Y'], 
#                                  c=heatmap_colors, s=DOT_SIZE)
#             plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[2], label='Mixing Score')

#         axes[2].set_title(f"Endoderm Mixing\n(Ref: Mesoderm)")

#         # --- COMMON FORMATTING ---
#         for ax in axes:
#             ax.set_xlim(-VIEW_WINDOW, VIEW_WINDOW)
#             ax.set_ylim(-VIEW_WINDOW, VIEW_WINDOW)
#             ax.set_aspect('equal')
#             ax.axis('on')

        
#         # Add a 200um Scale Bar to the first panel
#         scale_bar_x = [VIEW_WINDOW - 300, VIEW_WINDOW - 100]
#         scale_bar_y = [-VIEW_WINDOW + 100, -VIEW_WINDOW + 100]
#         axes[0].plot(scale_bar_x, scale_bar_y, color='black', lw=3)
#         axes[0].text(VIEW_WINDOW - 200, -VIEW_WINDOW + 130, '200 µm', 
#                      ha='center', va='bottom', fontsize=10)
        
        
#         plt.tight_layout()
#         plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial import ConvexHull, KDTree, cKDTree
import re

# ==============================================================================
# 1. DYNAMIC PATH CONFIGURATION
# ==============================================================================
# Base directory for Experiment 1
BASE_DIR = "data/GATA-HA_Rep1-3_Ex1"

dox_levels = [0, 10, 25, 50, 100, 250, 500, 1000]
files_map = {}

print(f"Searching for baseline files in: {BASE_DIR}")

# Use glob to find the files regardless of which Rep folder they are in
for d in dox_levels:
    # This pattern looks for the file in any Rep subfolder
    search_pattern = os.path.join(BASE_DIR, "**", f"{d}dox_GATA6-HA_001.csv")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if found_files:
        files_map[d] = found_files[0]
        print(f"  [FOUND] {d}ng/mL -> {os.path.relpath(found_files[0], BASE_DIR)}")
    else:
        print(f"  [MISSING] {d}ng/mL (Pattern: {d}dox_GATA6-HA_001.csv)")

# ==============================================================================
# 2. ANALYSIS CONSTANTS (From your Calibration)
# ==============================================================================
RADIUS = 50.0  
Z_RADIUS = 20.0  
DOT_SIZE = 4
VIEW_WINDOW = 800  # <--- ADD THIS LINE

# ==============================================================================
# 2. ANALYSIS ENGINE (Lineage-Specific Mixing)
# ==============================================================================
def standardize_columns(df):
    renames = {col: 'Global ' + col for col in ['X', 'Y', 'Z'] if col in df.columns}
    return df.rename(columns=renames)

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
# 3. VISUALIZATION LOOP
# ==============================================================================
base_palette = {2.0: '#d62728', 3.0: "#1fb471"} # Endo=Red, Meso=Green

for dox in sorted(files_map.keys()):
    df = pd.read_csv(files_map[dox])
    df = standardize_columns(df)
    
    # Simple Z-Slice Selection for Baseline Visualization
    z_center = df['Global Z'].median()
    df_slice = df[(df['Global Z'] >= z_center - Z_RADIUS) & 
                  (df['Global Z'] <= z_center + Z_RADIUS)].copy()
    
    if not df_slice.empty:
        # Center the organoid for the VIEW_WINDOW
        df_slice['Global X'] -= df_slice['Global X'].mean()
        df_slice['Global Y'] -= df_slice['Global Y'].mean()
        df_slice['local_mixing'] = calculate_local_mixing(df_slice, radius=RADIUS)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Anatomy Panel
        colors = df_slice['cell_type_dapi_adusted'].map(lambda x: base_palette.get(x, '#d3d3d3'))
        axes[0].scatter(df_slice['Global X'], df_slice['Global Y'], c=colors, s=DOT_SIZE, alpha=0.6)
        axes[0].set_title(f"Anatomy ({dox} ng/mL)")

        # Mixing Panels (Meso and Endo)
        for i, (lin_id, lin_name) in enumerate([(3.0, "Mesoderm"), (2.0, "Endoderm")]):
            sub = df_slice[df_slice['cell_type_dapi_adusted'] == lin_id]
            norm = plt.Normalize(0, 1.0)
            sc = axes[i+1].scatter(sub['Global X'], sub['Global Y'], c=sub['local_mixing'], 
                                   cmap='viridis', norm=norm, s=DOT_SIZE)
            plt.colorbar(sc, ax=axes[i+1], label='Mixing Score')
            axes[i+1].set_title(f"{lin_name} Mixing")

        for ax in axes:
            ax.set_xlim(-VIEW_WINDOW, VIEW_WINDOW); ax.set_ylim(-VIEW_WINDOW, VIEW_WINDOW)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()