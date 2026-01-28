# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial import KDTree

# # --- CONFIGURATION: PICK YOUR CHAMPIONS ---
# # Paste the full paths to one interesting 0dox file and one 1000dox file
# file_low_dox  = "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/0dox_GATA6-HA_001.csv"
# file_high_dox = "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/1000dox_GATA6-HA_001.csv"

# RADIUS = 50.0  # Search radius for the heatmap

# # --- FUNCTION: Calculate Local Mixing Score Per Cell ---
# def get_slice_data(filename, radius=50.0, z_thickness=40.0):
#     try:
#         df = pd.read_csv(filename)
#     except:
#         print(f"Could not read {filename}")
#         return None

#     # Filter for clean populations
#     if 'cell_type_dapi_adusted' not in df.columns: return None
#     df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])]
    
#     # 1. Take a "Virtual Slice" through the center Z
#     z_mid = df['Z'].median()
#     df_slice = df[ (df['Z'] > z_mid - z_thickness) & (df['Z'] < z_mid + z_thickness) ].copy()
    
#     if len(df_slice) < 10: return None

#     # 2. Calculate Local Mixing Score for EVERY cell in the slice
#     # (We re-build the tree just for the slice for visualization speed)
#     coords = df_slice[['X', 'Y']].values # 2D projection
#     tree = KDTree(coords)
    
#     local_scores = []
    
#     for idx, row in df_slice.iterrows():
#         # Find neighbors in 2D slice
#         indices = tree.query_ball_point([row['X'], row['Y']], r=radius)
#         neighbors = df_slice.iloc[indices]
        
#         my_type = row['cell_type_dapi_adusted']
#         # Count "Foreign" neighbors
#         foreign_count = len(neighbors[neighbors['cell_type_dapi_adusted'] != my_type])
#         total_neighbors = len(neighbors) - 1 # Exclude self
        
#         if total_neighbors > 0:
#             score = foreign_count / total_neighbors
#         else:
#             score = 0.0
#         local_scores.append(score)
        
#     df_slice['local_mixing'] = local_scores
#     return df_slice

# # --- LOAD DATA ---
# df_low = get_slice_data(file_low_dox)
# df_high = get_slice_data(file_high_dox)

# # --- PLOTTING ---
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# # ROW 1: ANATOMY (Blue=Meso, Red=Endo)
# # Low Dox
# sns.scatterplot(data=df_low, x='X', y='Y', hue='cell_type_dapi_adusted', 
#                 palette={2.0:'red', 3.0:'blue'}, s=10, ax=axes[0,0], legend=False)
# axes[0,0].set_title("Low Dox: Anatomy (Red=Endo)", fontsize=14)
# axes[0,0].axis('equal')

# # High Dox
# sns.scatterplot(data=df_high, x='X', y='Y', hue='cell_type_dapi_adusted', 
#                 palette={2.0:'red', 3.0:'blue'}, s=10, ax=axes[0,1], legend=True)
# axes[0,1].set_title("High Dox: Anatomy", fontsize=14)
# axes[0,1].axis('equal')

# # ROW 2: INTERACTION HEATMAP (Color = How mixed is this specific cell?)
# # Low Dox
# sc1 = axes[1,0].scatter(df_low['X'], df_low['Y'], c=df_low['local_mixing'], 
#                         cmap='viridis', s=10, vmin=0, vmax=1.0)
# axes[1,0].set_title("Low Dox: Mixing Heatmap", fontsize=14)
# axes[1,0].axis('equal')
# plt.colorbar(sc1, ax=axes[1,0], label='Local Mixing Score')

# # High Dox
# sc2 = axes[1,1].scatter(df_high['X'], df_high['Y'], c=df_high['local_mixing'], 
#                         cmap='viridis', s=10, vmin=0, vmax=1.0)
# axes[1,1].set_title("High Dox: Mixing Heatmap", fontsize=14)
# axes[1,1].axis('equal')
# plt.colorbar(sc2, ax=axes[1,1], label='Local Mixing Score')

# plt.suptitle("Qualitative Analysis: Loss of Boundary Precision", fontsize=16)
# plt.tight_layout()
# plt.show()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial import KDTree

# # --- 1. CONFIGURATION ---
# # Path to your main data folder
# root_folder = "data/GATA6-HA_Rep1-3/"
# RADIUS = 50.0        # Search radius for mixing
# Z_THICKNESS = 40.0   # Thickness of the virtual slice

# # --- 2. FILE FINDER ---
# def get_representative_files(folder):
#     """Scans folder and picks ONE representative csv for each Dox concentration."""
#     files_by_dox = {}
    
#     print(f"Scanning {folder}...")
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if "dox" in file and file.endswith(".csv"):
#                 try:
#                     # Parse Dox (e.g., "1000dox..." -> 1000)
#                     dox_part = file.split('dox')[0]
#                     # Handle cases like "0dox" or path issues
#                     if dox_part.isdigit():
#                         dox = int(dox_part)
                        
#                         # Only save if we haven't found this concentration yet
#                         # (This picks the first Replicate found as the "Representative")
#                         if dox not in files_by_dox:
#                             files_by_dox[dox] = os.path.join(root, file)
#                 except:
#                     continue
    
#     return dict(sorted(files_by_dox.items())) # Sort by concentration (0, 10, ...)

# # --- 3. MATH FUNCTION (Your existing logic) ---
# def get_slice_data(filename, radius=RADIUS, z_thickness=Z_THICKNESS):
#     try:
#         df = pd.read_csv(filename)
#     except:
#         return None

#     if 'cell_type_dapi_adusted' not in df.columns: return None
#     df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])]
    
#     # Virtual Slice
#     z_mid = df['Z'].median()
#     df_slice = df[ (df['Z'] > z_mid - z_thickness) & (df['Z'] < z_mid + z_thickness) ].copy()
    
#     if len(df_slice) < 10: return None

#     # Calculate Local Mixing
#     coords = df_slice[['X', 'Y']].values
#     tree = KDTree(coords)
#     local_scores = []
    
#     for idx, row in df_slice.iterrows():
#         indices = tree.query_ball_point([row['X'], row['Y']], r=radius)
#         neighbors = df_slice.iloc[indices]
#         my_type = row['cell_type_dapi_adusted']
#         foreign_count = len(neighbors[neighbors['cell_type_dapi_adusted'] != my_type])
#         total_neighbors = len(neighbors) - 1
        
#         score = foreign_count / total_neighbors if total_neighbors > 0 else 0.0
#         local_scores.append(score)
        
#     df_slice['local_mixing'] = local_scores
#     return df_slice

# # --- 4. EXECUTION & PLOTTING ---
# files_map = get_representative_files(root_folder)
# dox_levels = list(files_map.keys())
# n_cols = len(dox_levels)

# if n_cols == 0:
#     print("No files found!")
# else:
#     print(f"Found {n_cols} conditions: {dox_levels}")

#     # Create a dynamic grid: 2 Rows x N Columns
#     fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), constrained_layout=True)
    
#     # Handle single column case (unlikely but safe)
#     if n_cols == 1: axes = np.array([ [axes[0]], [axes[1]] ])

#     for i, dox in enumerate(dox_levels):
#         file_path = files_map[dox]
#         print(f"Processing {dox} Dox...")
        
#         df_slice = get_slice_data(file_path)
        
#         if df_slice is not None:
#             # --- ROW 1: ANATOMY (The Biopsy) ---
#             ax_top = axes[0, i]
#             sns.scatterplot(data=df_slice, x='X', y='Y', hue='cell_type_dapi_adusted', 
#                             palette={2.0:'#d62728', 3.0:'#1f77b4'}, s=10, 
#                             legend=False, ax=ax_top)
#             ax_top.set_title(f"{dox} ng/mL Dox", fontsize=14, fontweight='bold')
#             ax_top.set_aspect('equal')
#             ax_top.set_xlabel('')
#             ax_top.set_ylabel('')
#             ax_top.set_xticks([])
#             ax_top.set_yticks([])
#             if i == 0: ax_top.set_ylabel('Virtual Biopsy\n(Cell Type)', fontsize=12)

#             # --- ROW 2: HEATMAP (The Mechanism) ---
#             ax_bot = axes[1, i]
#             sc = ax_bot.scatter(df_slice['X'], df_slice['Y'], c=df_slice['local_mixing'], 
#                                 cmap='viridis', s=10, vmin=0, vmax=1.0)
#             ax_bot.set_aspect('equal')
#             ax_bot.set_xlabel('')
#             ax_bot.set_ylabel('')
#             ax_bot.set_xticks([])
#             ax_bot.set_yticks([])
#             if i == 0: ax_bot.set_ylabel('Mixing Heatmap\n(Local Score)', fontsize=12)

#     # Add a global colorbar on the right
#     cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.3]) # [left, bottom, width, height]
#     cbar = plt.colorbar(sc, cax=cbar_ax)
#     cbar.set_label('Mixing Score (0=Pure, 1=Mixed)', fontsize=12)

#     plt.suptitle(f"Spatial Patterning Gradient (Radius {RADIUS}µm)", fontsize=18)
#     plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree

# --- 1. MANUAL CONFIGURATION: PASTE YOUR BEST FILES HERE ---
# You must paste the FULL path for one representative file per concentration.
files_map = {
    0:    "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/0dox_GATA6-HA_001.csv",
    10:   "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/10dox_GATA6-HA_001.csv",
    25:   "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/25dox_GATA6-HA_001.csv",
    50:   "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/50dox_GATA6-HA_001.csv",
    100:  "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/100dox_GATA6-HA_001.csv",
    250:  "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/250dox_GATA6-HA_001.csv",
    500:  "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/500dox_GATA6-HA_001.csv",
    1000: "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/1000dox_GATA6-HA_001.csv"
}

RADIUS = 50.0
Z_THICKNESS = 40.0
VIEW_WINDOW = 700 # +/- 700 microns (Total box size 1400x1400)

# --- 2. PROCESSING FUNCTIONS ---
def process_organoid(filename):
    try:
        df = pd.read_csv(filename)
    except:
        print(f"File not found: {filename}")
        return None

    if 'cell_type_dapi_adusted' not in df.columns: return None
    
    # Filter valid cells
    df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])].copy()
    if len(df) < 100: return None

    # --- STEP A: CENTER THE ORGANOID ---
    # Shift X and Y so the median (center of mass) is at (0,0)
    df['X'] = df['X'] - df['X'].median()
    df['Y'] = df['Y'] - df['Y'].median()
    df['Z'] = df['Z'] - df['Z'].median() # Center Z too for slicing

    # --- STEP B: REMOVE SATELLITE COLONIES (The "Segregation" Fix) ---
    # Keep only cells within 600um of the center. 
    # This deletes the "second blob" if it's far away.
    df = df[ (df['X']**2 + df['Y']**2) < 600**2 ]

    # --- STEP C: VIRTUAL SLICE ---
    # Since we centered Z at 0, we just take -20 to +20
    df_slice = df[ (df['Z'] > -Z_THICKNESS/2) & (df['Z'] < Z_THICKNESS/2) ].copy()
    
    if len(df_slice) < 10: return None

    # --- STEP D: CALCULATE MIXING ---
    coords = df_slice[['X', 'Y']].values
    tree = KDTree(coords)
    local_scores = []
    
    for idx, row in df_slice.iterrows():
        indices = tree.query_ball_point([row['X'], row['Y']], r=RADIUS)
        neighbors = df_slice.iloc[indices]
        my_type = row['cell_type_dapi_adusted']
        foreign_count = len(neighbors[neighbors['cell_type_dapi_adusted'] != my_type])
        total_neighbors = len(neighbors) - 1
        
        score = foreign_count / total_neighbors if total_neighbors > 0 else 0.0
        local_scores.append(score)
        
    df_slice['local_mixing'] = local_scores
    return df_slice

# --- 3. PLOTTING ---
dox_levels = sorted(files_map.keys())
n_cols = len(dox_levels)

# Setup Grid
fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7), constrained_layout=True)

# Handle single column case
if n_cols == 1: axes = np.array([ [axes[0]], [axes[1]] ])

for i, dox in enumerate(dox_levels):
    print(f"Processing {dox} Dox...")
    df_slice = process_organoid(files_map[dox])
    
    if df_slice is not None:
        # ROW 1: ANATOMY
        ax_top = axes[0, i]
        sns.scatterplot(data=df_slice, x='X', y='Y', hue='cell_type_dapi_adusted', 
                        palette={2.0:'#d62728', 3.0:'#1f77b4'}, s=5, 
                        legend=False, ax=ax_top)
        
        # FIXED ZOOM & SIZE
        ax_top.set_xlim(-VIEW_WINDOW, VIEW_WINDOW)
        ax_top.set_ylim(-VIEW_WINDOW, VIEW_WINDOW)
        ax_top.set_aspect('equal')
        ax_top.set_title(f"{dox} ng/mL", fontsize=14, fontweight='bold')
        ax_top.axis('off') # Hides the box lines for a cleaner look

        # ROW 2: HEATMAP
        ax_bot = axes[1, i]
        sc = ax_bot.scatter(df_slice['X'], df_slice['Y'], c=df_slice['local_mixing'], 
                            cmap='viridis', s=5, vmin=0, vmax=1.0)
        
        # FIXED ZOOM & SIZE
        ax_bot.set_xlim(-VIEW_WINDOW, VIEW_WINDOW)
        ax_bot.set_ylim(-VIEW_WINDOW, VIEW_WINDOW)
        ax_bot.set_aspect('equal')
        ax_bot.axis('off')

# Colorbar
cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.3])
plt.colorbar(sc, cax=cbar_ax, label='Mixing Score (0=Pure, 1=Mixed)')

plt.suptitle(f"Spatial Patterning Gradient (Center Slice, Fixed Scale)", fontsize=18)
plt.show()