import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# # Configuration
# base_path = "/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis/GATA-HA_Rep1-3"
# dox_levels = [0, 10, 25, 50, 100, 250, 500, 1000]
# eps_range = [20, 30, 40, 50, 60]
# ms_range = [5, 10, 15, 20, 25]

# # Step 1: Map all organoids to their concentration
# dox_groups = {d: glob.glob(os.path.join(base_path, "GATA6-HA_Rep1", f"{d}dox_GATA6-HA_*.csv")) for d in dox_levels}



# def get_aggregated_heatmap(file_list, eps_vals, ms_vals):
#     all_organoid_results = []
    
#     for file_path in file_list:
#         df = pd.read_csv(file_path)
        
#         # Check for column variations automatically
#         cols = df.columns
#         x_col = 'Global X' if 'Global X' in cols else 'X'
#         y_col = 'Global Y' if 'Global Y' in cols else 'Y'
#         z_col = 'Global Z' if 'Global Z' in cols else 'Z'
        
#         target_df = df[df['cell_type_dapi_adusted'] == 2.0]
#         if len(target_df) < 5: continue
            
#         coords = target_df[[x_col, y_col, z_col]].values
#         # Assuming you are clustering Endoderm (Type 2.0)
#         target_df = df[df['cell_type_dapi_adusted'] == 2.0]
#         if len(target_df) < 5: continue
            
#         coords = target_df[['X', 'Y', 'Z']].values
#         grid = np.zeros((len(ms_vals), len(eps_vals)))
        
#         for i, ms in enumerate(ms_vals):
#             for j, eps in enumerate(eps_vals):
#                 db = DBSCAN(eps=eps, min_samples=ms).fit(coords)
#                 n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
#                 grid[i, j] = n_clusters
#         all_organoid_results.append(grid)
    
#     # Average across all organoids found for this Dox level
#     return np.mean(all_organoid_results, axis=0) if all_organoid_results else None

# def get_rep_trajectory(dox, rep_folder, eps_vals):
#     """Calculates cluster count trajectory for all organoids in one replicate."""
#     pattern = os.path.join(base_path, rep_folder, f"{dox}dox_GATA6-HA_*.csv")
#     files = glob.glob(pattern)
    
#     if not files: return None
    
#     all_traj = []
#     for f in files:
#         df = pd.read_csv(f)
#         # Robust column handling
#         cols = {c.replace('Global ', '').replace('Position ', ''): c for c in df.columns}
#         target_df = df[df['cell_type_dapi_adusted'] == 2.0] # Endo
        
#         if len(target_df) < FIXED_MS: continue
#         coords = target_df[[cols['X'], cols['Y'], cols['Z']]].values
        
#         counts = []
#         for eps in eps_vals:
#             db = DBSCAN(eps=eps, min_samples=FIXED_MS).fit(coords)
#             counts.append(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
#         all_traj.append(counts)
        
#     return np.array(all_traj) if all_traj else None




#(1) HEATMAP
# fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
# axes = axes.flatten()

# for i, dox in enumerate(dox_levels):
#     mean_grid = get_aggregated_heatmap(dox_groups[dox], eps_range, ms_range)
#     if mean_grid is not None:
#         sns.heatmap(mean_grid, ax=axes[i], annot=True, fmt=".1f", 
#                     xticklabels=eps_range, yticklabels=ms_range, cmap='magma')
#         axes[i].set_title(f"{dox} ng/mL (Mean of {len(dox_groups[dox])} Organoids)")

# plt.suptitle("Aggregated DBSCAN Parameter Optimization (Mean Cluster Counts)")
# plt.show()




#(2) Line plot with 
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# --- 1. CONFIGURATION ---
# Base path updated to match your folder structure screenshot
base_path = "/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis/GATA-HA_Rep1-3"
reps = ["GATA6-HA_Rep1", "GATA6-HA_Rep2", "GATA6-HA_Rep3"]
dox_levels = [0, 10, 25, 50, 100, 250, 500, 1000]

# Parameters for Trajectory sweep
eps_sweep = np.arange(10, 100, 5) # Detailed sweep from 10 to 100um
FIXED_MS = 20                     # Hold Min_Samples constant to see Epsilon effect

def get_rep_trajectory(dox, rep_folder, eps_vals):
    """Calculates cluster count trajectory for all organoids in one replicate."""
    pattern = os.path.join(base_path, rep_folder, f"{dox}dox_GATA6-HA_*.csv")
    files = glob.glob(pattern)
    
    if not files: return None
    
    all_traj = []
    for f in files:
        df = pd.read_csv(f)
        # Robust column handling
        cols = {c.replace('Global ', '').replace('Position ', ''): c for c in df.columns}
        target_df = df[df['cell_type_dapi_adusted'] == 2.0] # Endo
        
        if len(target_df) < FIXED_MS: continue
        coords = target_df[[cols['X'], cols['Y'], cols['Z']]].values
        
        counts = []
        for eps in eps_vals:
            db = DBSCAN(eps=eps, min_samples=FIXED_MS).fit(coords)
            counts.append(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
        all_traj.append(counts)
        
    return np.array(all_traj) if all_traj else None

# --- 2. EXECUTION & PLOTTING ---
fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
axes = axes.flatten()

colors = {'GATA6-HA_Rep1': '#1f77b4', 'GATA6-HA_Rep2': '#ff7f0e', 'GATA6-HA_Rep3': '#2ca02c'}

print(f"Starting Trajectory Sweep (Min_Samples={FIXED_MS})...")

for i, dox in enumerate(dox_levels):
    ax = axes[i]
    ax.set_title(f"{dox} ng/mL Dox", fontsize=14, fontweight='bold')
    
    for rep in reps:
        traj_data = get_rep_trajectory(dox, rep, eps_sweep)
        
        if traj_data is not None:
                    mean_vals = np.mean(traj_data, axis=0)
                    std_vals = np.std(traj_data, axis=0)
                    
                    # Use errorbar instead of separate plot and fill_between
                    ax.errorbar(
                        eps_sweep, 
                        mean_vals, 
                        yerr=std_vals, 
                        label=f"{rep} (n={len(traj_data)})", 
                        color=colors[rep], 
                        lw=1.5,           # Line width
                        marker='o',       # Marker shape
                        markersize=4,     # Size of marker
                        capsize=3,        # Adds horizontal "caps" to the error bars
                        alpha=0.8,        # Slight transparency for overlapping lines
                        elinewidth=1      # Thickness of the error bar lines
                    )
    
    ax.set_xlabel("Epsilon (Radius µm)")
    ax.set_ylabel("Cluster Count")
    ax.grid(True, alpha=0.3)
    if i == 0: ax.legend()

plt.suptitle(f"Parameter Sensitivity: Cluster Count vs Epsilon (Min_Samples={FIXED_MS})\nShaded Area = STD across Organoids", fontsize=20)
plt.show()