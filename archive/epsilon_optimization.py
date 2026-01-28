

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN

# # --- 1. CONFIGURATION ---
# # [MODIFICATION] Updated file dictionary
# files_map = {
#     0:    "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/0dox_GATA6-HA_001.csv",
#     10:   "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/10dox_GATA6-HA_001.csv",
#     25:   "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/25dox_GATA6-HA_001.csv",
#     50:   "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/50dox_GATA6-HA_001.csv",
#     100:  "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/100dox_GATA6-HA_001.csv",
#     250:  "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/250dox_GATA6-HA_001.csv",
#     500:  "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/500dox_GATA6-HA_001.csv",
#     1000: "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/1000dox_GATA6-HA_001.csv"
# }

# # [MODIFICATION] requested threshold
# MIN_SAMPLES = 5   
# # [MODIFICATION] Testing range for Epsilon (X-axis of the plot)
# EPS_RANGE = np.arange(5, 100, 5) 

# # --- 2. DATA PREP: THE "ONE VS ALL" TWEAK ---
# def prepare_data(df, mode='Endo_vs_Meso'):
#     """
#     [MODIFICATION] New function to handle the 'One vs All' logic
#     """
#     df = df.copy()
#     if 'cell_type_dapi_adusted' not in df.columns: return None

#     if mode == 'Endo_vs_Meso':
#         # Original logic: Filter for just 2 and 3
#         df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])]
#         return df
        
#     elif mode == 'Endo_vs_All':
#         # [MODIFICATION] Logic for Endo vs All
#         # Keep Endo (2.0) as is. Relabel EVERYONE else to 99.0
#         df['cell_type_dapi_adusted'] = np.where(
#             df['cell_type_dapi_adusted'] == 2.0, 
#             2.0,   # Keep Self
#             99.0   # Aggregate Others
#         )
#         return df

#     elif mode == 'Meso_vs_All':
#         # [MODIFICATION] Logic for Meso vs All
#         df['cell_type_dapi_adusted'] = np.where(
#             df['cell_type_dapi_adusted'] == 3.0, 
#             3.0,   
#             99.0   
#         )
#         return df
        
#     return None

# # --- 3. OPTIMIZATION LOOP ---
# def run_optimization(file_path, mode, ax):
#     raw_df = pd.read_csv(file_path)
#     df = prepare_data(raw_df, mode=mode)
#     if df is None or len(df) == 0: return

#     # Focus on the primary cell type for clustering
#     # If mode is Endo_vs_All, we cluster Endo (2.0)
#     # If mode is Meso_vs_All, we cluster Meso (3.0)
#     focus_type = 3.0 if mode == 'Meso_vs_All' else 2.0
#     subset = df[df['cell_type_dapi_adusted'] == focus_type]
    
#     if len(subset) < MIN_SAMPLES: return

#     coords = subset[['X', 'Y', 'Z']].values
    
#     n_clusters_list = []
    
#     # [MODIFICATION] The Sweep Loop
#     for eps in EPS_RANGE:
#         # Run DBSCAN with changing Epsilon
#         db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(coords)
#         labels = db.labels_
        
#         # Count clusters (excluding noise -1)
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#         n_clusters_list.append(n_clusters)

#     # Plotting
#     ax.plot(EPS_RANGE, n_clusters_list, 'b-o', markersize=4, label='Cluster Count')
#     ax.set_xlabel('Epsilon (Radius µm)')
#     ax.set_ylabel('Number of Detected Clusters', color='b')
#     ax.grid(True, alpha=0.3)
    
#     # Identify the "Knee" visually
#     # (Just taking a simple diff to find where slope flattens)
#     diffs = np.diff(n_clusters_list)
#     # Simple heuristic: find first point where change is small (< 10% of max change)
#     # This draws a vertical line where the curve starts to flatten
#     threshold = np.min(diffs) * 0.1
    
#     return n_clusters_list


# # --- 4. EXECUTION ---
# # [MODIFICATION] Toggle Analysis Mode Here
# CURRENT_MODE = 'Meso_vs_All'  # Options: 'Endo_vs_Meso', 'Endo_vs_All', 'Meso_vs_All'

# # [MODIFICATION] Create a 2x4 Grid for all 8 plots
# fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
# axes = axes.flatten() # Flattens the 2D grid into a 1D list [0...7]

# dox_levels = sorted(files_map.keys())

# print(f"Running Epsilon Optimization for ALL concentrations ({CURRENT_MODE})...")

# for i, dox in enumerate(dox_levels):
#     # Select the subplot
#     ax = axes[i]
    
#     # Title
#     ax.set_title(f"{dox} ng/mL Dox", fontsize=12, fontweight='bold')
    
#     # Run Logic
#     print(f"  Processing {dox} ng/mL...")
#     run_optimization(files_map[dox], CURRENT_MODE, ax)

# # Final Formatting
# plt.suptitle(f"Epsilon Justification: {CURRENT_MODE} (Threshold={MIN_SAMPLES})", fontsize=16)
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# --- 1. CONFIGURATION ---
files_map = {
    0:    "data/GATA6-HA_Rep3/0dox_GATA6-HA_003.csv",
    10:   "data/GATA6-HA_Rep3/10dox_GATA6-HA_003.csv",
    25:   "data/GATA6-HA_Rep3/25dox_GATA6-HA_003.csv",
    50:   "data/GATA6-HA_Rep3/50dox_GATA6-HA_003.csv",
    100:  "data/GATA6-HA_Rep3/100dox_GATA6-HA_003.csv",
    250:  "data/GATA6-HA_Rep3/250dox_GATA6-HA_003.csv",
    500:  "data/GATA6-HA_Rep3/500dox_GATA6-HA_003.csv",
    1000: "data/GATA6-HA_Rep3/1000dox_GATA6-HA_003.csv"
}


# --- 2. PARAMETER RANGES ---
# Adjust these ranges based on your intuition
eps_range = [10, 20, 30, 40, 50, 60, 70, 80]
min_samples_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# --- 3. HELPER FUNCTION ---
def compute_cluster_counts(df, eps_vals, min_samples_vals):
    results = np.zeros((len(min_samples_vals), len(eps_vals)))
    
    # Filter for Endo (Type 2.0) - Adjust if you want Meso
    target_df = df[df['cell_type_dapi_adusted'] == 2.0] 
    
    if len(target_df) < 5:
        return results # Return zeros if not enough data
        
    coords = target_df[['X', 'Y', 'Z']].values
    
    print(f"  Scanning {len(coords)} points...")
    
    for i, ms in enumerate(min_samples_vals):
        for j, eps in enumerate(eps_vals):
            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=ms).fit(coords)
            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            results[i, j] = n_clusters
            
    return results

# --- 4. EXECUTION ---
dox_levels = sorted(files_map.keys())
fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
axes = axes.flatten()

print("Starting Parameter Grid Search...")

for i, dox in enumerate(dox_levels):
    try:
        print(f"Processing {dox} ng/mL...")
        df = pd.read_csv(files_map[dox])
        
        # Compute the grid
        heatmap_data = compute_cluster_counts(df, eps_range, min_samples_range)
        
        # Plot Heatmap
        sns.heatmap(
            heatmap_data, 
            ax=axes[i], 
            annot=True,              # Show the number of clusters in the box
            fmt='g',                 # Generic number format
            xticklabels=eps_range,
            yticklabels=min_samples_range,
            cmap='viridis',
            cbar=False
        )
        
        axes[i].set_title(f"{dox} ng/mL (Endo Clusters)")
        axes[i].set_xlabel("Epsilon (Radius)")
        axes[i].set_ylabel("Min Samples")
        
    except Exception as e:
        print(f"Error {dox}: {e}")
        axes[i].text(0.5, 0.5, "Error", ha='center')

# Add a shared colorbar idea or just title
plt.suptitle(f"DBSCAN Parameter Optimization: Epsilon vs Min_Samples\n(Values = Number of Clusters Detected)", fontsize=16)
plt.show()

