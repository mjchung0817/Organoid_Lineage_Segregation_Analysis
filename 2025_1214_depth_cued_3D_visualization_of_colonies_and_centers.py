# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import DBSCAN

# # --- 1. CONFIGURATION ---
# files_map = {
#     0:    "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/0dox_GATA6-HA_001.csv",
#     1000: "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/1000dox_GATA6-HA_001.csv"
# }

# ISLAND_EPS = 40.0       
# MIN_ISLAND_SIZE = 10    

# # --- 2. HELPER: Z-BASED ALPHA MAPPING ---
# def get_depth_cued_colors(z_values, base_colors_rgb, min_alpha, max_alpha):
#     """
#     Takes an array of Z coordinates and returns an RGBA array where 
#     Alpha scales linearly with Z (Depth).
#     Higher Z (Front) = More Opaque. Lower Z (Back) = More Transparent.
#     """
#     # Normalize Z to 0.0 - 1.0 range
#     z_min, z_max = z_values.min(), z_values.max()
#     if z_max == z_min: 
#         norm_z = np.ones_like(z_values) # Flat Z
#     else:
#         norm_z = (z_values - z_min) / (z_max - z_min)
        
#     # Create the Alpha channel (Equation: alpha = min + (z * range))
#     alphas = min_alpha + (norm_z * (max_alpha - min_alpha))
    
#     # Combine Base RGB with Calculated Alpha
#     # base_colors_rgb should be an (N, 3) or (N, 4) array
#     if base_colors_rgb.shape[1] == 4:
#         base_colors_rgb = base_colors_rgb[:, :3] # Strip existing alpha if present
        
#     rgba = np.column_stack((base_colors_rgb, alphas))
#     return rgba

# # --- 3. 3D PLOTTING FUNCTION ---
# def visualize_depth_cued(filename, dox_conc, ax):
#     try:
#         df = pd.read_csv(filename)
#     except:
#         return

#     if 'cell_type_dapi_adusted' not in df.columns: return
    
#     endo_df = df[df['cell_type_dapi_adusted'] == 2.0].copy()
#     if len(endo_df) < MIN_ISLAND_SIZE: return

#     # --- A. MATH (DBSCAN) ---
#     coords = endo_df[['X', 'Y', 'Z']].values
#     db = DBSCAN(eps=ISLAND_EPS, min_samples=MIN_ISLAND_SIZE).fit(coords)
#     endo_df['island_id'] = db.labels_
    
#     clean_islands = endo_df[endo_df['island_id'] != -1]
#     centroids_3d = clean_islands.groupby('island_id')[['X', 'Y', 'Z']].mean().reset_index()
#     n_islands = len(centroids_3d)

#     # --- B. VISUALIZATION (DEPTH CUING) ---
    
#     # 1. CELLS: "Ghost Mist"
#     # Map Island IDs to Base RGB colors using a colormap (Tab20)
#     cmap = plt.get_cmap('tab20')
#     norm_ids = clean_islands['island_id'] / (clean_islands['island_id'].max() + 1)
#     base_rgb_cells = cmap(norm_ids) # Returns RGBA
    
#     # Calculate Depth Alpha (Range: 0.01 faint back -> 0.15 visible front)
#     rgba_cells = get_depth_cued_colors(clean_islands['Z'].values, base_rgb_cells, 
#                                        min_alpha=0.01, max_alpha=0.15)
    
#     ax.scatter(clean_islands['X'], clean_islands['Y'], clean_islands['Z'], 
#                c=rgba_cells, s=10, depthshade=False) # depthshade=False because we did it manually

#     # 2. CENTERS: "Solid Markers"
#     # Base color Black (0,0,0) for all centers
#     base_rgb_centers = np.zeros((len(centroids_3d), 3)) 
    
#     # Calculate Depth Alpha (Range: 0.3 faded back -> 1.0 solid front)
#     rgba_centers = get_depth_cued_colors(centroids_3d['Z'].values, base_rgb_centers, 
#                                          min_alpha=0.3, max_alpha=1.0)

#     ax.scatter(centroids_3d['X'], centroids_3d['Y'], centroids_3d['Z'], 
#                c=rgba_centers, marker='X', s=150, linewidth=2, depthshade=False,
#                label='Centers (Depth Cued)')

#     # Formatting
#     ax.set_title(f"{dox_conc} ng/mL Dox\n({n_islands} Islands)", fontsize=14)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z (Depth)')
    
#     # Equal aspect ratio hack
#     all_vals = np.concatenate([endo_df['X'], endo_df['Y'], endo_df['Z']])
#     ax.set_xlim(all_vals.min(), all_vals.max())
#     ax.set_ylim(all_vals.min(), all_vals.max())
#     ax.set_zlim(all_vals.min(), all_vals.max())

# # --- 4. EXECUTE ---
# fig = plt.figure(figsize=(16, 8))

# ax1 = fig.add_subplot(121, projection='3d')
# print("Rendering Depth-Cued Model for 0 Dox...")
# visualize_depth_cued(files_map[0], 0, ax1)

# ax2 = fig.add_subplot(122, projection='3d')
# print("Rendering Depth-Cued Model for 1000 Dox...")
# visualize_depth_cued(files_map[1000], 1000, ax2)

# plt.suptitle("3D Fragmentation: Depth-Cued Visualization", fontsize=18)
# plt.tight_layout()
# plt.show()










##Modifications: image/[dox], clusters considering both, fine tuning epsilon  

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import DBSCAN

# # --- 1. CONFIGURATION ---
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

# ISLAND_EPS = 25.0       
# MIN_ISLAND_SIZE = 10    

# # --- 2. DEPTH-CUING HELPER ---
# def get_depth_cued_colors(z_values, base_color_rgb, min_alpha, max_alpha):
#     z_min, z_max = z_values.min(), z_values.max()
#     if z_max > z_min:
#         norm_z = (z_values - z_min) / (z_max - z_min)
#     else:
#         norm_z = np.ones_like(z_values)
#     alphas = min_alpha + (norm_z * (max_alpha - min_alpha))
    
#     # Broadcast base color to all points
#     colors = np.zeros((len(z_values), 4))
#     colors[:, :3] = base_color_rgb
#     colors[:, 3] = alphas
#     return colors

# # --- 3. DUAL-POPULATION PROCESSING ---
# def visualize_dual_clusters(filename, dox_conc, ax):
#     try:
#         df = pd.read_csv(filename)
#     except:
#         return

#     if 'cell_type_dapi_adusted' not in df.columns: return

#     # --- LOOP THROUGH BOTH CELL TYPES ---
#     # Type 2 (Endo) = Red, Type 3 (Meso) = Blue
#     configs = [
#         {'type': 2.0, 'name': 'Endo', 'color': [1, 0, 0], 'marker': 'X'}, # Red
#         {'type': 3.0, 'name': 'Meso', 'color': [0, 0, 1], 'marker': 'o'}  # Blue
#     ]
    
#     total_islands_str = []

#     for cfg in configs:
#         sub_df = df[df['cell_type_dapi_adusted'] == cfg['type']].copy()
#         if len(sub_df) < MIN_ISLAND_SIZE: continue

#         # 1. Run DBSCAN
#         coords = sub_df[['X', 'Y', 'Z']].values
#         db = DBSCAN(eps=ISLAND_EPS, min_samples=MIN_ISLAND_SIZE).fit(coords)
#         sub_df['island_id'] = db.labels_
        
#         # 2. Filter Noise & Find Centers
#         clean = sub_df[sub_df['island_id'] != -1]
#         if len(clean) == 0: continue
        
#         centroids = clean.groupby('island_id')[['X', 'Y', 'Z']].mean().reset_index()
#         n_islands = len(centroids)
#         total_islands_str.append(f"{cfg['name']}: {n_islands}")

#         # 3. Plot Cells ("Ghost Mode")
#         # We use a SINGLE color (Red or Blue) but vary the alpha for depth
#         rgba_cells = get_depth_cued_colors(clean['Z'].values, np.array(cfg['color']), 0.01, 0.05)
#         ax.scatter(clean['X'], clean['Y'], clean['Z'], 
#                    c=rgba_cells, s=2, depthshade=False)

#         # 4. Plot Centers (Solid Mode)
#         rgba_centers = get_depth_cued_colors(centroids['Z'].values, np.array(cfg['color']), 0.5, 1.0)
#         ax.scatter(centroids['X'], centroids['Y'], centroids['Z'], 
#                    c=rgba_centers, marker=cfg['marker'], s=40, linewidth=1.5, 
#                    depthshade=False, label=cfg['name'])

#     # Formatting
#     title_str = f"{dox_conc} ng/mL\n" + " | ".join(total_islands_str)
#     ax.set_title(title_str, fontsize=10, fontweight='bold')
#     ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
#     # Scale fix
#     all_vals = np.concatenate([df['X'], df['Y'], df['Z']])
#     ax.set_xlim(all_vals.min(), all_vals.max())
#     ax.set_ylim(all_vals.min(), all_vals.max())
#     ax.set_zlim(all_vals.min(), all_vals.max())

# # --- 4. EXECUTION ---
# dox_levels = sorted(files_map.keys())
# n_cols = 4
# n_rows = 2
# fig = plt.figure(figsize=(16, 8))

# for i, dox in enumerate(dox_levels):
#     ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
#     print(f"Processing Dual Clusters for {dox} ng/mL...")
#     visualize_dual_clusters(files_map[dox], dox, ax)

# plt.suptitle(f"Dual-Lineage Cluster Analysis (Red=Endo, Blue=Meso)", fontsize=16)
# plt.tight_layout()
# plt.show()









# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import DBSCAN

# # --- 1. CONFIGURATION ---
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

# # --- NEW COLUMN DEFINITION ---
# # UPDATED: We now use the normalized column per user instruction
# TYPE_COL = 'cell_type_log1p_normed'

# # --- PARAMETERS ---
# ISLAND_EPS = 25.0       
# MIN_ISLAND_SIZE = 10 

# # Set True to treat the organoid as one object (Physical Integrity)
# # Set False to split by Cell Type (Pattern Integrity) <- RECOMMENDED FOR GATA6
# ANALYZE_ALL_CELLS = False 

# # --- HELPER: DEPTH CUING ---
# def get_depth_cued_colors(z_values, base_color_rgb, min_alpha, max_alpha):
#     z_min, z_max = z_values.min(), z_values.max()
#     norm_z = (z_values - z_min) / (z_max - z_min) if z_max > z_min else np.ones_like(z_values)
#     alphas = min_alpha + (norm_z * (max_alpha - min_alpha))
#     colors = np.zeros((len(z_values), 4))
#     colors[:, :3] = base_color_rgb
#     colors[:, 3] = alphas
#     return colors

# # --- VISUALIZATION FUNCTION ---
# def visualize_clusters_new_norm(filename, dox_conc, ax):
#     try:
#         df = pd.read_csv(filename)
#     except:
#         return

#     if TYPE_COL not in df.columns:
#         print(f"CRITICAL ERROR: Column '{TYPE_COL}' not found in {dox_conc} Dox file!")
#         return

#     # --- DIAGNOSTIC: Print Cell Types found ---
#     # This helps you check if "2" is still Endo and "3" is still Meso
#     if dox_conc == 0: 
#         counts = df[TYPE_COL].value_counts()
#         print(f"\n--- DIAGNOSTIC: Cell Types in {TYPE_COL} (0 Dox) ---")
#         print(counts)
#         print("---------------------------------------------------\n")

#     # Define what we are plotting
#     if ANALYZE_ALL_CELLS:
#         # Treat all cells as one group
#         groups = [{'name': 'All Cells', 'df': df, 'color': [0.5, 0.5, 0.5], 'marker': 'o'}]
#     else:
#         # Split by lineage (ASSUMING 2=Endo, 3=Meso based on previous logic)
#         # CHECK THE PRINT OUTPUT to confirm these numbers match your new column!
#         groups = [
#             {'name': 'Endo', 'df': df[df[TYPE_COL] == 2.0], 'color': [1, 0, 0], 'marker': 'X'}, # Red
#             {'name': 'Meso', 'df': df[df[TYPE_COL] == 3.0], 'color': [0, 0, 1], 'marker': 'o'}  # Blue
#         ]

#     title_parts = []

#     for grp in groups:
#         sub_df = grp['df'].copy()
#         if len(sub_df) < MIN_ISLAND_SIZE: continue

#         # 1. Run DBSCAN
#         coords = sub_df[['X', 'Y', 'Z']].values
#         db = DBSCAN(eps=ISLAND_EPS, min_samples=MIN_ISLAND_SIZE).fit(coords)
#         sub_df['island_id'] = db.labels_
        
#         # 2. Filter Noise & Find Centers
#         clean = sub_df[sub_df['island_id'] != -1]
#         if len(clean) == 0: continue
        
#         centroids = clean.groupby('island_id')[['X', 'Y', 'Z']].mean().reset_index()
#         n_islands = len(centroids)
#         title_parts.append(f"{grp['name']}: {n_islands}")

#         # 3. Plot Cells ("Ghost Mode")
#         rgba_cells = get_depth_cued_colors(clean['Z'].values, np.array(grp['color']), 0.01, 0.05)
#         ax.scatter(clean['X'], clean['Y'], clean['Z'], 
#                    c=rgba_cells, s=2, depthshade=False)

#         # 4. Plot Centers (Solid Mode)
#         rgba_centers = get_depth_cued_colors(centroids['Z'].values, np.array(grp['color']), 0.5, 1.0)
#         ax.scatter(centroids['X'], centroids['Y'], centroids['Z'], 
#                    c=rgba_centers, marker=grp['marker'], s=40, linewidth=1.5, 
#                    depthshade=False)

#     # Formatting
#     ax.set_title(f"{dox_conc} ng/mL\n" + " | ".join(title_parts), fontsize=10, fontweight='bold')
#     ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
#     # Scale fix
#     all_vals = np.concatenate([df['X'], df['Y'], df['Z']])
#     ax.set_xlim(all_vals.min(), all_vals.max())
#     ax.set_ylim(all_vals.min(), all_vals.max())
#     ax.set_zlim(all_vals.min(), all_vals.max())

# # --- EXECUTION ---
# dox_levels = sorted(files_map.keys())
# n_cols = 4
# n_rows = 2
# fig = plt.figure(figsize=(16, 8))

# for i, dox in enumerate(dox_levels):
#     ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
#     print(f"Processing {dox} ng/mL with {TYPE_COL}...")
#     visualize_clusters_new_norm(files_map[dox], dox, ax)

# plt.suptitle(f"Analysis using: {TYPE_COL} (eps={ISLAND_EPS}µm)", fontsize=16)
# plt.tight_layout()
# plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# --- 1. CONFIGURATION ---
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

# PARAMETERS
ISLAND_EPS = 50.0       
MIN_SAMPLES = 3         

# --- 2. DATA PREP ---
def get_data_for_mode(df, mode):
    df = df.copy()
    if 'cell_type_dapi_adusted' not in df.columns: return None, None
    
    if mode == 'Endo_vs_Meso':
        # Filter for only Types 2 and 3
        df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])]
        configs = [
            {'type': 2.0, 'color': [1, 0, 0], 'label': 'Endo', 'zorder': 10}, # Red
            {'type': 3.0, 'color': [0, 0, 1], 'label': 'Meso', 'zorder': 5}   # Blue
        ]
    elif mode == 'Endo_vs_All':
        # Endo (2.0) vs Background (99.0)
        df['cell_type_dapi_adusted'] = np.where(df['cell_type_dapi_adusted']==2.0, 2.0, 99.0)
        configs = [
            {'type': 2.0, 'color': [1, 0, 0], 'label': 'Endo', 'zorder': 10}, # Red
            {'type': 99.0, 'color': [0.85, 0.85, 0.85], 'label': 'Others', 'zorder': 1} # Gray
        ]
    elif mode == 'Meso_vs_All':
        df['cell_type_dapi_adusted'] = np.where(df['cell_type_dapi_adusted']==3.0, 3.0, 99.0)
        configs = [
            {'type': 3.0, 'color': [0, 0, 1], 'label': 'Meso', 'zorder': 10}, # Blue
            {'type': 99.0, 'color': [0.85, 0.85, 0.85], 'label': 'Others', 'zorder': 1} # Gray
        ]
    return df, configs

# --- 3. HELPER: DEPTH ALPHA ---
def get_depth_colors(z_values, base_rgb, min_alpha=0.1, max_alpha=1.0):
    z_min, z_max = z_values.min(), z_values.max()
    if z_max > z_min:
        norm_z = (z_values - z_min) / (z_max - z_min)
    else:
        norm_z = np.ones_like(z_values)
    alphas = min_alpha + (norm_z * (max_alpha - min_alpha))
    colors = np.zeros((len(z_values), 4))
    colors[:, :3] = base_rgb
    colors[:, 3] = alphas
    return colors

# --- 4. PLOTTING FUNCTION ---
def plot_xray(df, configs, ax, title):
    stats = []
    
    for cfg in configs:
        sub = df[df['cell_type_dapi_adusted'] == cfg['type']].copy()
        
        # --- IF TARGET TYPE (Run Clustering) ---
        if cfg['type'] != 99.0:
            if len(sub) >= MIN_SAMPLES:
                coords = sub[['X', 'Y', 'Z']].values 
                db = DBSCAN(eps=ISLAND_EPS, min_samples=MIN_SAMPLES).fit(coords)
                sub['island'] = db.labels_
                
                # 1. SPLIT DATA: Clusters vs. Noise
                noise = sub[sub['island'] == -1]
                clean = sub[sub['island'] != -1]
                
                # 2. PLOT NOISE (Faint, so you can still see the cells)
                if not noise.empty:
                    noise_colors = get_depth_colors(noise['Z'].values, np.array(cfg['color']), 0.05, 0.2)
                    ax.scatter(noise['X'], noise['Y'], c=noise_colors, s=5, linewidth=0, zorder=1)

                # 3. PLOT CLUSTERS (Solid)
                if not clean.empty:
                    n_islands = len(clean['island'].unique())
                    stats.append(f"{cfg['label']}: {n_islands}")
                    
                    clean_colors = get_depth_colors(clean['Z'].values, np.array(cfg['color']), 0.2, 1.0)
                    ax.scatter(clean['X'], clean['Y'], c=clean_colors, s=10, linewidth=0, zorder=cfg['zorder'])
                    
                    # 4. PLOT CENTERS (Adjusted Size)
                    centers = clean.groupby('island')[['X', 'Y', 'Z']].mean()
                    
                    # Make stars smaller so they don't block the view (s=30 instead of 200)
                    ax.scatter(
                        centers['X'], centers['Y'], 
                        c='yellow', edgecolor='black', marker='*', 
                        s=40, # <--- MUCH SMALLER SIZE
                        linewidth=0.5, zorder=100
                    )
            else:
                # Not enough points to cluster, just plot them as noise
                colors = get_depth_colors(sub['Z'].values, np.array(cfg['color']), 0.1, 0.5)
                ax.scatter(sub['X'], sub['Y'], c=colors, s=5, linewidth=0, zorder=1)
                stats.append(f"{cfg['label']}: <{MIN_SAMPLES}")

        # --- IF BACKGROUND ---
        else:
            if len(sub) > 0:
                colors = get_depth_colors(sub['Z'].values, np.array(cfg['color']), 0.1, 0.5)
                ax.scatter(sub['X'], sub['Y'], c=colors, s=5, linewidth=0, zorder=1)

    # Title & Formatting
    final_title = f"{title}\n{' | '.join(stats)}" if stats else title
    ax.set_title(final_title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off') 
    
    all_vals = np.concatenate([df['X'], df['Y']])
    ax.set_xlim(all_vals.min(), all_vals.max())
    ax.set_ylim(all_vals.min(), all_vals.max())
# --- 5. EXECUTION ---
CURRENT_MODE = 'Endo_vs_Meso' # Toggle: 'Endo_vs_Meso', 'Endo_vs_All'
dox_levels = sorted(files_map.keys())

fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
axes = axes.flatten()

print(f"Generating X-Ray Grid for {CURRENT_MODE}...")

for i, dox in enumerate(dox_levels):
    try:
        raw_df = pd.read_csv(files_map[dox])
        df_proc, configs = get_data_for_mode(raw_df, CURRENT_MODE)
        
        if df_proc is not None:
            plot_xray(df_proc, configs, axes[i], f"{dox} ng/mL")
        else:
            axes[i].text(0.5, 0.5, "No Data", ha='center')
            
    except Exception as e:
        print(f"Error {dox}: {e}")
        axes[i].text(0.5, 0.5, "Error", ha='center')

plt.suptitle(f"Top-Down X-Ray Analysis: {CURRENT_MODE}\n(Yellow Star = Cluster Center | Eps={ISLAND_EPS})", fontsize=16)
plt.show()