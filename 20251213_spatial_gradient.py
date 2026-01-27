# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial import KDTree
# from scipy.stats import entropy

# # --- 1. CONFIGURATION ---
# # Paste the paths to your representative 0 Dox (Patterned) and 1000 Dox (Mixed) files
# file_low_dox  = "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/0dox_GATA6-HA_001.csv"
# file_high_dox = "data/GATA6-HA_Rep1-3/GATA6-HA_Rep1/1000dox_GATA6-HA_001.csv"

# # The radii to scan (from 10um to 200um in steps of 10)
# RADII_LIST = np.arange(10, 300, 10)

# # --- 2. ENTROPY FUNCTION ---
# def calculate_spatial_entropy(counts):
#     """Calculates Shannon Entropy in bits for a list of counts [n_typeA, n_typeB]"""
#     total = sum(counts)
#     if total == 0: return 0.0
#     probs = [c / total for c in counts]
#     # base=2 gives units in "bits" (Max entropy for 2 classes = 1.0)
#     return entropy(probs, base=2) 

# def get_entropy_gradient(filename, radii):
#     try:
#         df = pd.read_csv(filename)
#     except:
#         print(f"Error reading {filename}")
#         return None, None

#     if 'cell_type_dapi_adusted' not in df.columns: return None, None
    
#     # Filter for Endo (2.0) and Meso (3.0)
#     df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])]
    
#     # Build Tree
#     coords = df[['X', 'Y', 'Z']].values
#     tree = KDTree(coords)
    
#     avg_entropies = []
    
#     # --- THE SCANNING LOOP ---
#     # We calculate the average entropy of the *neighborhoods* at each radius
#     for r in radii:
#         # Optimization: Sample 1000 cells if the dataset is huge (>5000) to speed it up
#         if len(coords) > 5000:
#             sample_indices = np.random.choice(len(coords), 1000, replace=False)
#             query_points = coords[sample_indices]
#         else:
#             query_points = coords

#         neighbors_list = tree.query_ball_point(query_points, r=r)
        
#         r_entropies = []
#         for indices in neighbors_list:
#             if len(indices) < 2: continue # Skip empty neighborhoods
            
#             # Get types of neighbors
#             neighbor_types = df.iloc[indices]['cell_type_dapi_adusted'].values
#             n_endo = np.sum(neighbor_types == 2.0)
#             n_meso = np.sum(neighbor_types == 3.0)
            
#             # Calculate Entropy for this specific cell's neighborhood
#             H = calculate_spatial_entropy([n_endo, n_meso])
#             r_entropies.append(H)
            
#         # Average entropy for this radius
#         if len(r_entropies) > 0:
#             avg_entropies.append(np.mean(r_entropies))
#         else:
#             avg_entropies.append(0.0)
            
#     return radii, avg_entropies

# # --- 3. RUN ANALYSIS ---
# print("Calculating Gradient for Low Dox...")
# r_low, ent_low = get_entropy_gradient(file_low_dox, RADII_LIST)

# print("Calculating Gradient for High Dox...")
# r_high, ent_high = get_entropy_gradient(file_high_dox, RADII_LIST)

# # --- 4. PLOTTING THE "PANEL E" ---
# plt.figure(figsize=(8, 6))

# if r_low is not None:
#     plt.plot(r_low, ent_low, marker='o', linewidth=2.5, label='0 ng/mL Dox (Low Noise)', color='#1f77b4') # Blue

# if r_high is not None:
#     plt.plot(r_high, ent_high, marker='o', linewidth=2.5, label='1000 ng/mL Dox (High Noise)', color='#d62728') # Red

# # Aesthetics
# plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Max Entropy (Random 50/50)')
# plt.title('Entropy Gradient: The Scale of Disorder', fontsize=16)
# plt.xlabel('Neighborhood Radius (µm)', fontsize=14)
# plt.ylabel('Spatial Shannon Entropy (Bits)', fontsize=14)
# plt.ylim(0, 1.1)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=12)

# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import entropy

# --- 1. CONFIGURATION ---
# ### [CHANGED] Replaced single variables with a Dictionary of all files
# PASTE YOUR FULL PATHS HERE
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

# The radii to scan (from 10um to 300um in steps of 10)
RADII_LIST = np.arange(10, 300, 10)

# --- 2. ENTROPY FUNCTION (Unchanged) ---
def calculate_spatial_entropy(counts):
    """Calculates Shannon Entropy in bits for a list of counts [n_typeA, n_typeB]"""
    total = sum(counts)
    if total == 0: return 0.0
    probs = [c / total for c in counts]
    # base=2 gives units in "bits" (Max entropy for 2 classes = 1.0)
    return entropy(probs, base=2) 

def get_entropy_gradient(filename, radii):
    try:
        df = pd.read_csv(filename)
    except:
        print(f"Error reading {filename}")
        return None, None

    if 'cell_type_dapi_adusted' not in df.columns: return None, None
    
    # Filter for Endo (2.0) and Meso (3.0)
    df = df[df['cell_type_dapi_adusted'].isin([2.0, 3.0])]
    
    # Build Tree
    coords = df[['X', 'Y', 'Z']].values
    tree = KDTree(coords)
    
    avg_entropies = []
    
    # --- THE SCANNING LOOP ---
    for r in radii:
        if len(coords) > 5000:
            sample_indices = np.random.choice(len(coords), 1000, replace=False)
            query_points = coords[sample_indices]
        else:
            query_points = coords

        neighbors_list = tree.query_ball_point(query_points, r=r)
        
        r_entropies = []
        for indices in neighbors_list:
            if len(indices) < 2: continue
            
            neighbor_types = df.iloc[indices]['cell_type_dapi_adusted'].values
            n_endo = np.sum(neighbor_types == 2.0)
            n_meso = np.sum(neighbor_types == 3.0)
            
            H = calculate_spatial_entropy([n_endo, n_meso])
            r_entropies.append(H)
            
        if len(r_entropies) > 0:
            avg_entropies.append(np.mean(r_entropies))
        else:
            avg_entropies.append(0.0)
            
    return radii, avg_entropies

# --- 3. RUN ANALYSIS ---
# ### [CHANGED] Loop through the dictionary to process all files
gradient_results = {}
dox_levels = sorted(files_map.keys()) # Ensure order (0, 10, 25...)

print("Calculating Entropy Gradients for all concentrations...")
for dox in dox_levels:
    path = files_map[dox]
    print(f"  Processing {dox} ng/mL...")
    r_vals, ent_vals = get_entropy_gradient(path, RADII_LIST)
    
    if r_vals is not None:
        gradient_results[dox] = (r_vals, ent_vals)

# --- 4. PLOTTING ---
plt.figure(figsize=(10, 7))

# ### [CHANGED] Generate a dynamic color palette (Blue -> Red/Yellow)
# We use 'viridis' or 'plasma' to distinguish the lines clearly
colors = plt.cm.jet(np.linspace(0, 1, len(gradient_results)))

# ### [CHANGED] Loop to plot every line found in gradient_results
for i, dox in enumerate(gradient_results.keys()):
    r_vals, ent_vals = gradient_results[dox]
    
    plt.plot(r_vals, ent_vals, marker='o', markersize=4, linewidth=2, 
             label=f'{dox} ng/mL', color=colors[i], alpha=0.8)

# Aesthetics
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Max Randomness')
plt.title('Entropy Gradient: Loss of Domain Size vs. Noise', fontsize=16)
plt.xlabel('Neighborhood Radius (µm)', fontsize=14)
plt.ylabel('Spatial Shannon Entropy (Bits)', fontsize=14)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend(title="Dox Concentration", fontsize=10, loc='lower right')

plt.tight_layout()
plt.show()