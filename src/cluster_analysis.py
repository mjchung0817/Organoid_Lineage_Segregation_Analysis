import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

# --- 1. CONFIGURATION ---
# Points to your Experiment 1 folder
BASE_PATH = "data/GATA-HA_Rep1-3_Ex1"

# Calibration Constants
PARAMS = {
    1.0: {'eps': 30, 'ms': 20, 'name': 'GATA6+'},
    2.0: {'eps': 30, 'ms': 20, 'name': 'Endo'},
    3.0: {'eps': 30, 'ms': 20, 'name': 'Meso'}
}

# --- 2. ENGINE: PER-ORGANOID SPATIAL METRICS ---
def run_exp1_pipeline():
    organoid_records = []
    distance_records = []
    
    # Recursive search to catch all Rep folders in Ex1
    csv_files = glob.glob(os.path.join(BASE_PATH, "**/*.csv"), recursive=True)
    print(f"Scanning {len(csv_files)} baseline organoids...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            # Extract Dox from filename (e.g., '100dox_...')
            dox = int(file_name.split('dox')[0])
            replicate = os.path.basename(os.path.dirname(file_path)) 
            
            # Safe column search for 'adusted' vs 'adjusted'
            target_col = next((c for c in df.columns if 'cell_type' in c.lower()), None)
            if not target_col: continue

            for ct_id, p in PARAMS.items():
                subset = df[df[target_col] == ct_id]
                if len(subset) < p['ms']: continue # Validity gate
                
                coords = subset[['X', 'Y', 'Z']].values
                db = DBSCAN(eps=p['eps'], min_samples=p['ms']).fit(coords)
                labels = db.labels_
                unique_labels = [l for l in np.unique(labels) if l != -1]
                
                cluster_coords_list = []
                if len(unique_labels) > 0:
                    # (A) Cluster Size Logic
                    sizes = [len(coords[labels == l]) for l in unique_labels]
                    organoid_records.append({
                        'Dox_Concentration': dox,
                        'Cell_Type': p['name'],
                        'Avg_Cluster_Size': np.mean(sizes),
                        'File': file_name
                    })
                    
                    # (B) Edge-to-Edge Distance Logic (KDTree)
                    for l in unique_labels:
                        cluster_coords_list.append(coords[labels == l])
                        
                    if len(cluster_coords_list) > 1:
                        pair_distances = []
                        for i in range(len(cluster_coords_list)):
                            tree_i = cKDTree(cluster_coords_list[i])
                            for j in range(i + 1, len(cluster_coords_list)):
                                dist, _ = tree_i.query(cluster_coords_list[j], k=1)
                                pair_distances.append(np.min(dist))
                        
                        distance_records.append({
                            'Dox_Concentration': dox,
                            'Cell_Type': p['name'],
                            'Edge_to_Edge_Distance_um': np.mean(pair_distances)
                        })
        except Exception as e:
            continue

    return pd.DataFrame(organoid_records), pd.DataFrame(distance_records)

# --- 3. EXECUTION AND VISUALIZATION ---
df_size, df_dist = run_exp1_pipeline()

if not df_size.empty:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: Cluster Size (Avg Cells per Cluster)
    sns.lineplot(data=df_size, x='Dox_Concentration', y='Avg_Cluster_Size', hue='Cell_Type', 
                 marker='o', errorbar=('ci', 95), ax=axes[0])
    axes[0].set_title("Cluster Size (Avg Cells per Cluster)")
    axes[0].set_xscale('symlog', linthresh=10)
    axes[0].set_xlim(left=0)

    # Panel 2: Inter-Cluster Separation (Edge-to-Edge)
    sns.lineplot(data=df_dist, x='Dox_Concentration', y='Edge_to_Edge_Distance_um', hue='Cell_Type', 
                 marker='s', errorbar=('ci', 95), ax=axes[1])
    axes[1].set_title("Inter-Cluster Separation (Edge-to-Edge)")
    axes[1].set_xscale('symlog', linthresh=10)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)

    plt.suptitle(f"Exp 1 Baseline Spatial Analysis (Eps: 30, MinSamples: 20)", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()