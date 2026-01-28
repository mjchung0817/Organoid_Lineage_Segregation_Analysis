import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from esda.moran import Moran, Moran_Local
from libpysal.weights import DistanceBand
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Manually toggle this path for Ex1, Ex2_high, Ex2_low, or Ex3
BASE_DIR = "data/GATA-HA-BMP4+Wnt5a_Ex3"

# DYNAMIC NAMING: Identifies experiment from folder name (e.g., 'Ex1' or 'Ex3')
EXP_TAG = os.path.basename(BASE_DIR)
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), f"Spatial_Plots_{EXP_TAG}")
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

CHANNELS = {'Endo': 'log1p_normed_546', 'Meso': 'log1p_normed_647'}
EPSILON = 30.0
MIN_SAMPLES = 20

# ==============================================================================
# MATH ENGINE: SPATIAL POWER (sPCA)
# ==============================================================================
def get_spca_loadings(df, channels_list):
    data_std = (df[channels_list] - df[channels_list].mean()) / df[channels_list].std()
    coords = df[['X', 'Y', 'Z']].values
    w = DistanceBand.from_array(coords, threshold=EPSILON, binary=True)
    lagged_data = np.array([np.mean(data_std.values[w.neighbors[i]], axis=0) if len(w.neighbors[i]) > 0 
                           else data_std.values[i] for i in range(len(data_std))])
    pca = PCA(n_components=1)
    pca.fit(lagged_data)
    return dict(zip(channels_list, np.abs(pca.components_[0])))

# ==============================================================================
# DATA AGGREGATION
# ==============================================================================
all_stats = []
file_list = glob.glob(os.path.join(BASE_DIR, "**/*.csv"), recursive=True)

for file_path in file_list:
    fname = os.path.basename(file_path)
    dose_match = re.search(r"(\d+)dox", fname)
    cond_match = re.search(r"\+(.+?)_", fname)
    
    dose = int(dose_match.group(1)) if dose_match else 0
    condition = cond_match.group(1) if cond_match else "Control"
    rep_name = os.path.basename(os.path.dirname(file_path))
    
    try:
        df = pd.read_csv(file_path)
        w = DistanceBand.from_array(df[['X', 'Y', 'Z']].values, threshold=EPSILON, binary=True)
        db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(df[['X', 'Y', 'Z']].values)
        df['cluster'] = db.labels_
        
        spca_results = get_spca_loadings(df, list(CHANNELS.values()))

        for label, channel in CHANNELS.items():
            if channel not in df.columns: continue
            vals = df[channel].values
            if len(vals) < 2 or np.all(vals == vals[0]): continue
            
            mi = Moran(vals, w)
            lisa = Moran_Local(vals, w)
            df['lisa_val'] = lisa.Is
            avg_cluster_lisa = df[df['cluster'] != -1]['lisa_val'].mean()
            
            all_stats.append({
                'Dose': dose, 'Condition': condition, 'Rep': rep_name, 'Lineage': label,
                'Morans_I': mi.I, 'Avg_Cluster_LISA': avg_cluster_lisa, 'Spatial_Power': spca_results[channel]
            })
    except Exception as e:
        print(f"Skipping {fname}: {e}")

stats_df = pd.DataFrame(all_stats)

# ==============================================================================
# VISUALIZATION: PER-CONDITION SEPARATED FILES
# ==============================================================================
sns.set_context("paper", font_scale=1.5)
metrics = [('Morans_I', "Global Moran's I"), 
           ('Avg_Cluster_LISA', "Avg LISA (Within Clusters)"),
           ('Spatial_Power', "Lineage Spatial Power (sPCA)")]

palette = {'Endo': '#d62728', 'Meso': '#1f77b4'}
# Brightness map for faint replicate lines
rep_alphas = {"GATA6-HA_Rep1": 0.15, "GATA6-HA_Rep2": 0.35, "GATA6-HA_Rep3": 0.55}

for cond in stats_df['Condition'].unique():
    cond_df = stats_df[stats_df['Condition'] == cond]
    
    for col, ylabel in metrics:
        plt.figure(figsize=(10, 7))
        
        # 1. Plot individual replicates (Faint dimmer lines)
        for rep in cond_df['Rep'].unique():
            rep_data = cond_df[cond_df['Rep'] == rep]
            sns.lineplot(data=rep_data, x='Dose', y=col, hue='Lineage', ax=plt.gca(), 
                         legend=False, alpha=rep_alphas.get(rep, 0.3), linewidth=1.5, palette=palette)
        
        # 2. Plot mean across org and rep with SE bars
        sns.lineplot(data=cond_df, x='Dose', y=col, hue='Lineage', ax=plt.gca(),
                     marker='o', markersize=10, linewidth=4, errorbar='se', 
                     palette=palette, err_style='bars')
        
        plt.xscale('symlog', linthresh=10)
        plt.xticks([0, 10, 25, 50, 100, 250, 500, 1000])
        plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.title(f"{EXP_TAG} | {cond}\n{ylabel}", fontweight='bold')
        plt.xlim(-1, 1100)
        plt.xlabel("Dox Concentration (ng/mL)")
        
        # FILENAME: Unique per experiment and condition
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{EXP_TAG}_{cond}_{col}.png"), dpi=300)
        plt.show()