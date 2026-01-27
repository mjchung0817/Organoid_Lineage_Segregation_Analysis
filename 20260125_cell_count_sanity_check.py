# import pandas as pd
# import glob
# import os

# # Path to Exp 1
# EXP1_PATH = "data/GATA-HA_Rep1-3_Ex1"

# files = glob.glob(os.path.join(EXP1_PATH, "**/*.csv"), recursive=True)

# print(f"{'File Name':<30} | {'Meso (3.0)':<10} | {'Endo (2.0)':<10}")
# print("-" * 55)

# for f in files:
#     if "0dox" in f:
#         df = pd.read_csv(f)
#         # Handle the common typo in your dataset
#         target_col = 'cell_type_dapi_adusted' if 'cell_type_dapi_adusted' in df.columns else 'cell_type_dapi_adjusted'
        
#         counts = df[target_col].value_counts()
#         meso = counts.get(3.0, 0)
#         endo = counts.get(2.0, 0)
        
#         print(f"{os.path.basename(f):<30} | {meso:<10} | {endo:<10}")




import pandas as pd
import glob
import os
import numpy as np
from sklearn.cluster import DBSCAN

# Path to Exp 1
EXP1_PATH = "data/GATA-HA_Rep1-3_Ex1"

# DBSCAN Params from your calibration
EPS, MS = 30, 20 

files = glob.glob(os.path.join(EXP1_PATH, "**/*.csv"), recursive=True)

print(f"{'File Name':<25} | {'Total Type 1':<12} | {'Clusters Found':<15} | {'Status'}")
print("-" * 75)

for f in files:
    # Checking 0dox and 10dox as requested
    if "0dox" in f or "10dox" in f:
        df = pd.read_csv(f)
        target_col = 'cell_type_dapi_adusted' if 'cell_type_dapi_adusted' in df.columns else 'cell_type_dapi_adjusted'
        
        # 1. Get the Raw Population Count
        type1_subset = df[df[target_col] == 1.0]
        total_type1 = len(type1_subset)
        
        # 2. Run DBSCAN to see if they are "Dense" enough
        cluster_count = 0
        status = "Too Sparse"
        
        if total_type1 >= MS:
            coords = type1_subset[['X', 'Y', 'Z']].values
            db = DBSCAN(eps=EPS, min_samples=MS).fit(coords)
            # Count labels that aren't noise (-1)
            cluster_count = len([l for l in np.unique(db.labels_) if l != -1])
            if cluster_count > 0:
                status = "Formed Clusters"
        elif total_type1 > 0:
            status = f"Below MS ({MS})"
        else:
            status = "Zero Cells"
            
        print(f"{os.path.basename(f):<25} | {total_type1:<12} | {cluster_count:<15} | {status}")