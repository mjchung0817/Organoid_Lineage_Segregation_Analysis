# Organoid Spatial Organization Analysis

Current spatial-analysis pipeline for GATA6-HA organoids (paper figures + CSV exports).

Note: raw biological data is not distributed in this repository.

## Scope

This repo currently tracks:
- Per-experiment spatial metrics and plots.
- Cross-experiment NMS/delta comparisons.
- 17-feature PCA trajectory analysis (non-predictive).
- Utility for pruning to max 3 organoids per `(replicate, dox, condition)`.

Predictive CN modeling is isolated under `predictive_cn_sidetrack/` and is not part of the main paper-figure pipeline.

## Active Scripts (Current)

### Structure
- `src/pipeline/` : main analysis and figure-generation scripts
- `src/tools/` : utility and maintenance scripts

### Main figure-generation scripts
- `src/pipeline/20260208_spatial_state_trajectory_PCA.py`
- `src/pipeline/20260202_cluster_proximity_adjacency_analysis.py`
- `src/pipeline/20260208_inter_intra_cluster_distance_analysis.py`
- `src/pipeline/cluster_analysis.py`
- `src/pipeline/mixing_score.py`
- `src/pipeline/delta_analysis.py`
- `src/pipeline/z_biopsy_visualization.py`
- `src/pipeline/spatial_statistics.py`

### Utility
- `src/tools/2026_0321_highest_variance_sample_identifier.py`
  - Prunes extra organoid CSVs by removing samples that most inflate cell-type composition variance.
  - Script is present and tracked in git.

## Input Data Expectations

- File type: `.csv`
- Coordinate columns:
  - `X`, `Y`, `Z` or `Global X`, `Global Y`, `Global Z` depending on script.
- Cell-type column:
  - `cell_type_dapi_adjusted` (preferred) or legacy typo `cell_type_dapi_adusted` (supported fallback).
- Folder organization:
  - Experiment folder -> replicate folders (`*Rep*`) -> organoid csv files.
- Filename parsing assumptions used by scripts:
  - Dox parsed from `(\d+)dox`
  - Condition parsed from `+<condition>_` if present (otherwise `BASAL`/`CTRL`)

## Sampling Policy

- Most `src/` scripts apply first-3 sampling per `(replicate, dox, condition)` internally.
- PCA script controls sampling explicitly via `--organoid-limit`:
  - `3` (default) means max 3 organoids per `(replicate,dox)`
  - `<=0` includes all organoids

## CLI Reference (Argparse)

Common experiment choices:
- `exp1`, `exp2_high_cn`, `exp2_low_cn`, `exp3`

### 1) PCA trajectory (17 features, non-predictive)
`src/pipeline/20260208_spatial_state_trajectory_PCA.py`

Required:
- `--experiment <one or more experiments>`

Optional:
- `--output-dir <path>` (default `results`)
- `--replicate-adjust {none,raw,residualized,both}` (default `both`)
- `--organoid-limit <int>` (default `3`; use `<=0` for all)
- `--trajectory-group-by {dox,experiment}` (default `dox`)
- `--pca-fit-basis {auto,all_organoids,exp_dox_centroids,exp_centroids}` (default `auto`)
- `--group-by-runtime {yes,no}` (default `yes`)

### 2) Inter-lineage cluster adjacency analysis
`src/pipeline/20260202_cluster_proximity_adjacency_analysis.py`

Required:
- `--experiment <exp>`

Optional:
- `--output-dir <path>` (default `results`)

### 3) Inter-/intra-cluster distance analysis
`src/pipeline/20260208_inter_intra_cluster_distance_analysis.py`

Required:
- `--experiment <exp>`

Optional:
- `--output-dir <path>` (default `results`)

### 4) DBSCAN cluster metrics + NMS
`src/pipeline/cluster_analysis.py`

Required:
- `--experiment <exp>`

Optional:
- `--output-dir <path>` (default `results`)
- `--append-only` (only append new 3-label cluster-count exports)

### 5) NMS cross-experiment comparison
`src/pipeline/mixing_score.py`

Required:
- `--baseline <exp>`
- `--treatment <exp>`

Optional:
- `--output-dir <path>` (default `results`)

### 6) Delta analysis (4-panel + NMS-only)
`src/pipeline/delta_analysis.py`

Required:
- `--baseline <exp>`
- `--treatment <exp>`

Optional:
- `--output-dir <path>` (default `results`)
- `--errorbar-mode {sd,se,ci95,none}` (default `sd`)
- `--no-save-nms-only`
- `--trend-experiments <exp...>` (enables cross-experiment metric line-plot mode)
- `--line-metrics {nms,cluster_size,cluster_count,intra_distance}...` (used in trend mode)
- `--organoid-limit <int>` (used in trend mode; default `3`, `<=0` uses all)

### 7) Z-biopsy visualization + local mixing heatmaps
`src/pipeline/z_biopsy_visualization.py`

Required:
- `--experiment <exp>`

Optional:
- `--output-dir <path>` (default `results`)
- `--append-only` (only geometry outputs, skip full regeneration)

### 8) Moran's I + spatial power (sPCA)
`src/pipeline/spatial_statistics.py`

Required:
- `--experiment <exp>`

Optional:
- `--output-dir <path>` (default `results`)

### 9) Sample-pruner utility
`src/tools/2026_0321_highest_variance_sample_identifier.py`

This script currently uses top-of-file config constants (not argparse):
- `MAX_SAMPLES`
- `DRY_RUN`
- `CELL_TYPE_COL`

Default behavior is dry-run; set `DRY_RUN = False` to apply moves.

## Output Structure

Outputs are script-specific subfolders under the chosen `--output-dir` (default `results`).

Examples:
- `results/<exp>_cluster_proximity/`
  - `<exp>_Cluster_Proximity_Adjacency_Analysis.png`
  - `<exp>_Cluster_Proximity_Organoid_Level.csv`
  - `<exp>_Cluster_Proximity_Panel1_Methods.csv`
  - `<exp>_Cluster_Proximity_Panel2_Minority_Frequency.csv`

- `results/<exp>_inter_intra_distance/`
  - `<exp>_InterCluster_Distance_Organoid_Level.csv`
  - `<exp>_IntraCluster_Distance_Organoid_Level.csv`
  - `<exp>_Inter_Intra_Plot_Data.csv`

- `results/<exp>_dbscan_cluster_analysis/`
  - `<exp>_Cluster_Size_Organoid_Level.csv`
  - `<exp>_InterCluster_Separation_Organoid_Level.csv`
  - `<exp>_Cluster_Count_Organoid_Level.csv`
  - `<exp>_NMS_Organoid_Level.csv`
  - `<exp>_4Panel_Spatial_Analysis.png` (and auxiliary plot CSVs)

- `results/<baseline>_vs_<treatment>_nms_comparison/`
  - `NMS_Radius_*um_*.png`
  - `NMS_Radius_*um_Data.csv`
  - `NMS_Radius_*um_Plot_Data_Long.csv`

- `results/<baseline>_vs_<treatment>_delta_analysis/`
  - `<baseline>_vs_<treatment>_<dose>ng_4Panel_Delta_Analysis.png`
  - `<baseline>_vs_<treatment>_<dose>ng_NMS_Delta_Analysis.png` (unless disabled)
  - organoid-level and long-format CSV exports

- `results/<exp_list_joined>_cross_experiment_line_metrics/` (delta trend mode)
  - `<exp_list_joined>_Line_<feature>.png`
  - `<exp_list_joined>_Line_<feature>_summary.csv`
  - `<exp_list_joined>_LineMetrics_pairwise_mean_diffs.csv`
  - `<exp_list_joined>_LineMetrics_organoid_level.csv`

- `results/<exp_list_joined>_spatial_trajectory/run_<timestamp>/` (PCA, default runtime grouping)
  - feature matrix, PCA scores, loadings, variance tables, significance tables (CSV)
  - per-experiment 2D/3D PCA figures
  - loadings-only heatmap and scree-only figure
  - `run_metadata.txt`

## Minimal Run Example (Paper Figures)

```bash
cd "/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis"

python src/pipeline/20260208_spatial_state_trajectory_PCA.py \
  --experiment exp1 exp2_high_cn exp2_low_cn \
  --replicate-adjust residualized

python src/pipeline/delta_analysis.py \
  --trend-experiments exp1 exp2_high_cn exp2_low_cn \
  --line-metrics nms cluster_size cluster_count

python src/pipeline/20260202_cluster_proximity_adjacency_analysis.py --experiment exp1
python src/pipeline/20260208_inter_intra_cluster_distance_analysis.py --experiment exp1
python src/pipeline/cluster_analysis.py --experiment exp1
python src/pipeline/z_biopsy_visualization.py --experiment exp1
```

## Dependencies

Install:

```bash
pip install -r requirements.txt
```

Core libraries:
- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `esda`, `libpysal` (for spatial statistics script)

## Contact

`mchung98@gatech.edu`
