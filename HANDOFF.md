# HANDOFF: Spatial Analysis + PCA Trajectory Pipeline

## 1) Current State (What is done)
- The core PCA trajectory script is `20260208_spatial_state_trajectory_PCA.py`.
- The pipeline now supports multi-experiment comparison in a shared PCA space, optional replicate residualization, runtime-grouped outputs, flexible trajectory grouping, and flexible PCA fit basis.
- Endo/Meso-only analysis is now the standard in active scripts (triple-negative/pluripotent excluded from core metrics and exports in updated scripts).
- Paper-facing scripts now export plot-ready CSVs (not just PNG figures).
- Adjacency/proximity analysis now includes three inter-lineage touching definitions in the left panel and CSV outputs.

## 2) Architectural Decisions
- Shared dataset registry: scripts use a common `DATASET_MAP` pattern (`exp1`, `exp2_high_cn`, `exp2_low_cn`, `exp3`).
- Standard calibration parameters (kept aligned across scripts):
  - DBSCAN: `eps = 30 um`, `min_samples = 20`
  - NMS neighborhood radius: `100 um` (global NMS), local heatmap radius remains separate where applicable
  - Endo-Meso adjacency threshold: `30 um`
- Balanced sampling default: up to first 3 organoids per `(replicate, dox)` unless disabled.
- Separation of concerns in PCA script:
  - feature extraction
  - preprocessing (normalization, residualization, imputation)
  - PCA fit/projection
  - statistical annotation (KW + signal ratio)
  - figure/CSV export

## 3) Naming Conventions
- Script names: date-prefixed, descriptive snake case (example: `20260208_spatial_state_trajectory_PCA.py`).
- Output folders:
  - `results/<exp_label>_spatial_trajectory/run_YYYYMMDD_HHMMSS/` when runtime grouping is enabled.
- PCA output file prefixes:
  - `<label>_<mode>_pca_scores.csv`
  - `<label>_<mode>_pca_loadings.csv`
  - `<label>_<mode>_pca_explained_variance.csv`
  - `<label>_<mode>_pca_group_separation.csv`
- Modes:
  - `raw`
  - `residualized`
- Grouping controls:
  - `--trajectory-group-by dox|experiment`
  - `--pca-fit-basis auto|all_organoids|exp_dox_centroids|exp_centroids`

## 4) PCA Trajectory Pipeline Evolution (Most Important)

### Stage A: Initial trajectory framing
- Started as multifeature PCA trajectory over dox with core spatial metrics.
- Added cross-experiment mode using shared PCA axes for direct comparison.

### Stage B: Data-retention and preprocessing fixes
- Clarified missingness impact from complete-case filtering.
- Added two-step imputation to reduce sample loss:
  1. median within `(Experiment, Dox_Concentration)`
  2. global feature median fallback
- Added composition normalization:
  - `Total_Cells` max-normalized across included samples
  - `% Endo` and `% Meso` constrained to valid fraction range

### Stage C: Batch/replicate handling
- Added residualized mode for replicate correction while preserving dox-level structure:
  - `x_adj = x - mean(x|experiment,dox,replicate) + mean(x|experiment,dox)`
- Added `--replicate-adjust` options: `none/raw/residualized/both`.

### Stage D: Feature-space expansion and labeling cleanup
- Expanded to 17 PCA features:
  1. `Total_Cells` (max-normalized)
  2. `Pct_Endo`
  3. `Pct_Meso`
  4. `Radial_Mean_Endo`
  5. `Radial_Std_Endo`
  6. `Radial_Mean_Meso`
  7. `Radial_Std_Meso`
  8. `NMS_Endo`
  9. `NMS_Meso`
  10. `Cluster_Count_Endo`
  11. `Cluster_Count_Meso`
  12. `Cluster_Size_Endo`
  13. `Cluster_Size_Meso`
  14. `Intra_Distance_Endo`
  15. `Intra_Distance_Meso`
  16. `Inter_Distance`
  17. `Adjacency_Pct`
- Improved display labels so cluster metrics are explicit (count/size by lineage).

### Stage E: PCA fit controls + visualization clarity
- Added `--trajectory-group-by experiment` mode (cross-experiment trajectory ordering).
- Added `--pca-fit-basis` to decouple fit-set definition from projection:
  - `all_organoids` (full variance, noisier)
  - `exp_dox_centroids` (dox-structure emphasis)
  - `exp_centroids` (experiment-structure emphasis)
- Added dedicated 3D-trajectory-only figure export to reduce clutter.
- Added runtime folder grouping for run traceability.
- Ensured PCA explained variance is recomputed each run (fit-set dependent).

## 5) Statistics/Interpretation Decisions
- Kruskal-Wallis is used per feature, per experiment, across dox groups to test distributional differences.
- Signal ratio is retained as effect-size-style variance partition (`1 - within-dox variance / total variance`).
- Important interpretation decision:
  - High signal ratio indicates within-experiment dox structure.
  - It does not automatically imply strong cross-experiment separation in global PCA scatter.

## 6) Results Discussed from Latest Outputs
(From the figures/tables reviewed during discussion)
- Residualized 3D trajectories look cleaner (less replicate-driven scatter), with visible directional trajectories inside each experiment.
- Grouped-by-experiment PCA still shows partial overlap among experiments in shared space.
- Example scree (residualized grouped-by-experiment run):
  - PC1 about 48.8%
  - PC2 about 23.2%
  - cumulative about 72% by PC2, about 82% by PC3
- Group-separation table interpretation (shared):
  - PC1: low between-experiment ratio (~0.026), high between-dox ratio (~0.825)
  - PC2/PC3: more mixed contributions (both experiment and dox components)
  - This is consistent with strong dox-driven structure but modest pure experiment separation on PC1.
- Practical conclusion already aligned in discussion:
  - lack of strict clustering by experiment does not mean the pipeline is broken;
  - it can reflect true overlap in feature distributions plus dominant dox structure.

## 7) Paper-Support Script Status (CSV exports)
Updated scripts targeted for manuscript figure recreation now emit CSV data for mentor-side visualization cleanup:
- `20260202_cluster_proximity_adjacency_analysis.py`
- `20260208_inter_intra_cluster_distance_analysis.py`
- `20260120_organoid_dbscan_cluster_size_and_inter_distance_analysis.py`
- `2025_0124_delta_analysis_nms_dbscan_cluster.py`
- `20251218_image_confined_z_biopsy_visualization_and_nms_heatmap_v2.py`
- `20251213_calc_nms_replicates_v2.py`

Adjacency script now reports three adjacency formulations:
- minority touching / total minority
- majority touching / total majority
- bipartite adjacency density `sum(A_endo,meso) / (N_endo * N_meso)`

## 8) Current Analytical Direction with Andrew (High Priority)
- Objective: transition from a categorical experiment-comparison framing to a dosage-continuous predictive framing.
- Refined mechanism: assign copy number (CN) per virtual organoid using Poisson sampling with `N = 100` cells and experiment-specific means (`lambda`), then use this sampled CN representation with dox to predict PC coordinates.
- Failure-mode protection (confounding guard): this probabilistic CN assignment is intended to create common support (overlap) between Exp1 and low-CN batches so that separation is not forced by hard experiment labels alone.
- Main predictive goal: train on lower CN range and test whether the model predicts the high-CN (`~9.0`) trajectory/state in PC space, consistent with a continuous genomic-dosage-to-morphology mapping.
- Biological interpretation currently under test:
  - PC2 captures stronger meso-linked variation,
  - PC3 captures stronger endo-linked variation,
  - PC1 is mixed.
- Nonlinear dose-response behavior (larger shifts at lower dox, possible plateau above ~250 ng/mL) should be tested explicitly with nonlinear models.
- Temporal dynamics extension planned:
  - test whether within-dose temporal trajectories align with across-dose trajectory geometry.

## 9) Recommended Next Implementation Steps
1. Freeze one PCA basis for predictive work (no refit during evaluation).
2. Build the probabilistic-input model `dox + sampled_CN_distribution_features -> (PC1, PC2, PC3)` using the virtual-organoid CN logic (`Poisson(lambda), N=100`).
3. Run strict holdouts:
   - leave-one-replicate-out for replicate generalization,
   - lower-CN-train / high-CN-test extrapolation check as the primary continuity claim.
4. Add trajectory error metrics (centroid path error, direction/angle similarity, endpoint error), not only pointwise PC regression metrics.
5. Back-project predicted PCs to feature estimates and compare against observed feature distributions.
6. Add temporal alignment module once time labels are finalized.

## 10) One-Line Operational Reminder
For experiment-driven axes in grouped-by-experiment mode, use:
- `--trajectory-group-by experiment --pca-fit-basis exp_centroids`
For dox-transition-focused axes, use:
- `--pca-fit-basis exp_dox_centroids`
