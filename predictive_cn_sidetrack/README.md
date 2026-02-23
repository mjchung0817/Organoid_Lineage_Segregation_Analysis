# Predictive CN Sidetrack

This sidetrack implements the predictive workflow discussed for continuous dox/CN modeling in PCA space.

## What it does
- Builds organoid-level spatial features from raw CSVs using the same feature engine as `20260208_spatial_state_trajectory_PCA.py`.
- Freezes one PCA target space (scores + loadings + scaler artifacts).
- Builds probabilistic CN inputs using `Poisson(lambda)` with configurable virtual cell count (`N=100` default).
- Trains/evaluates:
  - `ridge_linear` (`[dox, cn]` baseline)
  - `poly_ridge`
    - `--poly-mode selected` (default): `dox, dox^2, dox^3, cn, dox*cn, dox^2*cn`
    - `--poly-mode full`: full sklearn polynomial expansion up to `--poly-degree`
  - `gam_like` (pyGAM if available, otherwise spline+ridge fallback)
- Runs strict holdouts:
  - LOR: leave-one-(experiment,replicate)-out
  - LOEX: leave-one-experiment-out
- Exports per-PC metrics, trajectory metrics, and feature back-projection errors.

## Script
- `/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis/predictive_cn_sidetrack/scripts/run_predictive_cn_pipeline.py`

## Dependency note
Use the same Python environment you use for the main spatial pipeline. At minimum, install:

```bash
pip install -r "/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis/requirements.txt"
```

## Recommended run
```bash
python "/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis/predictive_cn_sidetrack/scripts/run_predictive_cn_pipeline.py" \
  --experiments exp1 exp2_high_cn exp2_low_cn \
  --replicate-adjust residualized \
  --trajectory-group-by experiment \
  --pca-fit-basis exp_dox_centroids \
  --organoid-limit 3 \
  --target-pcs 3 \
  --cn-map "exp1:4.5,exp2_high_cn:9,exp2_low_cn:4" \
  --cn-cells 100 \
  --poly-mode selected \
  --poly-degree 3 \
  --eval-lor --eval-loex
```

## Output structure
Each run writes to:
- `/Users/minjaechung/Desktop/GaTech/KempLab/Andrew's Paper Spatial Analysis/predictive_cn_sidetrack/runs/run_YYYYMMDD_HHMMSS/`

Key files:
- `artifacts/frozen_pca_scores.csv`
- `artifacts/frozen_pca_loadings.csv`
- `artifacts/frozen_scaler_params.csv`
- `metrics/per_pc_metrics_by_fold.csv`
- `metrics/per_pc_metrics_summary.csv`
- `metrics/trajectory_metrics_by_fold.csv`
- `metrics/trajectory_metrics_summary.csv`
- `metrics/high_cn_extrapolation_metrics.csv` (when applicable)
- `predictions/fold_pc_predictions.csv`
- `predictions/fold_feature_reconstruction_predictions.csv`
- `summary.md`
- `run_manifest.json`
