# Organoid Spatial Organization Analysis

Spatial analysis pipeline for quantifying cell mixing, domain formation, and lineage segregation in GATA6-HA organoids across doxycycline dose responses.

> **Note:** This repository contains the analysis pipeline only. Due to lab privacy policies, raw experimental data is excluded.

## Overview

This project analyzes 3D organoid spatial organization to quantify how induction strength (doxycycline concentration) affects endoderm vs mesoderm lineage segregation. The pipeline computes spatial mixing metrics, entropy-based gradient analysis, DBSCAN clustering, and spatial autocorrelation statistics across multiple experimental conditions and replicates.

## Key Features

### 1. Normalized Mixing Score (NMS)
| Feature | Description |
|---------|-------------|
| **Radius-based neighbor analysis** | KDTree spatial queries to identify cell neighbors within defined radius |
| **Per-cell mixing ratio** | Fraction of "foreign" lineage neighbors for each cell |
| **Global NMS** | Average mixing across all cells per organoid/condition |
| **Dual-lineage NMS** | Separate mixing scores for endoderm and mesoderm perspectives |
| **Replicate aggregation** | Mean + SEM across biological replicates |

### 2. Spatial Entropy Gradient
| Feature | Description |
|---------|-------------|
| **Multi-scale analysis** | Shannon entropy computed at radii from 10-300 um |
| **Dose-response curves** | Entropy vs neighborhood radius per doxycycline concentration |
| **Domain organization detection** | Identifies loss of spatial domains at higher induction |

### 3. Virtual Z-Biopsy & Heatmap Visualization
| Feature | Description |
|---------|-------------|
| **Z-slice selection** | Filters cells within defined Z-thickness for 2D projections |
| **Optimal slice detection** | ConvexHull area maximization to find best cross-section |
| **Per-cell mixing heatmap** | Color-coded local mixing ratios overlaid on spatial anatomy |
| **3-panel visualization** | Anatomy + mesoderm mixing + endoderm mixing |

### 4. DBSCAN Cluster Analysis
| Feature | Description |
|---------|-------------|
| **3D spatial clustering** | DBSCAN on cell coordinates to identify lineage domains |
| **Cluster size quantification** | Average domain size across conditions |
| **Inter-cluster distance** | Edge-to-edge distances between domains |
| **Parameter optimization** | Grid search over eps and min_samples |

### 5. Delta Analysis (Condition Comparison)
| Feature | Description |
|---------|-------------|
| **Cross-condition comparison** | Compares 100 dox vs 1000 dox with additives (BMP4, Wnt5a) |
| **Relative percent change** | Delta metrics between experimental conditions |
| **Dual-lineage delta** | Separate delta for endoderm and mesoderm segregation |

### 6. Spatial Autocorrelation (Moran's I & sPCA)
| Feature | Description |
|---------|-------------|
| **Global Moran's I** | Quantifies overall spatial clustering of marker intensities |
| **LISA** | Local Indicators of Spatial Association for hotspot detection |
| **Spatial PCA** | Identifies which marker channels drive spatial organization |

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies
- `pandas`, `numpy`, `scipy` - data manipulation and statistics
- `scikit-learn` - DBSCAN clustering, PCA
- `esda`, `libpysal` - spatial autocorrelation (Moran's I, LISA)
- `matplotlib`, `seaborn` - plotting

## Scripts

| Script | Purpose |
|--------|---------|
| `20251213_calc_nms_replicates_v2.py` | NMS calculation with dual-lineage metrics and abundance tracking |
| `20251213_spatial_gradient.py` | Multi-scale Shannon entropy gradient analysis |
| `20251218_image_confined_z_biopsy_visualization_and_nms_heatmap_v2.py` | Virtual Z-biopsy with 3-panel mixing heatmaps |
| `2025_0124_delta_analysis_nms_dbscan_cluster.py` | Cross-condition delta analysis (NMS + DBSCAN) |
| `20260120_organoid_dbscan_cluster_size_and_inter_distance_analysis.py` | Cluster size and inter-cluster distance trends |
| `20260124_spatial_cluster_analysis_moransI_sPCA.py` | Moran's I spatial autocorrelation and spatial PCA |
| `20260125_cell_count_sanity_check.py` | Data validation for cell type populations |
| `20251224_dbscan_param_optimization.py` | DBSCAN parameter grid search optimization |

### Archive (Earlier Iterations)
| Script | Purpose |
|--------|---------|
| `20251213_calc_nms_replicates_v1.py` | NMS v1 (superseded by v2) |
| `20251213_image_confined_z_biopsy_visualization_and_nms_heatmap_v1.py` | Z-biopsy v1 (superseded by v2) |
| `20251218_epsilon_optimization_for_3D_clustering_and_center_detection.py` | DBSCAN optimization (commented out, superseded) |
| `2025_1214_depth_cued_3D_visualization_of_colonies_and_centers.py` | 3D depth visualization (commented out, superseded) |

## Input Data Format

- **File type:** `.csv`
- **Required columns:**
  - `X`, `Y`, `Z` (or `Global X`, `Global Y`, `Global Z`) - 3D spatial coordinates
  - `cell_type_dapi_adusted` - cell type classification (1.0, 2.0=Endoderm, 3.0=Mesoderm)
- **Organization:** Files organized by experiment, replicate, and doxycycline concentration
  - Naming convention: `{dox_concentration}dox_{sample_id}.csv`

## Output Structure

```
{experiment}_output/
├── summary_spatial_analysis.csv        # Global NMS per condition/replicate
├── summary_spatial_dual_nms.csv        # Dual-lineage NMS + abundance
├── Organoid_Cluster_Sizes.csv          # DBSCAN cluster metrics
├── Organoid_InterCluster_Distances.csv # Inter-domain distances
└── *.png                               # Visualization plots
```

## Experimental Design

- **System:** GATA6-HA inducible organoids
- **Variable:** Doxycycline concentration (0, 100, 250, 500, 1000 ng/mL)
- **Lineages:** Endoderm (type 2.0) vs Mesoderm (type 3.0)
- **Experiments:**
  - Ex1: Standard dose-response (3 replicates)
  - Ex2: High vs low copy number comparison (3 replicates each)
  - Ex3: BMP4 + Wnt5a additive conditions (3 replicates)

## License

[Add your license here]

## Contact

[Add your contact information or lab website]
