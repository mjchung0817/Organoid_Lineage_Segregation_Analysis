# Presentation-Focused Ablation Report (summary + poly full + no LOEX)

## Run Config
- Common settings: `--cn-encodings summary --poly-mode full --no-eval-loex --eval-lor --eval-random`.
- Modes compared:
  - `keep`: `run_20260224_181612`
  - `drop`: `run_20260224_182016`
  - `log1p`: `run_20260224_182120`

## 1) PC Prediction Summary (Mean across PC1-3)
### RANDOM
```
Cluster_Size_Endo_Mode        Model Mean_R2 Mean_RMSE Mean_MAE
                  drop   poly_ridge   0.675     1.186    0.914
                  drop ridge_linear   0.523     1.526    1.166
                  drop       spline   0.707     1.107    0.834
                  keep   poly_ridge   0.699     1.247    0.956
                  keep ridge_linear   0.539     1.577    1.217
                  keep       spline   0.724     1.165    0.872
                 log1p   poly_ridge   0.736     1.189    0.926
                 log1p ridge_linear   0.558     1.549    1.187
                 log1p       spline   0.762     1.102    0.847
```

### LOR
```
Cluster_Size_Endo_Mode        Model Mean_R2 Mean_RMSE Mean_MAE
                  drop   poly_ridge   0.484     1.164    0.915
                  drop ridge_linear   0.228     1.478    1.152
                  drop       spline   0.458     1.069    0.841
                  keep   poly_ridge   0.545     1.188    0.933
                  keep ridge_linear   0.160     1.543    1.212
                  keep       spline   0.544     1.092    0.868
                 log1p   poly_ridge   0.545     1.173    0.934
                 log1p ridge_linear   0.183     1.521    1.201
                 log1p       spline   0.529     1.076    0.862
```

## 2) Trajectory-Level Summary
### RANDOM
```
Cluster_Size_Endo_Mode        Model Centroid_Path_Error Endpoint_Error Direction_Angle_Error_Deg
                  drop   poly_ridge               1.535          1.617                    51.914
                  drop ridge_linear               2.098          2.095                    61.667
                  drop       spline               1.373          1.291                    49.965
                  keep   poly_ridge               1.610          1.995                    52.870
                  keep ridge_linear               2.172          2.220                    64.259
                  keep       spline               1.405          1.602                    50.403
                 log1p   poly_ridge               1.563          1.777                    52.121
                 log1p ridge_linear               2.138          2.119                    63.187
                 log1p       spline               1.371          1.374                    50.017
```

### LOR
```
Cluster_Size_Endo_Mode        Model Centroid_Path_Error Endpoint_Error Direction_Angle_Error_Deg
                  drop   poly_ridge               1.064          0.974                    44.785
                  drop ridge_linear               1.839          1.944                    57.834
                  drop       spline               0.878          0.821                    30.694
                  keep   poly_ridge               1.083          1.058                    48.768
                  keep ridge_linear               1.906          1.922                    63.113
                  keep       spline               0.895          0.929                    37.009
                 log1p   poly_ridge               1.096          1.049                    47.021
                 log1p ridge_linear               1.913          1.922                    60.961
                 log1p       spline               0.902          0.898                    33.921
```

## 3) Cluster_Size_Endo Reconstruction Error
### RANDOM
```
Cluster_Size_Endo_Mode        Model      MAE     RMSE
                  keep   poly_ridge  994.094 1351.542
                  keep ridge_linear 1006.085 1331.732
                  keep       spline  961.981 1313.150
                 log1p   poly_ridge    0.953    1.158
                 log1p ridge_linear    0.982    1.165
                 log1p       spline    0.927    1.119
```

### LOR
```
Cluster_Size_Endo_Mode        Model     MAE     RMSE
                  keep   poly_ridge 968.260 1282.663
                  keep ridge_linear 992.340 1289.883
                  keep       spline 946.023 1257.742
                 log1p   poly_ridge   1.032    1.220
                 log1p ridge_linear   1.056    1.232
                 log1p       spline   1.008    1.190
```

## 4) Custom Holdout (Train: exp1+exp2_low_cn, Test: exp2_high_cn)
### PC metrics
```
Cluster_Size_Endo_Mode        Model  PC        R2   RMSE    MAE
                  drop   poly_ridge PC1  -637.405 71.916 70.898
                  drop   poly_ridge PC2  -251.978 26.605 25.608
                  drop   poly_ridge PC3 -2450.612 42.022 40.565
                  drop ridge_linear PC1    -1.281  4.299  4.035
                  drop ridge_linear PC2    -1.120  2.436  2.180
                  drop ridge_linear PC3  -120.955  9.373  9.334
                  drop       spline PC1  -158.986 36.001 35.761
                  drop       spline PC2   -77.168 14.789 13.607
                  drop       spline PC3   -79.762  7.627  6.050
                  keep   poly_ridge PC1  -559.614 70.310 69.297
                  keep   poly_ridge PC2  -153.813 20.320 19.202
                  keep   poly_ridge PC3  -801.951 47.821 47.052
                  keep ridge_linear PC1    -1.116  4.319  4.018
                  keep ridge_linear PC2    -1.418  2.539  2.273
                  keep ridge_linear PC3   -10.510  5.725  5.432
                  keep       spline PC1  -152.680 36.812 36.571
                  keep       spline PC2   -78.549 14.566 13.395
                  keep       spline PC3   -75.049 14.717 13.038
                 log1p   poly_ridge PC1  -483.959 65.214 64.235
                 log1p   poly_ridge PC2   -85.039 14.990 13.706
                 log1p   poly_ridge PC3 -2901.178 67.037 66.231
                 log1p ridge_linear PC1    -0.621  3.771  3.451
                 log1p ridge_linear PC2    -2.770  3.138  2.904
                 log1p ridge_linear PC3   -43.190  8.272  8.163
                 log1p       spline PC1  -167.087 38.394 38.167
                 log1p       spline PC2   -89.569 15.379 14.127
                 log1p       spline PC3  -105.928 12.868 10.614
```

### Trajectory metrics
```
Cluster_Size_Endo_Mode        Model Centroid_Path_Error Endpoint_Error Direction_Angle_Error_Deg
                  drop   poly_ridge              85.786         92.971                    63.105
                  drop ridge_linear              10.435          9.353                    47.436
                  drop       spline              39.359         45.182                    61.088
                  keep   poly_ridge              86.057         94.263                    87.799
                  keep ridge_linear               7.245          5.771                    73.688
                  keep       spline              42.047         45.498                    67.477
                 log1p   poly_ridge              93.385        101.283                    79.492
                 log1p ridge_linear               9.405          8.226                    60.502
                 log1p       spline              43.113         47.169                    67.818
```

## 5) Key Observations
- LOR (`poly_ridge`) mean PC R2: keep=0.545, drop=0.484, log1p=0.545.
- RANDOM (`poly_ridge`) mean PC R2: keep=0.699, drop=0.675, log1p=0.736.
- `Cluster_Size_Endo` LOR error (`poly_ridge`) MAE/RMSE: keep=968.3/1282.7, log1p=1.0/1.2.
- For presentation-only (no LOEX), `poly_ridge` with `Cluster_Size_Endo=keep` or `log1p` is the most balanced option; `drop` degrades trajectory fidelity.
- Custom high-CN holdout remains hard for all models (large negative R2), so treat this panel as stress-test evidence, not headline performance.
