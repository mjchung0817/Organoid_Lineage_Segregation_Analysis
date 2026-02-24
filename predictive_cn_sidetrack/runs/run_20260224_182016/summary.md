# Predictive CN Sidetrack Summary

- Run: `20260224_182016`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- Cluster_Size_Endo mode: `drop`
- CN encodings: summary
- Target PCs: PC1, PC2, PC3

- Custom stress holdout train: exp1, exp2_low_cn
- Custom stress holdout test: exp2_high_cn

## Per-PC Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  PC        R2     RMSE      MAE
    summary        LOR   poly_ridge PC1  0.795200 1.301762 1.009773
    summary        LOR   poly_ridge PC2  0.641744 0.999174 0.798897
    summary        LOR   poly_ridge PC3  0.015842 1.190095 0.936047
    summary        LOR ridge_linear PC1  0.737103 1.504292 1.170331
    summary        LOR ridge_linear PC2 -0.075003 1.728114 1.330303
    summary        LOR ridge_linear PC3  0.021599 1.201019 0.956021
    summary        LOR       spline PC1  0.896054 0.911492 0.718495
    summary        LOR       spline PC2  0.518484 1.162990 0.918613
    summary        LOR       spline PC3 -0.039642 1.132863 0.885950
    summary     RANDOM   poly_ridge PC1  0.823733 1.318223 1.059812
    summary     RANDOM   poly_ridge PC2  0.814990 1.030897 0.793032
    summary     RANDOM   poly_ridge PC3  0.387525 1.210036 0.888086
    summary     RANDOM ridge_linear PC1  0.746012 1.585130 1.236474
    summary     RANDOM ridge_linear PC2  0.482981 1.730055 1.331391
    summary     RANDOM ridge_linear PC3  0.340041 1.262435 0.930166
    summary     RANDOM       spline PC1  0.898634 0.993914 0.768088
    summary     RANDOM       spline PC2  0.750063 1.199897 0.894914
    summary     RANDOM       spline PC3  0.471025 1.126809 0.838706

## Trajectory Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
    summary        LOR   poly_ridge             1.063599        0.974304                  44.785321
    summary        LOR ridge_linear             1.839449        1.944420                  57.834352
    summary        LOR       spline             0.877664        0.821428                  30.693666
    summary     RANDOM   poly_ridge             1.535343        1.616918                  51.913971
    summary     RANDOM ridge_linear             2.097623        2.095461                  61.666612
    summary     RANDOM       spline             1.373089        1.291342                  49.964625
