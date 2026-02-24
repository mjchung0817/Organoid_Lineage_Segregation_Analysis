# Predictive CN Sidetrack Summary

- Run: `20260224_182120`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- Cluster_Size_Endo mode: `log1p`
- CN encodings: summary
- Target PCs: PC1, PC2, PC3

- Custom stress holdout train: exp1, exp2_low_cn
- Custom stress holdout test: exp2_high_cn

## Per-PC Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  PC        R2     RMSE      MAE
    summary        LOR   poly_ridge PC1  0.798864 1.302565 1.010030
    summary        LOR   poly_ridge PC2  0.625965 0.978444 0.789038
    summary        LOR   poly_ridge PC3  0.210543 1.236846 1.002122
    summary        LOR ridge_linear PC1  0.742623 1.500852 1.168243
    summary        LOR ridge_linear PC2 -0.064079 1.645245 1.270526
    summary        LOR ridge_linear PC3 -0.130776 1.418264 1.163228
    summary        LOR       spline PC1  0.889880 0.949982 0.748619
    summary        LOR       spline PC2  0.498802 1.134100 0.906460
    summary        LOR       spline PC3  0.198678 1.143094 0.931665
    summary     RANDOM   poly_ridge PC1  0.828665 1.318215 1.060389
    summary     RANDOM   poly_ridge PC2  0.794596 1.001708 0.773044
    summary     RANDOM   poly_ridge PC3  0.585899 1.245945 0.944436
    summary     RANDOM ridge_linear PC1  0.756604 1.574367 1.224119
    summary     RANDOM ridge_linear PC2  0.464355 1.627963 1.236972
    summary     RANDOM ridge_linear PC3  0.451674 1.443441 1.101080
    summary     RANDOM       spline PC1  0.896301 1.019778 0.788544
    summary     RANDOM       spline PC2  0.726659 1.161802 0.881039
    summary     RANDOM       spline PC3  0.661957 1.124871 0.870704

## Trajectory Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
    summary        LOR   poly_ridge             1.096041        1.048650                  47.021389
    summary        LOR ridge_linear             1.913026        1.922214                  60.961292
    summary        LOR       spline             0.901874        0.897725                  33.920543
    summary     RANDOM   poly_ridge             1.562591        1.776516                  52.121040
    summary     RANDOM ridge_linear             2.138368        2.118763                  63.186818
    summary     RANDOM       spline             1.371227        1.373593                  50.017090
