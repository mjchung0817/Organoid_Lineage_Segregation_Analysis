# Predictive CN Sidetrack Summary

- Run: `20260224_181612`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- Cluster_Size_Endo mode: `keep`
- CN encodings: summary
- Target PCs: PC1, PC2, PC3

- Custom stress holdout train: exp1, exp2_low_cn
- Custom stress holdout test: exp2_high_cn

## Per-PC Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  PC        R2     RMSE      MAE
    summary        LOR   poly_ridge PC1  0.788781 1.335653 1.039456
    summary        LOR   poly_ridge PC2  0.619140 0.985704 0.792002
    summary        LOR   poly_ridge PC3  0.227320 1.242156 0.967921
    summary        LOR ridge_linear PC1  0.732026 1.530254 1.195406
    summary        LOR ridge_linear PC2 -0.078593 1.658447 1.273860
    summary        LOR ridge_linear PC3 -0.174354 1.439401 1.166360
    summary        LOR       spline PC1  0.887007 0.959750 0.757058
    summary        LOR       spline PC2  0.485565 1.146557 0.917580
    summary        LOR       spline PC3  0.258698 1.169347 0.928083
    summary     RANDOM   poly_ridge PC1  0.818998 1.360490 1.104679
    summary     RANDOM   poly_ridge PC2  0.795900 1.019732 0.774426
    summary     RANDOM   poly_ridge PC3  0.481903 1.361710 0.988899
    summary     RANDOM ridge_linear PC1  0.748158 1.607460 1.258567
    summary     RANDOM ridge_linear PC2  0.471825 1.649501 1.258780
    summary     RANDOM ridge_linear PC3  0.395599 1.473188 1.134598
    summary     RANDOM       spline PC1  0.889474 1.054909 0.811809
    summary     RANDOM       spline PC2  0.731745 1.175136 0.886303
    summary     RANDOM       spline PC3  0.552131 1.264063 0.918418

## Trajectory Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
    summary        LOR   poly_ridge             1.082670        1.057817                  48.768216
    summary        LOR ridge_linear             1.906236        1.922479                  63.113212
    summary        LOR       spline             0.894776        0.929473                  37.009181
    summary     RANDOM   poly_ridge             1.609997        1.994508                  52.870021
    summary     RANDOM ridge_linear             2.171788        2.220208                  64.259450
    summary     RANDOM       spline             1.405091        1.601516                  50.402687
