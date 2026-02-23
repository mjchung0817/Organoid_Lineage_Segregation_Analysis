# Predictive CN Sidetrack Summary

- Run: `20260223_125649`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- Target PCs: PC1, PC2, PC3

## Per-PC Metrics (Mean Across Folds)

Split_Type        Model  PC        R2     RMSE      MAE
      LOEX     gam_like PC1  0.707305 1.586646 1.284663
      LOEX     gam_like PC2 -0.964264 2.187934 1.680007
      LOEX     gam_like PC3 -1.662371 2.043514 1.600272
      LOEX   poly_ridge PC1  0.123639 2.506857 2.014702
      LOEX   poly_ridge PC2 -4.794134 3.448541 2.569489
      LOEX   poly_ridge PC3 -4.447528 3.100880 2.757637
      LOEX ridge_linear PC1  0.211453 2.421897 2.023085
      LOEX ridge_linear PC2 -3.895983 3.251715 2.848925
      LOEX ridge_linear PC3 -4.647808 3.126045 2.765033
       LOR     gam_like PC1  0.838815 1.168554 0.947037
       LOR     gam_like PC2  0.264781 1.337860 0.996293
       LOR     gam_like PC3 -0.213256 1.444282 1.105632
       LOR   poly_ridge PC1  0.768537 1.417201 1.093324
       LOR   poly_ridge PC2  0.239121 1.391610 0.988678
       LOR   poly_ridge PC3 -0.156225 1.423145 1.110652
       LOR ridge_linear PC1  0.723422 1.553228 1.201930
       LOR ridge_linear PC2 -0.137018 1.704303 1.290840
       LOR ridge_linear PC3 -0.438797 1.549115 1.222383

## Trajectory Metrics (Mean Across Folds)

Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
      LOEX     gam_like             2.613710        1.236885                  49.920279
      LOEX   poly_ridge             4.389690        3.397459                  58.473363
      LOEX ridge_linear             4.454689        4.284203                  61.511830
       LOR     gam_like             1.369041        1.000774                  42.361102
       LOR   poly_ridge             1.458646        1.049324                  48.941045
       LOR ridge_linear             1.955453        1.709910                  61.094987
