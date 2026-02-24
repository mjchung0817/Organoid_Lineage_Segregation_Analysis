# Predictive CN Sidetrack Summary

- Run: `20260224_175359`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- CN encodings: sample_mean
- Target PCs: PC1, PC2, PC3

- Custom stress holdout train: exp1, exp2_low_cn
- Custom stress holdout test: exp2_high_cn

## Per-PC Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  PC         R2      RMSE       MAE
sample_mean       LOEX   poly_ridge PC1   0.213185  2.464866  2.040742
sample_mean       LOEX   poly_ridge PC2  -4.646793  3.861395  3.302672
sample_mean       LOEX   poly_ridge PC3  -4.888099  3.356567  3.087396
sample_mean       LOEX ridge_linear PC1   0.199273  2.543605  2.107316
sample_mean       LOEX ridge_linear PC2  -3.572174  3.292162  2.969453
sample_mean       LOEX ridge_linear PC3  -5.391245  3.355492  3.028430
sample_mean       LOEX       spline PC1 -95.079552 17.682224 17.476518
sample_mean       LOEX       spline PC2 -14.089319  5.938672  5.498000
sample_mean       LOEX       spline PC3 -40.253369  7.926793  7.386389
sample_mean        LOR   poly_ridge PC1   0.783614  1.371380  1.082847
sample_mean        LOR   poly_ridge PC2   0.332727  1.307838  1.007888
sample_mean        LOR   poly_ridge PC3   0.151553  1.273496  1.012566
sample_mean        LOR ridge_linear PC1   0.739599  1.508355  1.181258
sample_mean        LOR ridge_linear PC2  -0.068146  1.651519  1.266782
sample_mean        LOR ridge_linear PC3  -0.142159  1.423076  1.146046
sample_mean        LOR       spline PC1   0.889024  0.953763  0.747580
sample_mean        LOR       spline PC2   0.497185  1.137161  0.905317
sample_mean        LOR       spline PC3   0.262934  1.168571  0.928795
sample_mean     RANDOM   poly_ridge PC1   0.795568  1.448455  1.141215
sample_mean     RANDOM   poly_ridge PC2   0.677875  1.288014  0.954796
sample_mean     RANDOM   poly_ridge PC3   0.493507  1.344524  1.005959
sample_mean     RANDOM ridge_linear PC1   0.752771  1.592740  1.245200
sample_mean     RANDOM ridge_linear PC2   0.480406  1.636117  1.251439
sample_mean     RANDOM ridge_linear PC3   0.401106  1.464852  1.118203
sample_mean     RANDOM       spline PC1   0.886296  1.070042  0.819143
sample_mean     RANDOM       spline PC2   0.733410  1.170958  0.882558
sample_mean     RANDOM       spline PC3   0.546956  1.270400  0.923687

## Trajectory Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
sample_mean       LOEX   poly_ridge             5.343107        6.072324                  59.809674
sample_mean       LOEX ridge_linear             5.065749        5.475827                  65.408958
sample_mean       LOEX       spline            21.340740       21.928237                  57.678706
sample_mean        LOR   poly_ridge             1.435468        1.330012                  49.506849
sample_mean        LOR ridge_linear             1.884361        1.899840                  61.490465
sample_mean        LOR       spline             0.884925        0.885586                  36.747621
sample_mean     RANDOM   poly_ridge             1.776861        1.865181                  54.973112
sample_mean     RANDOM ridge_linear             2.161655        2.216484                  63.811494
sample_mean     RANDOM       spline             1.412668        1.618876                  50.735460
