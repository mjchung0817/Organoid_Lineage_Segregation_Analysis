# Predictive CN Sidetrack Summary

- Run: `20260224_000957`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- Target PCs: PC1, PC2, PC3

- Custom stress holdout train: exp1, exp2_high_cn
- Custom stress holdout test: exp2_low_cn

## Per-PC Metrics (Mean Across Folds)

Split_Type        Model  PC         R2      RMSE       MAE
      LOEX   poly_ridge PC1   0.213185  2.464866  2.040742
      LOEX   poly_ridge PC2  -4.646793  3.861395  3.302672
      LOEX   poly_ridge PC3  -4.888099  3.356567  3.087396
      LOEX ridge_linear PC1   0.199273  2.543605  2.107316
      LOEX ridge_linear PC2  -3.572174  3.292162  2.969453
      LOEX ridge_linear PC3  -5.391245  3.355492  3.028430
      LOEX       spline PC1 -57.318987 14.047504 13.837604
      LOEX       spline PC2 -28.045632  7.671493  7.074985
      LOEX       spline PC3 -37.628311  7.612277  7.006268
       LOR   poly_ridge PC1   0.783614  1.371380  1.082847
       LOR   poly_ridge PC2   0.332727  1.307838  1.007888
       LOR   poly_ridge PC3   0.151553  1.273496  1.012566
       LOR ridge_linear PC1   0.739599  1.508355  1.181258
       LOR ridge_linear PC2  -0.068146  1.651519  1.266782
       LOR ridge_linear PC3  -0.142159  1.423076  1.146046
       LOR       spline PC1   0.890666  0.944469  0.737945
       LOR       spline PC2   0.522996  1.108359  0.881670
       LOR       spline PC3   0.278288  1.154148  0.916381

## Trajectory Metrics (Mean Across Folds)

Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
      LOEX   poly_ridge             5.343107        6.072324                  59.809674
      LOEX ridge_linear             5.065749        5.475827                  65.408958
      LOEX       spline            18.372793       19.901564                  53.296348
       LOR   poly_ridge             1.435468        1.330012                  49.506849
       LOR ridge_linear             1.884361        1.899840                  61.490465
       LOR       spline             0.822967        0.794122                  36.014703
