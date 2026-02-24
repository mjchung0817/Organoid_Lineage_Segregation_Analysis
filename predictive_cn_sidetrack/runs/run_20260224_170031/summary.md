# Predictive CN Sidetrack Summary

- Run: `20260224_170031`
- Experiments: exp1, exp2_high_cn, exp2_low_cn
- PCA basis (effective): `exp_dox_centroids`
- Replicate adjust: `residualized`
- CN encodings: lambda, sample_mean, summary
- Target PCs: PC1, PC2, PC3

- Custom stress holdout train: exp1, exp2_low_cn
- Custom stress holdout test: exp2_high_cn

## Per-PC Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  PC          R2      RMSE       MAE
     lambda       LOEX   poly_ridge PC1   -0.577256  3.195254  2.908474
     lambda       LOEX   poly_ridge PC2  -53.896368  9.540956  9.099378
     lambda       LOEX   poly_ridge PC3  -17.495017  5.588093  5.353697
     lambda       LOEX ridge_linear PC1   -0.340962  2.994475  2.691106
     lambda       LOEX ridge_linear PC2  -23.014637  6.825842  6.560549
     lambda       LOEX ridge_linear PC3  -12.812197  4.748395  4.467704
     lambda       LOEX       spline PC1   -5.053754  5.186473  4.650103
     lambda       LOEX       spline PC2 -224.452959 17.022206 15.686178
     lambda       LOEX       spline PC3 -107.593452 12.023428 11.233547
     lambda        LOR   poly_ridge PC1    0.785217  1.367016  1.079145
     lambda        LOR   poly_ridge PC2    0.350960  1.294302  0.998538
     lambda        LOR   poly_ridge PC3    0.148834  1.274654  1.013277
     lambda        LOR ridge_linear PC1    0.740291  1.506107  1.180334
     lambda        LOR ridge_linear PC2   -0.063732  1.646756  1.264339
     lambda        LOR ridge_linear PC3   -0.139722  1.422503  1.142903
     lambda        LOR       spline PC1    0.900869  0.895264  0.706923
     lambda        LOR       spline PC2    0.666148  0.925742  0.736375
     lambda        LOR       spline PC3    0.405427  1.065007  0.812537
sample_mean       LOEX   poly_ridge PC1    0.213185  2.464866  2.040742
sample_mean       LOEX   poly_ridge PC2   -4.646793  3.861395  3.302672
sample_mean       LOEX   poly_ridge PC3   -4.888099  3.356567  3.087396
sample_mean       LOEX ridge_linear PC1    0.199273  2.543605  2.107316
sample_mean       LOEX ridge_linear PC2   -3.572174  3.292162  2.969453
sample_mean       LOEX ridge_linear PC3   -5.391245  3.355492  3.028430
sample_mean       LOEX       spline PC1  -57.318987 14.047504 13.837604
sample_mean       LOEX       spline PC2  -28.045632  7.671493  7.074985
sample_mean       LOEX       spline PC3  -37.628311  7.612277  7.006268
sample_mean        LOR   poly_ridge PC1    0.783614  1.371380  1.082847
sample_mean        LOR   poly_ridge PC2    0.332727  1.307838  1.007888
sample_mean        LOR   poly_ridge PC3    0.151553  1.273496  1.012566
sample_mean        LOR ridge_linear PC1    0.739599  1.508355  1.181258
sample_mean        LOR ridge_linear PC2   -0.068146  1.651519  1.266782
sample_mean        LOR ridge_linear PC3   -0.142159  1.423076  1.146046
sample_mean        LOR       spline PC1    0.890666  0.944469  0.737945
sample_mean        LOR       spline PC2    0.522996  1.108359  0.881670
sample_mean        LOR       spline PC3    0.278288  1.154148  0.916381
    summary       LOEX   poly_ridge PC1   -0.089925  2.852522  2.411026
    summary       LOEX   poly_ridge PC2   -4.185473  3.703692  3.210738
    summary       LOEX   poly_ridge PC3   -5.760803  3.499609  3.228050
    summary       LOEX ridge_linear PC1   -0.146009  3.011833  2.586248
    summary       LOEX ridge_linear PC2   -3.198700  3.046384  2.713951
    summary       LOEX ridge_linear PC3   -6.581791  3.526945  3.197327
    summary       LOEX       spline PC1  -65.618072 14.928726 14.713989
    summary       LOEX       spline PC2  -19.535705  6.734787  6.255205
    summary       LOEX       spline PC3  -49.373362  8.494015  7.908999
    summary        LOR   poly_ridge PC1    0.778977  1.386499  1.090382
    summary        LOR   poly_ridge PC2    0.317637  1.318894  1.028458
    summary        LOR   poly_ridge PC3    0.126771  1.288659  1.026664
    summary        LOR ridge_linear PC1    0.732026  1.530254  1.195406
    summary        LOR ridge_linear PC2   -0.078593  1.658447  1.273860
    summary        LOR ridge_linear PC3   -0.174354  1.439401  1.166360
    summary        LOR       spline PC1    0.888207  0.952236  0.749954
    summary        LOR       spline PC2    0.513382  1.115750  0.889632
    summary        LOR       spline PC3    0.273899  1.155508  0.913120

## Trajectory Metrics (Mean Across Folds)

CN_Encoding Split_Type        Model  Centroid_Path_Error  Endpoint_Error  Direction_Angle_Error_Deg
     lambda       LOEX   poly_ridge            11.046164       13.988994                  59.275158
     lambda       LOEX ridge_linear             8.486271        9.114664                  62.691544
     lambda       LOEX       spline            19.911694       26.683756                  60.106774
     lambda        LOR   poly_ridge             1.421548        1.285097                  48.272897
     lambda        LOR ridge_linear             1.878545        1.896484                  60.676998
     lambda        LOR       spline             0.577764        0.397235                  31.404187
sample_mean       LOEX   poly_ridge             5.343107        6.072324                  59.809674
sample_mean       LOEX ridge_linear             5.065749        5.475827                  65.408958
sample_mean       LOEX       spline            18.372793       19.901564                  53.296348
sample_mean        LOR   poly_ridge             1.435468        1.330012                  49.506849
sample_mean        LOR ridge_linear             1.884361        1.899840                  61.490465
sample_mean        LOR       spline             0.822967        0.794122                  36.014703
    summary       LOEX   poly_ridge             5.533825        6.343268                  60.128096
    summary       LOEX ridge_linear             5.311174        5.667549                  67.016604
    summary       LOEX       spline            19.254870       20.564255                  54.491837
    summary        LOR   poly_ridge             1.450480        1.342905                  51.590889
    summary        LOR ridge_linear             1.906236        1.922479                  63.113212
    summary        LOR       spline             0.834055        0.843416                  36.083569
