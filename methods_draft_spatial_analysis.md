# Methods: Spatial Analysis of Lineage Segregation in Organoids

## Spatial data preprocessing and cell type assignment

Individual cells within each organoid were resolved in three-dimensional space (X, Y, Z coordinates in micrometers) from [confocal / lightsheet — specify imaging modality]. Cell lineage identity was assigned based on DAPI-adjusted fluorescence thresholds [describe thresholding or GMM classification criteria used to assign cell_type_dapi_adjusted values], yielding three populations: GATA6-HA–negative cells (triple negative, label 0.0), endoderm-committed cells (Endo, label 2.0), and mesoderm-committed cells (Meso, label 3.0). Up to three organoids per biological replicate per doxycycline concentration were included in all downstream analyses to maintain balanced sampling across conditions.


## Normalized mixing score

To quantify the degree of spatial intermixing between lineages, we computed a normalized mixing score (NMS) for each cell population using a k-d tree–based neighborhood query. For a given target lineage (e.g., Endo), all cells of that lineage and all non-target cells were indexed into separate k-d trees (scipy.spatial.cKDTree). For each target cell, the number of non-target neighbors (n_foreign) and same-lineage neighbors (n_self) within a fixed radial distance r were counted using a ball-point query. The NMS was then defined as:

NMS = (n_foreign / n_self) / (N_foreign / N_self)

where N_foreign and N_self denote the total population sizes of the non-target and target lineages, respectively. This normalization controls for differences in population abundance: an NMS of 1.0 indicates spatially random mixing, values below 1.0 indicate segregation (cells preferentially neighbor their own lineage), and values above 1.0 indicate heterotypic association.

NMS was computed at three neighborhood radii (r = 30, 50, and 100 μm) to assess spatial mixing at multiple length scales. [Specify which radius was used for the primary analyses reported in figures, and rationale for that choice.] Self-counts were corrected by subtracting one from same-lineage neighbor tallies to exclude the query cell itself. Populations with fewer than 10 cells of either the target or non-target class were excluded from NMS calculation for that organoid.


## Cross-sectional visualization and local mixing heatmaps

To provide qualitative spatial context for the quantitative mixing analyses, two-dimensional cross-sectional views of representative organoids were generated. [NOTE: The current v2 script uses median(Z) ± 20 μm as the cross-section. ConvexHull is imported but unused. If a convex-hull-based optimization was intended (scanning Z slices, computing 2D convex hull area at each slice, selecting the Z with maximal cross-sectional area), that logic is not present in the current codebase. Confirm which approach was used and update accordingly.] For each organoid, [the Z-plane yielding the largest cross-sectional area was identified by scanning candidate Z-slices and computing the 2D convex hull area of all cells within ±20 μm of each candidate / OR / the median Z-coordinate across all cells was computed, and a thin optical section was extracted by retaining cells within ±20 μm of this median plane]. X and Y coordinates were then mean-centered to place the organoid at the origin of the visualization canvas.

Within each cross-sectional slice, a local mixing score was computed for every cell. Using a k-d tree constructed on the 2D coordinates of the slice, all neighbors within a radius of 50 μm were identified for each cell. The local mixing score was defined as the fraction of neighbors belonging to a different lineage:

local_mixing = n_foreign_neighbors / n_total_neighbors

Three-panel visualizations were produced for each organoid: (i) an anatomy panel showing cell positions colored by lineage identity (Endo, Meso, or triple negative), (ii) a mesoderm local mixing heatmap, and (iii) an endoderm local mixing heatmap, with mixing scores mapped to a yellow-to-red color gradient (0 = fully segregated, 1.0 = fully mixed).


## Density-based clustering and parameter optimization

Spatial clusters of each lineage were identified independently using DBSCAN (density-based spatial clustering of applications with noise), which groups cells into clusters based on local point density without requiring a predetermined number of clusters. Two parameters govern DBSCAN behavior: epsilon (ε), the maximum distance between two cells for them to be considered neighbors, and min_samples, the minimum number of cells required to form a dense region (cluster core).

Parameter selection proceeded in two stages. First, a two-dimensional grid search was performed over ε ∈ {10, 20, 30, 40, 50, 60, 70, 80} μm and min_samples ∈ {5, 10, 15, …, 100}, with the resulting cluster counts visualized as heatmaps for each doxycycline concentration to identify stable parameter regimes. Second, min_samples was fixed at 20 and a finer epsilon sweep (ε = 10–100 μm, step size 5 μm) was performed across all organoids in three biological replicates. Cluster count was plotted as a function of epsilon for each replicate (mean ± s.d. across organoids), and the value ε = 30 μm was selected as the point at which cluster count stabilized across replicates and doxycycline concentrations [confirm: selected at the "knee" of the curve where cluster count plateaus / OR selected based on criteria from a Gaussian mixture model–based optimization pipeline as described in [Andrew's reference / citation]]. Cells not assigned to any cluster (DBSCAN noise label = −1) were excluded from all cluster-level metrics.


## Cluster morphometrics

For each organoid and each lineage, the following cluster-level metrics were computed from the DBSCAN output:

**Cluster count.** The number of spatially distinct clusters identified for a given lineage (excluding noise).

**Cluster size.** The mean number of cells per cluster, averaged across all clusters of a given lineage within an organoid.

**Intra-lineage inter-cluster separation.** For each pair of clusters belonging to the same lineage, the edge-to-edge distance was computed as the minimum Euclidean distance between any cell in cluster *i* and any cell in cluster *j*, determined using k-d tree nearest-neighbor queries (scipy.spatial.cKDTree). Pairwise minimum distances were averaged across all unique cluster pairs to yield a single per-organoid separation value for each lineage. This metric required at least two clusters of the same lineage to be present.

**Normalized mixing score (NMS).** Computed as described above (see "Normalized mixing score") using a neighborhood radius of 100 μm.

These four metrics — cluster count, cluster size, inter-cluster separation, and NMS — were reported as functions of doxycycline concentration for each lineage (triple negative, Endo, Meso).


## Global spatial autocorrelation (Moran's I)

To assess the degree of global spatial clustering of gene expression intensity, Moran's I statistic was computed for each fluorescence channel independently. For each organoid, a binary spatial weights matrix was constructed using a distance-band criterion (libpysal.weights.DistanceBand) with a threshold of 30 μm: cells within 30 μm of one another were assigned a weight of 1, and all others a weight of 0. Moran's I was then computed on the log1p-normalized fluorescence values for each channel using the esda.moran.Moran implementation. Moran's I ranges from −1 (perfect dispersion) through 0 (spatial randomness) to +1 (perfect positive autocorrelation, i.e., clustered expression). The significance of each Moran's I value was assessed against a null distribution generated by [permutation — confirm the default in esda, typically 999 permutations].

Local Indicators of Spatial Association (LISA; Moran_Local) were additionally computed to identify individual cells contributing disproportionately to the global autocorrelation pattern. The average LISA value across all cells assigned to DBSCAN clusters (excluding noise) was reported as a summary measure of within-cluster spatial coherence.


## Inter- and intra-cell-type cluster distance analysis

To characterize the spatial relationship between endoderm and mesoderm clusters, two complementary distance metrics were computed:

**Inter-cell-type cluster distance.** For every pair consisting of one endoderm cluster and one mesoderm cluster, the edge-to-edge distance was determined as the minimum Euclidean distance between any cell in the endoderm cluster and any cell in the mesoderm cluster, computed via k-d tree nearest-neighbor queries. All pairwise minimum distances were averaged to yield a single per-organoid inter-cell-type distance, reflecting the typical spatial gap between lineage domains.

**Intra-cell-type cluster distance.** For each lineage independently, pairwise edge-to-edge distances between all clusters of the same lineage were computed using the same k-d tree approach. The mean pairwise minimum distance was reported as the intra-cell-type cluster separation, capturing the spatial dispersion of clusters within a single lineage. This metric was computed separately for Endo and Meso and required at least two clusters of the same lineage.

Both metrics were reported as functions of doxycycline concentration with standard error of the mean.


## Inter-lineage cluster adjacency analysis

To quantify the spatial co-localization of endoderm and mesoderm cluster domains — a structural prerequisite for recapitulating organized tissue architectures such as the sinusoidal–parenchymal interface — we developed an inter-lineage cluster adjacency metric. This metric assesses what fraction of the minority lineage's clusters are positioned in direct proximity to clusters of the majority lineage.

For each organoid, DBSCAN clusters were identified separately for Endo and Meso populations using the parameters described above (ε = 30 μm, min_samples = 20). The minority lineage was defined as the lineage with fewer detected clusters, reflecting the cell type with fewer independent spatial domains — the limiting factor for tissue-level organization. For each minority-lineage cluster, a k-d tree was constructed over its constituent cells, and the nearest-neighbor distance to every cell in every majority-lineage cluster was queried. A minority cluster was classified as "adjacent" to the majority lineage if the minimum edge-to-edge distance to any majority cluster was ≤ 30 μm [this threshold matches ε — discuss whether this is intentional or coincidental, and whether a sensitivity analysis across thresholds was performed]. The adjacency percentage was then calculated as:

adjacency % = (number of adjacent minority clusters / total minority clusters) × 100

This formulation uses the minority lineage as the denominator because it represents the structurally limiting population: a high adjacency percentage indicates that most of the scarce lineage's clusters are properly positioned next to the abundant lineage, consistent with organized tissue patterning.

An accompanying summary metric tracked which lineage was identified as the minority (by cluster count) at each doxycycline concentration, providing context for interpreting shifts in cluster adjacency across conditions.


## Comparative delta analysis

To directly compare the spatial organization of GATA6-HA organoids cultured under standard conditions versus BMP4 + Wnt5a supplementation [confirm treatment identity for exp3], a delta analysis was performed between the baseline experiment (Experiment 1: GATA6-HA, three biological replicates) and the treatment experiment (Experiment 3: GATA6-HA + BMP4/Wnt5a). For each organoid in both conditions, four spatial metrics were computed per lineage (Endo and Meso): NMS (r = 100 μm), mean cluster size, cluster count, and inter-cluster separation (centroid-based pairwise distances [note: the delta script uses centroid-based pdist, not edge-to-edge — confirm this is intentional and consistent with how you want to describe it]).

For each doxycycline concentration (100 and 1000 ng/mL), the mean value of each metric across all baseline organoids served as the reference. Treatment organoid values were then expressed as:

- **NMS**: absolute difference from baseline mean (Δ = treatment − baseline)
- **Cluster size, count, and separation**: percent change from baseline mean (% change = ((treatment − baseline) / baseline) × 100)

NMS was reported as an absolute difference rather than a percent change because its baseline-normalized formulation (centered on 1.0 for random mixing) renders absolute deviations directly interpretable. [If there is additional rationale for this choice, add here.]


## Software and reproducibility

All spatial analyses were implemented in Python [version] using pandas [version] for data handling, scikit-learn [version] for DBSCAN clustering, scipy [version] for k-d tree construction and distance computation, libpysal [version] and esda [version] for spatial weights and Moran's I statistics, and matplotlib [version] / seaborn [version] for visualization. [Add any additional packages or specific version numbers.] All analysis code is available at [repository URL].
