# Supplementary tables

Complete numerical results for *Hub Inputs Funnel Control into the Connectome's Dominant Mode*, one CSV per table (LaTeX markup removed; values unchanged).

Unless a description says otherwise: scales are 68, 114 and 219 regions; $c = 0.1$, $\epsilon = 10^{-5}$, infinite horizon, $k \in \{1,2,3,5\}$, 70 subjects, 100 random driver sets per subject, scale and $k$. Counts such as `64/70` are subjects with a positive difference.

**Tables for the Supporting Information.** The files `S<n>_*.csv` give every setting of the analyses summarized in Section S*n* of the Supporting Information.

**Additional analyses.** The files `additional_*.csv` hold analyses that are not part of the paper. Their methods are described in [`additional_analyses.md`](additional_analyses.md).

**Reproducing.** `python scripts/reproduce.py --run paper` runs every analysis and writes the full tables to `runs/paper/paper/tables/`. `scripts/paper/publish_tables.py` converts them to the CSV files here. [`docs/paper_map.md`](../docs/paper_map.md) maps each table to the code and outputs it is built from.


## Tables for the Supporting Information

### S2. Regularization and dynamical robustness

| Table | Description |
| --- | --- |
| [`S2_eps_energy`](tables/S2_eps_energy.csv) | Average control energy across the eps sweep (c = 0.1, k-averaged). "Frac. eig. numerically 0": fraction of eigenvalues of W that are zero to numerical precision. The last three columns give the cycle-minus-degree difference in log10 E_avg (x1e-3), the number of subjects with a positive difference, and Holm-corrected Wilcoxon p-values across the 7 eps values and 3 scales. |
| [`S2_eps_floor`](tables/S2_eps_floor.csv) | Floored fraction, average energy, and landscape spread at three values of eps (c = 0.1). Fractions are ranges across degree-strength, eigenvector-centrality, cycle-participation, betweenness, and random sets. Landscape: interquartile range of log10 E_avg across the candidate driver sets (all sets when C(N, k) <= 5,000, otherwise 5,000 sampled sets), averaged over subjects and k; "above-eps part" uses only the eigenvalues of W above eps. Holm correction across 3 eps values and 3 scales. |
| [`S2_eps_er`](tables/S2_eps_er.csv) | Cycle-minus-degree differences in effective rank and participation ratio across eps, at c = 0.1 (primary) and c = 0.01. Entries are mean paired differences (k-averaged) with the number of subjects with a positive difference in parentheses. At eps >= 1e-3 both measures are dominated by the floor. |
| [`S2_eps_q1`](tables/S2_eps_q1.csv) | Dependence of a floored analogue of q_1 on eps (c = 0.1). Column 3: q_1 on the unregularized Gramian, averaged over k (the measure used in the main text). Remaining columns: lambda_max/trace at k = 5 computed on the raw Gramian and on W_eps at each eps. The random row uses one random set per subject. |
| [`S2_eigextremes`](tables/S2_eigextremes.csv) | Smallest and largest eigenvalues of W_eps by criterion (c = 0.1, eps = 1e-5, k-averaged). Columns: mean log10 lambda_min; percentage of driver sets with lambda_min = eps; mean log10 lambda_max (SD); fraction of eigenvalues of W numerically zero; fraction below eps; lambda_max/trace(W_eps). |
| [`S2_horizon`](tables/S2_horizon.csv) | Cycle-minus-degree differences in effective rank and participation ratio for finite-horizon and infinite-horizon Gramians. Entries are mean paired differences (k-averaged), with the number of subjects (of 70) with a positive difference. |
| [`S2_csweep_q1`](tables/S2_csweep_q1.csv) | Sweep of the stabilization margin for q_1 (25 log-spaced values of c; k-averaged). Entries are means over 70 subjects with the number of subjects with a positive difference in parentheses. A positive cycle-minus-degree q_1 means cycle-participation sets are more concentrated; a positive degree-minus-random q_1 means degree-strength sets are more concentrated than random sets. The last three columns give the cycle-minus-degree difference in effective rank. |
| [`S2_csweep_er`](tables/S2_csweep_er.csv) | Cycle-minus-degree differences in effective rank and participation ratio for the 13 values of c used in the main analyses (eps = 1e-5, k-averaged). Entries are mean paired differences with the number of subjects with a positive difference in parentheses. |
| [`S2_csweep_er_rand`](tables/S2_csweep_er_rand.csv) | Degree-strength sets minus the mean of random sets, effective rank and participation ratio, for the 13 values of c (eps = 1e-5, k-averaged). |
| [`S2_signflip`](tables/S2_signflip.csv) | Values of c at which the cycle-minus-degree difference changes sign (eps = 1e-5, k-averaged). Mean and median curve: interpolated zero crossing of the mean or median paired difference. n_+ < 35: c at which fewer than half the subjects show a positive difference. Last Holm-significant positive and first Holm-significant negative values on the grid. A dash means no sign change up to c = 2. |
| [`S2_nctpy`](tables/S2_nctpy.csv) | Cycle-minus-degree differences under our model at c = 1 and under the nctpy normalization (eps = 1e-5 in both). |

### S3. Driver-selection comparisons and controls

| Table | Description |
| --- | --- |
| [`S3_crit_q1`](tables/S3_crit_q1.csv) | All criteria compared with random sets of the same size, q_1 (c = 0.1). Delta: criterion minus the mean of the subject's 100 random sets, averaged over subjects. Pct.: mean fraction of random sets with a smaller value. n_>: subjects with Delta > 0. "avg k >= 2" averages k = 2, 3, 5; "avg all" also includes k = 1. |
| [`S3_crit_er`](tables/S3_crit_er.csv) | All criteria compared with random sets, effective rank (c = 0.1, eps = 1e-5). Columns as in S3_crit_q1. Positive Delta means more evenly distributed than random. |
| [`S3_crit_pr`](tables/S3_crit_pr.csv) | All criteria compared with random sets, participation ratio (c = 0.1, eps = 1e-5). Columns as in S3_crit_q1. |
| [`S3_matched_q1`](tables/S3_matched_q1.csv) | Matched and residualized controls, q_1 (c = 0.1). Mean paired difference (first set minus second set, or minus the mean of the matched or random sets), the number of subjects with a positive difference, and Holm-corrected Wilcoxon p. Spatially matched sets are defined only for k >= 2, so their averages use k = 2, 3, 5. Degree-residualized cycle participation: cycle participation after regressing out degree strength. |
| [`S3_matched_er`](tables/S3_matched_er.csv) | Matched and residualized controls, effective rank (c = 0.1, eps = 1e-5). Columns as in S3_matched_q1. |
| [`S3_matched_pr`](tables/S3_matched_pr.csv) | Matched and residualized controls, participation ratio (c = 0.1, eps = 1e-5). Columns as in S3_matched_q1. |
| [`S3_spin`](tables/S3_spin.csv) | Spin tests for the correlations between node rankings and cortical gradients. Spearman correlations across parcels of the subject-averaged connectome, with parametric p-values and two-sided spin-test p-values from 10,000 rotations (Vasa method, primary; Alexander-Bloch method, secondary). G1 and G2 are the first and second gradients of Margulies et al. (2016), resampled from fsLR to fsaverage5 through the registered sphere. Rank divergence is degree rank minus cycle-participation rank. |
| [`S3_rep_agree`](tables/S3_rep_agree.csv) | Agreement of the four representative conventions with degree strength and with Ripser cocycles. Median within-subject Spearman correlations across regions (interquartile range), and the mean number of regions shared with the degree-strength set of the same size. |
| [`S3_rep_er`](tables/S3_rep_er.csv) | Cycle-minus-degree effective rank under each representative convention, by k and averaged over k = 1, 2, 3, 5 (c = 0.1, eps = 1e-5). Mean paired difference (number of subjects with a positive difference). |
| [`S3_rep_dist`](tables/S3_rep_dist.csv) | Does distance from the degree set explain the differences between conventions? Regression of the cycle-minus-degree effective-rank difference on the Jaccard overlap J with the degree set, across 4 conventions and 70 subjects. beta: slope per standard deviation of J. R^2_J: variance explained by J. R^2_var/J and p: additional variance explained by convention identity. Last column: median within-subject correlation. |

### S4. Structural eigenmodes and hub concentration

| Table | Description |
| --- | --- |
| [`S4_gapcorr`](tables/S4_gapcorr.csv) | Across-subject Spearman correlations between the spectral gap and the size of the effect at six values of c, raw (with p) and partial, controlling for mean strength or for mean strength, edge density, and the coefficient of variation of strength. Degree-minus-random q_1: positive means degree-strength sets are more concentrated than random. Cycle-minus-degree q_1: a negative rho means a larger gap goes with a larger separation. |
| [`S4_canonical`](tables/S4_canonical.csv) | Hub concentration in weighted model networks of matched density and in the human connectomes (c = 0.1). Gap: mean spectral gap. Remaining columns: excess q_1 of degree-strength sets over random sets, with the mean percentile among random sets in parentheses, for single drivers (k = 1) and for k = 2 to 5. Model networks (50 per model and scale) received edge weights drawn from the pooled empirical weights. |

### S5. Null networks and the spectral gap

| Table | Description |
| --- | --- |
| [`S5_fidelity`](tables/S5_fidelity.csv) | Fidelity of the null networks (means over subjects and 50 nulls per subject). Strength r: correlation with the empirical strengths. Max rel. strength err.: largest relative strength error. Top-5 hub Jaccard: overlap of the five strongest regions. Degree preserved: fraction of regions with unchanged binary degree. Edge overlap: fraction of empirical edges retained. Mean and weight-weighted connection length (mm). |
| [`S5_nullspectra`](tables/S5_nullspectra.csv) | Leading eigenvalues, spectral gap, and the inverse participation ratio (IPR) of v_1 in empirical and null networks. Real and Null: means. median z: median of each subject's z-score against their own 50 nulls. Counts: subjects above the 95th and below the 5th percentile of their null distribution. |
| [`S5_nullq1`](tables/S5_nullq1.csv) | Degree-set q_1 (mean over k = 2, 3, 5, as in the main text) in empirical and null networks at fixed and gap-matched damping. Mean, median z, number of subjects above the 95th null percentile, and Holm-corrected Wilcoxon p on the z-scores. |
| [`S5_nullperp`](tables/S5_nullperp.csv) | Degree-set q_1 after projecting out the leading mode (k-averaged), empirical and null networks. Columns as in S5_nullq1. |
| [`S5_rayleigh`](tables/S5_rayleigh.csv) | Rayleigh accounting of the empirical-minus-null excess in l_1 and in the spectral gap. Shares are medians across subjects of the per-subject ratio of the excess in R_m to the excess in l_1 (or in the gap). Shares above 1 for the gap arise because l_2 also differs between empirical and null networks. Also shown: within-null correlations of R_1, r_W, and the toolbox assortativity with l_1, and the correlation of r_W with s^->p A_0 s. |
| [`S5_assort`](tables/S5_assort.csv) | Strength assortativity in empirical and null networks: edge-weighted (r_W) and the Brain Connectivity Toolbox version (r_bct). Means, median z, and the number of subjects above the 95th null percentile. |
| [`S5_hubloading`](tables/S5_hubloading.csv) | Hub loading in empirical and null networks. Quantities in the identity v_1(i) = s_i m_i/l_1, where m_i is the strength-weighted mean of v_1 over the neighbors of i. Means across subjects; for nulls, the mean of each subject's 50-null mean, with the number of subjects whose empirical value exceeded their null mean. Strength assortativity in this table is the Brain Connectivity Toolbox version. |

### S6. Core connections and eigenvalue perturbations

| Table | Description |
| --- | --- |
| [`S6_corecut_fo`](tables/S6_corecut_fo.csv) | Observed and first-order predicted changes when weight is removed from each class of edges. Entries are "observed / predicted", medians over subjects (matched controls averaged over 20 draws within subject first). Change in gap is a percentage of each subject's unperturbed gap at cut strengths alpha = 0.25, 0.5, 0.75, and 1. Delta l_1 and Delta l_2 are absolute changes at alpha = 0.5. Front-back: every front-back edge scaled by 1-alpha. Within halves and outside core: strength-matched controls removing the same total weight. Uniform: the same total weight removed evenly from every edge of the class. Anatomical: label halves; median coordinate: core split at the median anterior-posterior centroid coordinate; v_2 sign: halves defined by each subject's v_2 (reference only). rho: Spearman correlation across subjects between predicted and observed changes in the gap at alpha = 0.5. v_2 swap: share of subjects whose perturbed v_2 has absolute cosine below 0.5 with the unperturbed v_2 at alpha = 1. |

## Additional analyses

### Larger driver sets

| Table | Description |
| --- | --- |
| [`additional_largek`](tables/additional_largek.csv) | Cycle-minus-degree differences in effective rank and participation ratio for larger driver sets (c = 0.1, eps = 1e-5). Holm-corrected Wilcoxon p-values. |

### Decay-ratio collapse

| Table | Description |
| --- | --- |
| [`additional_collapse`](tables/additional_collapse.csv) | Decay-ratio collapse. D is the between-scale variance of the three scale-mean curves divided by the total variance across subjects, averaged over the grid (smaller values mean a tighter collapse), on the full range of each axis and on matched windows. |

### Rank divergence and resting-state networks

| Table | Description |
| --- | --- |
| [`additional_rankdiv`](tables/additional_rankdiv.csv) | Mean rank divergence (degree rank minus cycle-participation rank; positive values mean higher priority under cycle participation) within each of the seven resting-state networks, with the number of parcels and the uncorrected spin-test p-value. |

### Cycle-participation representatives and perturbed connectomes

| Table | Description |
| --- | --- |
| [`additional_rep_pr`](tables/additional_rep_pr.csv) | Cycle-minus-degree participation ratio under each representative convention. Columns as in S3_rep_er. |
| [`additional_rep_null`](tables/additional_rep_null.csv) | Effective rank of each convention's top-k set relative to three random baselines: uniform random sets, random sets sharing as many regions with the degree set as the convention does (overlap-matched), and random sets matched to the convention's drivers in strength rank within +/-5% of N (strength-matched). Cells: mean difference from the null mean / mean percentile within the null / number of subjects above the null mean. The degree row compares the degree-strength set with uniform random sets. |
| [`additional_perturb_rho`](tables/additional_perturb_rho.csv) | Rank stability under perturbation: median Spearman correlation between perturbed and unperturbed rankings (interquartile range), with the median correlation between the unperturbed rankings of different subjects as a reference. |
| [`additional_perturb_jac`](tables/additional_perturb_jac.csv) | Stability of the highest-ranked regions under perturbation: mean Jaccard overlap between perturbed and unperturbed top-k sets for k = 2 / 3 / 5. |
| [`additional_perturb_er`](tables/additional_perturb_er.csv) | Cycle-minus-degree effective rank (Ripser cocycles) recomputed on perturbed connectomes, with rankings and Gramians recomputed for each perturbation. Mean difference (number of subjects with a positive difference). |

### Single-region measures, set-level predictors, and the driver Gram matrix

| Table | Description |
| --- | --- |
| [`additional_nodelevel_corr`](tables/additional_nodelevel_corr.csv) | Within-subject Spearman correlations across regions between single-region controllability measures and the ranking criteria (c = 0.1). Median across subjects, with the number of subjects with rho > 0. Modal controllability and v_1(i)^2 do not depend on c; average controllability does. |
| [`additional_withinsubj`](tables/additional_withinsubj.csv) | Within-subject Spearman correlations with q_1 across each subject's 100 random driver sets, by c and k. Median across subjects (number with rho > 0). q_1(G_S): lambda_max(G_S)/trace(G_S). r_bar_G: mean pairwise cosine of the drivers' slow-mode profiles; r_bar_Gperp: the same with the leading mode removed; r_bar_F: mean pairwise cosine of single-driver Gramians. Partial correlations control for the node-level scores L_S, mean modal controllability, and mean average controllability. |
| [`additional_drivergram`](tables/additional_drivergram.csv) | Driver Gram bounds on q_1, pooled over all driver sets and subjects. rho: Spearman correlation between q_1(G_S) and q_1. Upper: median of q_1(G_S) - q_1. Lower: median of q_1 - pi_S. Both gaps were non-negative in every set. |
| [`additional_commgeo`](tables/additional_commgeo.csv) | Driver overlap in slow modes compared with adjacency. Within-subject Spearman correlations with q_1 across the 114 driver sets of each subject (14 criteria and 100 random sets), averaged over k = 2, 3, 5, median across subjects. comm: mean driver Gram cosine (r_bar_G). geo: mean pairwise shortest-path length between drivers, with edge length 1/w or -log w. Partial correlations control for L_S. n better: subjects in which the partial correlation of comm exceeded that of geo_1/w in absolute value; p: paired Wilcoxon test (uncorrected). |
| [`additional_modal`](tables/additional_modal.csv) | Modal-controllability, average-controllability, and lowest-loading sets compared with cycle-participation sets (c = 0.1). Top: mean Jaccard overlap. Bottom: paired differences, cycle participation minus the comparison set, median across subjects (number with a positive difference). |
| [`additional_coverage`](tables/additional_coverage.csv) | Coverage distance of criterion sets relative to random sets (k >= 2). Coverage distance is the mean over non-driver regions of the shortest-path distance (edge length 1/w) to the nearest driver. Delta: criterion minus random. pct: percentile among random sets. n: subjects in which the criterion set was closer than random. |
| [`additional_coverage_rho`](tables/additional_coverage_rho.csv) | Within-subject Spearman correlation between coverage distance and q_1 across driver sets, raw and partial given L_S (k = 2, 3, 5 averaged; median across subjects, with the number of subjects with rho > 0). |

### Secondary null statistics and mediation

| Table | Description |
| --- | --- |
| [`additional_null114`](tables/additional_null114.csv) | Cycle-minus-degree differences in effective rank and q_1 at 114 regions (k-averaged), empirical and null networks at fixed and gap-matched damping. |
| [`additional_mediation`](tables/additional_mediation.csv) | Proportion of the empirical-minus-null difference removed by each candidate mediator (1 - beta_direct/beta_total, ordinary least squares with subject fixed effects; c = 0.1, fixed damping). Strength assortativity is the Brain Connectivity Toolbox version. Binary rich club is identical in empirical and weight-shuffled networks, so it is undefined there. |
| [`additional_mediators`](tables/additional_mediators.csv) | Candidate structural statistics in empirical and null networks: weight-degree correlation, share of weight among the top 20% of regions by degree, strength assortativity (toolbox version), weighted rich club at the 75th, 85th, and 95th degree percentiles, binary and weighted rich club at the 90th percentile, hub strength fraction, and IPR of v_1. Means, median z, and counts above and below the null percentiles. |

### Target states

| Table | Description |
| --- | --- |
| [`additional_targets_full`](tables/additional_targets_full.csv) | Target energies, full targets (c = 0.1, eps = 1e-5). Median percent difference 100x(E_cycle - E_degree)/E_degree (k-averaged), the number of subjects with negative and positive differences, and Holm-corrected Wilcoxon p within scale and target family. |
| [`additional_targets_excl`](tables/additional_targets_excl.csv) | Target energies with the drivers of both methods excluded from each target. Columns as in additional_targets_full. |
| [`additional_overlap`](tables/additional_overlap.csv) | Overlap between drivers and targets: fraction of subject-by-k cases with at least one driver inside the target, cycle-participation / degree-strength sets. |
| [`additional_proximity`](tables/additional_proximity.csv) | Proximity and target energy. Median difference in distance to the nearest driver (cycle minus degree, mm), the number of subjects in which cycle-participation drivers were closer, across-subject Spearman correlations of the energy difference with the distance and coupling differences, the energy difference predicted at equal distance (median regression intercept, bootstrap 95% CI), and the median energy difference in the cases where no driver of either method lay inside the target. |
| [`additional_singleparcel`](tables/additional_singleparcel.csv) | Single-parcel targets: median across parcels of the subject-median energy difference (95% CI), number of parcels significantly favoring each method (Benjamini-Hochberg corrected), correlations across parcels with the distance and coupling differences, and the median within-subject correlation with distance. |
| [`additional_floor_targets`](tables/additional_floor_targets.csv) | Fraction of target energy carried by directions at the regularization floor (c = 0.1), medians over targets, subjects, and k. |
| [`additional_eps_targets`](tables/additional_eps_targets.csv) | Sensitivity of target energies to eps and c, full targets. Median percent difference; an asterisk marks Holm-corrected p < 0.05. |
| [`additional_parcels`](tables/additional_parcels.csv) | Exact parcels in each target mask at each scale (base region, with left and right parcel indices). |

### Reciprocal hub lesions

| Table | Description |
| --- | --- |
| [`additional_lesion_main`](tables/additional_lesion_main.csv) | Mean loss (intact minus lesioned) of effective rank and participation ratio, raw and in percentile points relative to random sets, under the static protocol, pooled over scales, lesion sizes, and k (c = 0.1, eps = 1e-5). Positive values indicate degradation. |
| [`additional_lesion_cross`](tables/additional_lesion_cross.csv) | Difference in loss, D = cycle-participation loss minus degree-strength loss (static protocol, pooled). Positive D means cycle-participation sets degrade more. Bootstrap 95% CI, number of subjects with D > 0, Holm-corrected Wilcoxon p, and rank-biserial correlation. Interaction rows compare D between lesion types. |
| [`additional_lesion_protocol`](tables/additional_lesion_protocol.csv) | Dependence on the evaluation protocol. Mean losses under the static and recompute protocols, the difference D, the number of subjects with D > 0, and Holm-corrected p. |
| [`additional_lesion_abs`](tables/additional_lesion_abs.csv) | Absolute effective rank before and after the lesion for degree-strength and cycle-participation sets, and the post-lesion difference (cycle minus degree) with the number of subjects with a positive difference and uncorrected Wilcoxon p. |
| [`additional_lesion_energy`](tables/additional_lesion_energy.csv) | Change in log10 E_avg after a lesion (lesioned minus intact, static protocol) for each strategy and for random sets, with the fraction of eigenvalues at the floor before and after the lesion, compared with the change expected from network size alone, log10 [(N - h)/N]. |
