# Additional analyses

Methods for the analyses whose results are in the `tables/additional_*.csv` files listed in the
[README](README.md). Notation follows the paper. Unless stated otherwise, the settings are 68, 114 and 219
regions, $c = 0.1$, $\epsilon = 10^{-5}$, $k \in \{1,2,3,5\}$ and 70 subjects.

## Larger driver sets

The cycle-minus-degree comparison of effective rank and participation ratio, repeated for driver sets of up to
20% of regions: $k \in \{1,2,3,5,7,10,14\}$ at 68 regions, $\{1,2,3,5,6,10,11,23\}$ at 114 regions, and
$\{1,2,3,5,10,11,22,44\}$ at 219 regions. $q_1$ is not computed for these sizes.
Table: `additional_largek`.

## Decay-ratio collapse

The effect curves of the $c$ sweep (cycle-minus-degree $q_1$ and effective rank, degree-minus-random $q_1$) are
re-plotted against the decay ratio $d = c/(c + g)$, where $g$ is the subject's spectral gap. The between-scale
dispersion of the curves is compared on matched windows of $c$ and of $d$. Per-subject sign changes in effective
rank are located on both axes and compared across scales with a Kruskal–Wallis test.
Table: `additional_collapse`. Figure: `concentration/fig_decay_collapse.png` in the run directory.

## Rank divergence, cortical gradients and resting-state networks

Rank divergence is degree rank minus cycle-participation rank, computed on the subject-averaged connectome. It is
correlated across parcels with the principal and second cortical gradients of Margulies et al. (2016), with
spin-test p-values from 10,000 rotations, and averaged within the seven resting-state networks of Yeo et al.
(2011), with spin tests for enrichment.
Tables: `S3_spin` (correlations), `additional_rankdiv` (network enrichment).

## Cycle-participation representatives and perturbed connectomes

The four representative conventions are described in Section S3 of the Supporting Information. Each convention's
top-$k$ set is compared with three random baselines: uniform random sets, and random sets whose members are
matched in strength rank to the convention's drivers.

Rankings and Gramians are recomputed on perturbed connectomes: multiplicative Gaussian noise on existing edge
weights ($\sigma = 0.05$, 0.10, 0.20) and random removal of existing edges ($p = 0.05$, 0.10, 0.20), with 50
perturbations per level, subject and scale. When a perturbed weight exceeds 1, the matrix is renormalized by its
maximum. Rank stability is the Spearman correlation between perturbed and unperturbed rankings, compared with the
correlation between different subjects' rankings; top-set stability is the Jaccard index of the top-$k$ sets.
Tables: `additional_rep_pr`, `additional_rep_null`, `additional_perturb_rho`, `additional_perturb_jac`,
`additional_perturb_er`.

## Single-region measures, set-level predictors and the driver Gram matrix

**Average and modal controllability.** For a driver set $S$, $\mathrm{trace}(W) = \sum_{i \in S} \mathrm{AC}_i$ with
$\mathrm{AC}_i = [(-2A)^{-1}]_{ii} = \frac{1}{2C}\sum_{n \geq 0} (A_0^n)_{ii}/C^n$ and $C = \ell_1 + c$. When $C$ is
large, the leading non-constant term is $\sum_j (A_0)_{ij}^2 / 2C^3$. When $c$ is small, $\mathrm{AC}_i$ is
dominated by $v_1(i)^2/2c$. The modal controllability of Gu et al. (2015) uses $A_d = A_0/(1 + \sigma_{\max})$ and
satisfies $\phi_i = \sum_a (1 - \xi_a^2) v_a(i)^2 = 1 - \sum_j (A_0)_{ij}^2/(1 + \sigma_{\max})^2$. It therefore
decreases with the squared connection weights of region $i$.

**Driver Gram matrix.** Let $G_S = B^\top(-2A)^{-1}B$, a $k \times k$ matrix. Its diagonal holds the drivers'
average controllabilities, and $\mathrm{trace}(G_S) = \mathrm{trace}(W)$. In the eigenbasis of $A$, with
$|\mu_a| = \ell_1 - \ell_a + c$ and $\tilde B = V^\top B$, the Gramian is
$\tilde W_{ab} = (\tilde B\tilde B^\top)_{ab}/(|\mu_a| + |\mu_b|)$. This is the Hadamard product $H \circ K$ of two
matrices:

- $H_{ab} = (\tilde B\tilde B^\top)_{ab}/(2\sqrt{|\mu_a||\mu_b|})$, which equals $D^{1/2}\tilde B\tilde B^\top D^{1/2}$
  with $D = \mathrm{diag}(1/2|\mu_a|)$. Its nonzero eigenvalues are therefore those of $\tilde B^\top D\tilde B = G_S$.
- $K_{ab} = 2\sqrt{|\mu_a||\mu_b|}/(|\mu_a| + |\mu_b|)$, which is positive semidefinite with unit diagonal, because
  $K_{ab} = \int_0^\infty f_a f_b\,dt$ with $f_a(t) = \sqrt{2|\mu_a|}\,e^{-|\mu_a|t}$.

By the Schur product theorem, $(\lambda_{\max}(H)I - H) \circ K \succeq 0$, so $W \preceq \lambda_{\max}(G_S) I$.
Together with the equal traces and $\lambda_{\max}(W) \geq v_1^\top W v_1$, this gives

$$\pi_S = \frac{v_1^\top W v_1}{\mathrm{trace}(W)} \;\leq\; q_1(W) \;\leq\; \frac{\lambda_{\max}(G_S)}{\mathrm{trace}(G_S)}.$$

Writing $W = \sum_{i \in S} W_i$ as a sum of single-driver Gramians, the participation ratio of the unregularized
Gramian is $(\sum_i \mathrm{AC}_i)^2 / \sum_{i,j \in S} \langle W_i, W_j\rangle_F$. The cross terms $i \neq j$ are
not determined by single-region scores.

**Predictors of $q_1$.** Within each subject, $L_S$, mean average controllability and
$\lambda_{\max}(G_S)/\mathrm{trace}(G_S)$ are correlated with $q_1$ across the 100 random sets. The overlap and
effective rank of modal-controllability and cycle-participation sets are compared directly. Coverage distance is
the mean distance from each non-driver region to the nearest driver, compared with random sets and correlated
with $q_1$.
Tables: `additional_nodelevel_corr`, `additional_withinsubj`, `additional_drivergram`, `additional_commgeo`,
`additional_modal`, `additional_coverage`, `additional_coverage_rho`.

## Secondary null statistics and mediation

For every empirical and null network, the following statistics are recorded:

- the strength assortativity of the Brain Connectivity Toolbox;
- the binary rich-club coefficient at the 90th percentile of the empirical degree distribution;
- the weighted rich-club coefficient at the 75th, 85th, 90th and 95th percentiles;
- the Spearman correlation between edge weight and the product of the endpoints' binary degrees;
- the share of weight on edges among the 20% of regions with the highest binary degree;
- the hub strength fraction, the share of weight among the five strongest regions.

Separately for each null model and scale, the outcome (degree-set $q_1$ or the gap) is regressed on an indicator
for the empirical network, with subject fixed effects, with and without the candidate statistic as a covariate.
The proportion mediated is $1 - \beta_{\mathrm{direct}}/\beta_{\mathrm{total}}$. With one empirical network per
subject, these regressions extrapolate from the null distribution and are descriptive. The hub strength fraction
is computed from the same regions as the $k = 5$ degree set. The cycle-minus-degree comparison in null networks
at 114 regions is also given here.
Tables: `additional_null114`, `additional_mediation`, `additional_mediators`.

## Target states

The minimum energy to reach a unit-norm target $x^\ast$ from zero is $x^{\ast\top}W_\epsilon^{-1}x^\ast$. For each
subject, the energy difference is $100 \times (E_{\mathrm{cycle}} - E_{\mathrm{degree}})/E_{\mathrm{degree}}$,
averaged over $k$; negative values favor cycle-participation drivers. The targets are:

- **Networks.** Indicator vectors of the seven Yeo networks. Each parcel is assigned to the network held by the
  majority of its fsaverage5 vertices, excluding the medial wall.
- **Anatomical masks.** Exact matches of base-region labels. The occipital mask comprises the cuneus, lateral
  occipital, lingual and pericalcarine parcels.
- **Principal gradient.** The Margulies et al. (2016) principal gradient, resampled from fsLR to fsaverage5
  through the registered fsLR sphere and averaged within parcels.
- **Single parcels.** One target per parcel.

In the driver-exclusion analysis, each target is set to zero at the drivers of both methods and renormalized.
Distance from a target to a driver set is the target-weighted mean distance to the nearest driver. Structural
coupling is $\sum_i x_i^{\ast 2}\sum_{s \in S}(A_0)_{is}$. The energy difference is related across subjects to
the differences in coupling and distance, and compared at equal distance. The floor share is the fraction of
target energy carried by Gramian directions at the regularization floor.
Tables: `additional_targets_full`, `additional_targets_excl`, `additional_overlap`, `additional_proximity`,
`additional_singleparcel`, `additional_floor_targets`, `additional_eps_targets`, `additional_parcels`.

## Reciprocal hub lesions

The top $h \in \{5, 10, 15\}$ regions by degree strength, the top $h$ by cycle participation, or $h$ random regions
(100 draws) are removed, and both driver-selection strategies are evaluated under each lesion. Each lesioned
network is restabilized with the same $c$. There are two protocols:

- **Static:** drivers are the top-$k$ surviving regions by the intact ranking.
- **Recompute:** rankings are recomputed on the lesioned network.

Losses (intact minus lesioned) are given raw and in percentile units relative to random sets on the same network.
The change in average energy is related to $\log_{10}[(N - h)/N]$.
Tables: `additional_lesion_main`, `additional_lesion_cross`, `additional_lesion_protocol`, `additional_lesion_abs`,
`additional_lesion_energy`.
