# Additional analyses

These analyses were run for the study but are not part of the paper. Their results are in the
`tables/additional_*.csv` files listed in the [README](README.md). Notation follows the paper. Unless stated
otherwise, values are given for 68, 114 and 219 regions in that order, at $c = 0.1$ and $\epsilon = 10^{-5}$, with
$k \in \{1,2,3,5\}$ and 70 subjects.

## Larger driver sets

The cycle-minus-degree comparison of effective rank and participation ratio was repeated for driver sets of up to
20% of regions: $k \in \{1,2,3,5,7,10,14\}$ at 68 regions, $\{1,2,3,5,6,10,11,23\}$ at 114 regions, and
$\{1,2,3,5,10,11,22,44\}$ at 219 regions. $q_1$ was not computed for these sizes.
Table: `additional_largek`.

## Decay-ratio collapse

If the spectral gap $g$ sets the size of the effect, curves plotted against the decay ratio $d = c/(c + g)$ should
coincide across scales more closely than curves plotted against $c$. The collapse was partial. Above
$d \approx 0.15$ the three scales nearly coincided, but at small $d$ the plateau still rose with the number of
regions. On matched windows, the between-scale dispersion fell by about 30% (cycle-minus-degree $q_1$: 0.022 to
0.015; effective rank: 0.037 to 0.026; degree-minus-random $q_1$: 0.030 to 0.021). Per-subject sign changes in
effective rank spread less across scales on the $d$ axis (ratio of the largest to the smallest scale median 1.4,
against 2.0 on the $c$ axis) but still differed between scales (Kruskal–Wallis $p < 10^{-3}$). The test has
limited power because the mean gap differs little between scales (0.43, 0.54 and 0.44).
Table: `additional_collapse`. Figure: `concentration/fig_decay_collapse.png` in the run directory.

## Rank divergence, cortical gradients and resting-state networks

Rank divergence is degree rank minus cycle-participation rank. Across parcels of the subject-averaged
connectome, it was weakly correlated with the principal cortical gradient of Margulies et al. (2016)
($\rho = -0.28$, $-0.26$, $-0.16$), but no correlation survived spin tests ($p_{\mathrm{spin}} = 0.16$, 0.12, 0.23;
10,000 rotations). It was not correlated with the second gradient. Averaged within the seven resting-state
networks of Yeo et al. (2011), it showed no consistent enrichment in any network after spin tests.
Tables: `S3_spin` (correlations), `additional_rankdiv` (network enrichment).

## Cycle-participation representatives and perturbed connectomes

The four representative conventions are described in Section S3 of the Supporting Information. Compared with
random sets whose members were matched in strength rank to each convention's drivers, no convention had an
effective rank higher by more than 0.12 at any scale or $k$, and most were at or below this baseline.

Rankings and Gramians were also recomputed on perturbed connectomes. The perturbations were multiplicative
Gaussian noise on existing edge weights ($\sigma = 0.05$, 0.10, 0.20) and random removal of existing edges
($p = 0.05$, 0.10, 0.20), with 50 perturbations per level, subject and scale. When a perturbed weight exceeded 1,
the matrix was renormalized by its maximum; this happened in 51% to 65% of the noise perturbations.

Ripser cycle participation kept a median rank correlation of 0.94 to 0.95 with the unperturbed ranking at
$\sigma = 0.05$, and 0.56 to 0.63 after removing 20% of edges. Even then it was more similar to the subject's own
unperturbed ranking than the rankings of different subjects were to each other. Degree strength was more stable
than every convention. The highest-ranked regions changed more than the ranking as a whole. The cycle-minus-degree
difference in effective rank was nearly unchanged on perturbed connectomes.
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

Both bounds held in every driver set analyzed. At $c = 0.1$ the upper bound exceeded $q_1$ by a median of 0.01 to
0.04, whereas the lower bound was loose for small sets. Writing $W = \sum_{i \in S} W_i$ as a sum of single-driver
Gramians, the participation ratio of the unregularized Gramian is
$(\sum_i \mathrm{AC}_i)^2 / \sum_{i,j \in S} \langle W_i, W_j\rangle_F$. The cross terms $i \neq j$ are not determined
by single-region scores.

**Predictors of $q_1$.** Across each subject's random sets, $L_S$, mean average controllability and
$\lambda_{\max}(G_S)/\mathrm{trace}(G_S)$ were correlated with $q_1$ from $k = 3$ onward, more strongly for larger
$k$. With two drivers at $c = 0.1$, the correlations were weak or negative. Modal-controllability sets and
cycle-participation sets shared almost no regions (one region in one subject across all scales and sizes). At
$k = 3$, cycle-participation sets had the higher effective rank (median differences 0.44, 0.48 and 0.74; positive
in 55, 65 and 69 subjects). The distance from each non-driver region to the nearest driver was smallest for
betweenness sets and largest for modal-controllability sets, and this coverage did not follow the ordering of
criteria by $q_1$.
Tables: `additional_nodelevel_corr`, `additional_withinsubj`, `additional_drivergram`, `additional_commgeo`,
`additional_modal`, `additional_coverage`, `additional_coverage_rho`.

## Secondary null statistics and mediation

For every empirical and null network, we recorded the following statistics:

- the strength assortativity of the Brain Connectivity Toolbox;
- the binary rich-club coefficient at the 90th percentile of the empirical degree distribution;
- the weighted rich-club coefficient at the 75th, 85th, 90th and 95th percentiles;
- the Spearman correlation between edge weight and the product of the endpoints' binary degrees;
- the share of weight on edges among the 20% of regions with the highest binary degree;
- the hub strength fraction, the share of weight among the five strongest regions.

Separately for each null model and scale, we regressed the outcome (degree-set $q_1$ or the gap) on an indicator
for the empirical network, with subject fixed effects. Each regression was fit with and without the candidate
statistic as a covariate. The proportion mediated was $1 - \beta_{\mathrm{direct}}/\beta_{\mathrm{total}}$.

With one empirical network per subject, these regressions extrapolate from the null distribution and are
descriptive only. The hub strength fraction removed 62% to 93% of the difference in $q_1$, but it is computed
from the same regions as the $k = 5$ degree set and is therefore partly circular. Rich-club coefficients removed at
most 26%, and usually none. The cycle-minus-degree comparison in null networks at 114 regions is also given here.
Tables: `additional_null114`, `additional_mediation`, `additional_mediators`.

## Target states

The minimum energy to reach a unit-norm target $x^\ast$ from zero is $x^{\ast\top}W_\epsilon^{-1}x^\ast$. For each
subject we computed $100 \times (E_{\mathrm{cycle}} - E_{\mathrm{degree}})/E_{\mathrm{degree}}$, averaged over $k$.
Negative values favor cycle-participation drivers, and medians are reported. The targets were:

- **Networks.** Indicator vectors of the seven Yeo networks. Each parcel was assigned to the network held by the
  majority of its fsaverage5 vertices, excluding the medial wall (median purity 0.62, 0.68 and 0.80).
- **Anatomical masks.** Exact matches of base-region labels. The occipital mask comprises the cuneus, lateral
  occipital, lingual and pericalcarine parcels.
- **Principal gradient.** The Margulies et al. (2016) principal gradient, resampled from fsLR to fsaverage5
  through the registered fsLR sphere and averaged within parcels.

In the driver-exclusion analysis, each target was set to zero at the drivers of both methods and renormalized.
Distance from a target to a driver set was the target-weighted mean distance to the nearest driver. Structural
coupling was $\sum_i x_i^{\ast 2}\sum_{s \in S}(A_0)_{is}$.

**Occipital and visual targets.** Cycle-participation sets reached the occipital mask with less energy ($-8.2\%$,
$-8.5\%$, $-1.2\%$), and the same held for the Yeo visual network ($-7.2\%$, $-8.9\%$, $-1.8\%$).

**Drivers inside the targets.** Drivers often lay inside the targets, and asymmetrically. Cycle-participation
drivers fell inside the visual network in 16%, 22% and 15% of cases, against 3%, 1% and 1% for degree-strength
drivers. Excluding drivers left the visual effect in place ($-5.7\%$, $-8.9\%$, $-1.7\%$).

**Coupling and distance.** Across subjects, the energy difference for nearly every target followed the difference
in structural coupling. For the visual network, $\rho = -0.77$, $-0.84$ and $-0.82$. It also followed the
difference in distance to the nearest driver. At equal distance, the visual effect was not significant at 68
regions ($-1.1\%$) and was smaller at 114 and 219 regions ($-6.9\%$, $-2.8\%$). Across single-parcel targets, the energy
difference was predicted by the difference in coupling ($\rho = -0.67$ to $-0.79$; $-0.90$ to $-0.94$ with
drivers excluded).

**The regularization floor.** The median share of target energy carried by directions at the regularization
floor was 86% to nearly 100%.

Each method therefore reached targets near, and strongly connected to, its own drivers more cheaply.
Tables: `additional_targets_full`, `additional_targets_excl`, `additional_overlap`, `additional_proximity`,
`additional_singleparcel`, `additional_floor_targets`, `additional_eps_targets`, `additional_parcels`.

## Reciprocal hub lesions

We removed the top $h \in \{5, 10, 15\}$ regions by degree strength, the top $h$ by cycle participation, or $h$
random regions (100 draws), and evaluated both driver-selection strategies under each lesion. Each lesioned
network was restabilized with the same $c$. There were two protocols:

- **Static:** drivers are the top-$k$ surviving regions by the intact ranking.
- **Recompute:** rankings are recomputed on the lesioned network.

Losses are reported raw and in percentile units relative to random sets on the same network.

**Degree lesions.** Under the static protocol, degree-strength sets lost more effective rank than
cycle-participation sets (difference in loss $-0.35$; 66 of 70 subjects). Under the recompute protocol, this
reversed: degree-strength sets gained effective rank. The next-ranked degree regions have an unusually high
effective rank in the intact network (3.64, against 2.40 for the top-ranked degree set and 3.27 for
cycle-participation sets). They lose it when their neighboring hubs are removed.

**Cycle and random lesions.** The two strategies lost similar amounts in raw units. In percentile units,
cycle-participation sets lost 3.6 more points in effective rank and 5.2 more in participation ratio.

**Average energy.** The change in average energy after a lesion was determined by network size
($r = 0.989$ with $\log_{10}[(N - h)/N]$). The two strategies differed by at most 0.006 log units.

Tables: `additional_lesion_main`, `additional_lesion_cross`, `additional_lesion_protocol`, `additional_lesion_abs`,
`additional_lesion_energy`.
