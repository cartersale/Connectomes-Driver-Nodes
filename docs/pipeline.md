# Pipeline steps

Generated from `STEPS` in `scripts/reproduce.py` (`uv run python scripts/reproduce.py --list` shows the status of
each step in a run). Steps run in this order; each needs the steps listed under *requires*. Outputs are
relative to the run directory `runs/<name>/`; the runner checks that they exist after the step. Steps marked
*smoke* also run in the synthetic smoke run (`--smoke`); the others need external atlas data or the real
connectomes' anatomy.


## primary

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `primary/00_prepare_data` | `scripts/analysis/primary/00_prepare_data.py` | – | `processed/connectomes_scale-*.npz`<br>`metadata/*.csv` | yes |
| `primary/01_primary_analysis` | `scripts/analysis/primary/01_primary_analysis.py` | `primary/00_prepare_data` | `primary/node_scores.csv`<br>`primary/driver_sets.csv`<br>`primary/landscape_samples*`<br>`statistics/*.csv` | yes |
| `primary/02_robustness` | `scripts/analysis/primary/02_robustness.py` `--skip-perturbations` | `primary/00_prepare_data` | `robustness/sensitivity_summary.csv` | yes |

## spectra

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `spectra/01_compute_spectra` | `scripts/analysis/spectra/01_compute_spectra.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `spectra/set_metrics.csv`<br>`spectra/centroids_scale-*.csv`<br>`spectra/spectra_scale-*.npz` | yes |
| `spectra/02_analyze` | `scripts/analysis/spectra/02_analyze.py` | `spectra/01_compute_spectra` | `spectra/tables/t0_cycle_minus_degree.csv`<br>`spectra/tables/t2_eigenvalue_extremes.csv`<br>`spectra/tables/t6_condition_number_check.csv` | yes |

## sensitivity

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `sensitivity/00_validate` | `scripts/analysis/sensitivity/00_validate.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`primary/02_robustness` | `sensitivity/validation.json` | yes |
| `sensitivity/01_csweep_perron` | `scripts/analysis/sensitivity/01_csweep_perron.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `sensitivity/csweep_sets_methods.csv`<br>`sensitivity/csweep_spearman.csv` | yes |
| `sensitivity/02_nctpy_equivalence` | `scripts/analysis/sensitivity/02_nctpy_equivalence.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `sensitivity/nctpy_cohort_summary.csv`<br>`sensitivity/nctpy_equivalence_detail.csv` | yes |
| `sensitivity/03_horizon_eps_largek` | `scripts/analysis/sensitivity/03_horizon_eps_largek.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `sensitivity/horizon_eps_largek_tidy.csv.gz`<br>`sensitivity/epsilon_summary.csv`<br>`sensitivity/finite_horizon_summary.csv`<br>`sensitivity/large_k_summary.csv` | yes |
| `sensitivity/04_energy_floor` | `scripts/analysis/sensitivity/04_energy_floor.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `sensitivity/energy_floor_driver_sets.csv`<br>`sensitivity/energy_floor_landscape.csv`<br>`sensitivity/energy_floor_delta_log10E_summary.csv` | yes |
| `sensitivity/05_summarize_csweep` | `scripts/analysis/sensitivity/05_summarize_csweep.py` | `sensitivity/01_csweep_perron` | `sensitivity/csweep_cycle_minus_degree.csv`<br>`sensitivity/csweep_vs_random.csv`<br>`sensitivity/csweep_signflip.csv` | yes |
| `sensitivity/06_perron_diagnostic` | `scripts/analysis/sensitivity/06_perron_diagnostic.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `sensitivity/perron_by_method.csv`<br>`sensitivity/perron_diagnostic.csv` | yes |
| `sensitivity/07_report_tables` | `scripts/analysis/sensitivity/07_report_tables.py` | `sensitivity/01_csweep_perron`<br>`sensitivity/02_nctpy_equivalence`<br>`sensitivity/03_horizon_eps_largek`<br>`sensitivity/04_energy_floor`<br>`sensitivity/05_summarize_csweep`<br>`sensitivity/06_perron_diagnostic` | `sensitivity/csweep_per_k.csv`<br>`sensitivity/csweep_signflip_per_k.csv` | yes |

## modal_controllability

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `modal_controllability/01_compute_modal` | `scripts/analysis/modal_controllability/01_compute_modal.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `modal_controllability/set_level_modal.csv.gz`<br>`modal_controllability/node_level_modal.csv.gz` | yes |
| `modal_controllability/02_analyze_modal` | `scripts/analysis/modal_controllability/02_analyze_modal.py` | `primary/01_primary_analysis`<br>`modal_controllability/01_compute_modal` | `modal_controllability/within_subject_summary.csv`<br>`modal_controllability/node_level_correlations.csv`<br>`modal_controllability/set_overlap.csv` | yes |

## null_models

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `null_models/00_validate` | `scripts/analysis/null_models/00_validate.py` | `primary/00_prepare_data`<br>`spectra/01_compute_spectra` | `null_models/validation.json` | yes |
| `null_models/01_run_nulls` | `scripts/analysis/null_models/01_run_nulls.py` | `primary/00_prepare_data`<br>`spectra/01_compute_spectra` | `null_models/chunks/*/scale-*/*_sets.csv.gz` | yes |
| `null_models/02_analyze` | `scripts/analysis/null_models/02_analyze.py` | `null_models/01_run_nulls` | `null_models/inference.csv`<br>`null_models/network_inference.csv`<br>`null_models/null_fidelity.csv`<br>`null_models/sets.csv.gz` | yes |
| `null_models/03_gap_mediators` | `scripts/analysis/null_models/03_gap_mediators.py` | `primary/00_prepare_data`<br>`null_models/01_run_nulls`<br>`null_models/02_analyze` | `null_models/gap_split.csv`<br>`null_models/mediation_noncircular.csv`<br>`null_models/mediator_contrasts.csv` | yes |
| `null_models/04_rayleigh_bound` | `scripts/analysis/null_models/04_rayleigh_bound.py` | `primary/00_prepare_data`<br>`null_models/01_run_nulls` | `null_models/rayleigh/per_subject.csv` | yes |

## concentration

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `concentration/01_compute_q1` | `scripts/analysis/concentration/01_compute_q1.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`modal_controllability/01_compute_modal`<br>`spectra/01_compute_spectra` | `concentration/set_q1.csv.gz` | yes |
| `concentration/02_tables` | `scripts/analysis/concentration/02_tables.py` | `concentration/01_compute_q1` | `concentration/q1_vs_random.csv`<br>`concentration/q1_matched.csv`<br>`concentration/q1_perp.csv` | yes |
| `concentration/03_decay_collapse` | `scripts/analysis/concentration/03_decay_collapse.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `concentration/decay_curves.csv.gz`<br>`concentration/decay_gap_correlations.csv`<br>`concentration/decay_gap_partial_correlations.csv`<br>`concentration/fig_decay_collapse.png` | yes |
| `concentration/04_redundancy_geodesic` | `scripts/analysis/concentration/04_redundancy_geodesic.py` | `primary/00_prepare_data`<br>`concentration/01_compute_q1` | `concentration/redundancy_summary.csv` | yes |
| `concentration/06_er_avgctrl` | `scripts/analysis/concentration/06_er_avgctrl.py` | `concentration/01_compute_q1` | `concentration/er_avgctrl.csv` | yes |

## mechanisms

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `mechanisms/01_hub_loading` | `scripts/analysis/mechanisms/01_hub_loading.py` | `primary/00_prepare_data`<br>`null_models/01_run_nulls` | `mechanisms/hub_loading.csv.gz`<br>`mechanisms/hub_loading_summary.csv` | yes |
| `mechanisms/02_canonical_graphs` | `scripts/analysis/mechanisms/02_canonical_graphs.py` | `primary/00_prepare_data` | `mechanisms/canonical_graphs.csv.gz`<br>`mechanisms/canonical_summary.csv` | yes |
| `mechanisms/03_coverage_distance` | `scripts/analysis/mechanisms/03_coverage_distance.py` | `primary/00_prepare_data`<br>`concentration/01_compute_q1` | `mechanisms/coverage_by_criterion.csv`<br>`mechanisms/coverage_rho.csv` | yes |
| `mechanisms/04_leading_eigenvectors` | `scripts/analysis/mechanisms/04_leading_eigenvectors.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`spectra/01_compute_spectra` | `mechanisms/eigvec_subjects.csv`<br>`mechanisms/eigvec_regions.csv` | yes |
| `mechanisms/05_second_subspace` | `scripts/analysis/mechanisms/05_second_subspace.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`spectra/01_compute_spectra` | `mechanisms/second_subspace_subjects.csv`<br>`mechanisms/second_subspace_axes.csv` | yes |
| `mechanisms/07_core_cut` | `scripts/analysis/mechanisms/07_core_cut.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`spectra/01_compute_spectra` | `mechanisms/core_cut.csv.gz`<br>`mechanisms/core_cut_regions.csv` | no |
| `mechanisms/08_gramian_alignment` | `scripts/analysis/mechanisms/08_gramian_alignment.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `mechanisms/gramian_alignment.csv`<br>`mechanisms/node_v1_strength.csv` | yes |

## target_states

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `target_states/01_build_targets` | `scripts/analysis/target_states/01_build_targets.py` | `primary/00_prepare_data` | `target_states/target_catalog.csv`<br>`target_states/target_mask_parcels.csv`<br>`target_states/targets_scale-*.npz` | no |
| `target_states/02_target_energies` | `scripts/analysis/target_states/02_target_energies.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`target_states/01_build_targets` | `target_states/target_energies_long.csv.gz` | no |
| `target_states/03_target_stats` | `scripts/analysis/target_states/03_target_stats.py` | `target_states/02_target_energies` | `target_states/named_target_summary.csv`<br>`target_states/single_parcel_summary.csv`<br>`target_states/driver_target_overlap.csv` | no |
| `target_states/04_sensitivity` | `scripts/analysis/target_states/04_sensitivity.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis`<br>`target_states/01_build_targets` | `target_states/sensitivity_summary.csv` | no |
| `target_states/05_spin_tests` | `scripts/analysis/target_states/05_spin_tests.py` | `primary/00_prepare_data`<br>`primary/01_primary_analysis` | `target_states/spin_tests.csv` | no |
| `target_states/06_network_enrichment` | `scripts/analysis/target_states/06_network_enrichment.py` | `primary/00_prepare_data`<br>`target_states/01_build_targets`<br>`target_states/05_spin_tests` | `target_states/rank_divergence_network_enrichment.csv`<br>`target_states/top15_network_composition.csv` | no |
| `target_states/07_proximity_controls` | `scripts/analysis/target_states/07_proximity_controls.py` | `target_states/02_target_energies` | `target_states/proximity_controls.csv` | no |
| `target_states/09_report_tables` | `scripts/analysis/target_states/09_report_tables.py` | `target_states/01_build_targets`<br>`target_states/03_target_stats`<br>`target_states/04_sensitivity`<br>`target_states/05_spin_tests`<br>`target_states/06_network_enrichment`<br>`target_states/07_proximity_controls` | `target_states/*.md` | no |

## lesions

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `lesions/01_run_reciprocal_lesion` | `scripts/analysis/lesions/01_run_reciprocal_lesion.py` | `primary/00_prepare_data` | `lesions/lesion_rows.csv.gz` | yes |
| `lesions/02_analyze_reciprocal_lesion` | `scripts/analysis/lesions/02_analyze_reciprocal_lesion.py` | `lesions/01_run_reciprocal_lesion` | `lesions/summary_3x2.csv`<br>`lesions/crossover_stats.csv` | yes |

## topology

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `topology/run_representatives_all_scales` | `scripts/analysis/topology/run_representatives_all_scales.py` | `primary/00_prepare_data` | `topology/representatives/per_subject_scale-*.csv` | yes |
| `topology/run_perturbation_stability` | `scripts/analysis/topology/run_perturbation_stability.py` | `primary/00_prepare_data` | `topology/perturbation/per_draw_scale-*.csv` | yes |
| `topology/summarize_representatives` | `scripts/analysis/topology/summarize_representatives.py` | `topology/run_representatives_all_scales` | `topology/tables/rep_agreement.csv` | yes |
| `topology/summarize_perturbation` | `scripts/analysis/topology/summarize_perturbation.py` | `topology/run_perturbation_stability`<br>`topology/run_representatives_all_scales` | `topology/tables/perturb_rank_stability.csv` | yes |

## figures

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `figures/fig1` | `scripts/figures/fig1.py` | `primary/00_prepare_data` | `figures/fig1.pdf` | no |
| `figures/fig2` | `scripts/figures/fig2.py` | `concentration/01_compute_q1`<br>`mechanisms/08_gramian_alignment` | `figures/fig2.pdf` | yes |
| `figures/fig3` | `scripts/figures/fig3.py` | `mechanisms/04_leading_eigenvectors`<br>`mechanisms/08_gramian_alignment` | `figures/fig3.pdf` | no |
| `figures/fig3b` | `scripts/figures/fig3b.py` | `mechanisms/08_gramian_alignment` | `figures/fig3b.pdf` | yes |
| `figures/fig4` | `scripts/figures/fig4.py` | `mechanisms/05_second_subspace`<br>`concentration/02_tables`<br>`concentration/03_decay_collapse` | `figures/fig4.pdf` | no |
| `figures/fig5` | `scripts/figures/fig5.py` | `null_models/02_analyze`<br>`null_models/04_rayleigh_bound`<br>`mechanisms/07_core_cut` | `figures/fig5.pdf` | no |

## paper

| Step | Script | Requires | Checked outputs | Smoke |
|---|---|---|---|---|
| `paper/tables` | `scripts/paper/tables.py` | `concentration/02_tables`<br>`concentration/03_decay_collapse`<br>`concentration/04_redundancy_geodesic`<br>`null_models/03_gap_mediators`<br>`null_models/04_rayleigh_bound`<br>`target_states/09_report_tables`<br>`lesions/02_analyze_reciprocal_lesion`<br>`topology/summarize_perturbation`<br>`topology/summarize_representatives`<br>`mechanisms/02_canonical_graphs`<br>`mechanisms/03_coverage_distance`<br>`sensitivity/00_validate`<br>`sensitivity/07_report_tables`<br>`spectra/02_analyze`<br>`modal_controllability/02_analyze_modal` | `paper/tables/*.tex` | no |
| `paper/numbers` | `scripts/paper/key_numbers.py` | `paper/tables`<br>`mechanisms/04_leading_eigenvectors`<br>`mechanisms/05_second_subspace`<br>`mechanisms/07_core_cut`<br>`mechanisms/08_gramian_alignment`<br>`concentration/06_er_avgctrl`<br>`null_models/02_analyze`<br>`null_models/00_validate`<br>`primary/01_primary_analysis` | `paper/key_results.json`<br>`paper/numbers.csv`<br>`paper/numbers.tex` | no |
| `paper/validate` | `scripts/paper/validate.py` | `paper/numbers` | `paper/validation_report.md` | no |
| `paper/assemble` | `scripts/paper/assemble.py` | `paper/validate` | `paper/provenance.json` | no |
