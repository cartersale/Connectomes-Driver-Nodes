# Target-state analyses

The target-state analyses of the preprint of this work are part of the pipeline but not of the paper. This page
lists their steps, inputs and tables. Two of their outputs are used by the paper: the spin test of the correlation
between degree strength and cycle participation (`05_spin_tests`) and the overlap of the 15 highest-ranked regions
(`06_network_enrichment`), both quoted in Supplementary Section S3.

## Steps

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

`primary/01_primary_analysis` also builds the targets (`src/hoc/targets.py`, settings `[targets]` and
`[anatomy]` in `config/default.toml`) and computes their energies for every driver set. Shared helpers are in
`src/hoc/analysis/target_states.py`.

## Inputs

**Yeo (2011) resting-state networks.** The 7- and 17-network parcellations on fsaverage5 (Yeo et al., 2011;
Computational Brain Imaging Group):

```text
data/external/yeo2011_fsaverage5/{lh,rh}.Yeo2011_{7,17}Networks_N1000.annot
```

**Margulies (2016) gradients and surface atlases.** The first two functional gradients (`fcgradient01`,
`fcgradient02`, fsLR 32k) and the fsLR / fsaverage surface atlases are downloaded by neuromaps on first use
(`primary/01_primary_analysis` and the target-state steps), into `$NEUROMAPS_DATA` (default
`~/neuromaps-data`). `reference/inputs.sha256` lists them with the prefix `$NEUROMAPS_DATA/`.

Both are needed by `scripts/reproduce.py --run paper` and checked by `--check-inputs`.

## Tables

`scripts/paper/sections/s10.py` builds the tables; `scripts/paper/publish_tables.py` publishes them in
`supplementary/tables/`. Their methods are in `supplementary/additional_analyses.md` (sections "Target states"
and "Rank divergence, cortical gradients and resting-state networks").

| Builder | Full tables (id → published CSV) | Reads (run-relative) |
|---|---|---|
| `scripts/paper/sections/s10.py` | S10_1 → `additional_targets_full.csv`<br>S10_2 → `additional_targets_excl.csv`<br>S10_3 → `additional_overlap.csv`<br>S10_4 → `additional_proximity.csv`<br>S10_5 → `additional_singleparcel.csv`<br>S10_6 → `additional_eps_targets.csv`<br>S10_8 → `additional_floor_targets.csv`<br>S10_9 → `additional_spin.csv`<br>S10_10 → `additional_rankdiv.csv`<br>S10_11 → `additional_parcels.csv` | `target_states/driver_target_overlap.csv`<br>`target_states/named_target_summary.csv`<br>`target_states/proximity_controls.csv`<br>`target_states/rank_divergence_network_enrichment.csv`<br>`target_states/sensitivity_long.csv.gz`<br>`target_states/sensitivity_summary.csv`<br>`target_states/single_parcel_summary.csv`<br>`target_states/spin_tests.csv`<br>`target_states/target_mask_parcels.csv` |
