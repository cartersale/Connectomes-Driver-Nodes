# Input data

None of the inputs are redistributed in this repository. After placing them, check them with

```bash
uv run python scripts/reproduce.py --check-inputs
```

which compares all 50 files with `reference/inputs.sha256`, the checksums of the files the paper's results
were computed from.
Every run also records the checksum of each of these files that is present in its `manifest.json` (under
`inputs`, with the same names), so the run's provenance names the exact data it read.

## Structural connectomes

Griffa, A., Alemán-Gómez, Y. & Hagmann, P. (2019). *Structural and functional connectome from 70 young healthy
adults.* Zenodo. https://doi.org/10.5281/zenodo.2872624

Download `Individual_Connectomes.mat` and place it at

```text
data/raw/Individual_Connectomes.mat
```

The pipeline reads the `connMatrices.SC` stack at the Lausanne scales 68, 114 and 219 (cortical regions; see
`config/default.toml`, `[analysis] parcellation_scales`). Preprocessing (symmetrization, `log(1 + w)`,
max-normalization, spectral stabilization) is described in the paper's Methods and implemented in
`src/hoc/dataset.py` and `src/hoc/preprocessing.py`.

## Cammoun (2012) parcellation

The fsaverage5 annotation files of the Lausanne/Cammoun 2012 atlas at all five resolutions, named as
netneurotools' `fetch_cammoun2012(version="fsaverage5")` names them:

```text
data/external/atlas_cache/atl-cammoun2012/fsaverage5/atl-Cammoun2012_space-fsaverage5_res-{033,060,125,250,500}_hemi-{L,R}_deterministic.annot
```

For example:

```bash
uv run python -c "from netneurotools.datasets import fetch_cammoun2012; fetch_cammoun2012('fsaverage5', data_dir='data/external/atlas_cache')"
```

They give region labels, the parcel centroids used by the geometry-preserving nulls, and the brain surfaces
in the figures (drawn on nilearn's packaged fsaverage5 mesh).

## Yeo (2011) resting-state networks

The 7- and 17-network parcellations on fsaverage5 (Yeo et al., 2011; Computational Brain Imaging Group):

```text
data/external/yeo2011_fsaverage5/{lh,rh}.Yeo2011_{7,17}Networks_N1000.annot
```

Used only by the target-state analyses (`scripts/analysis/target_states/`).

## Margulies (2016) gradients

The first two functional gradients (`fcgradient01`, `fcgradient02`, fsLR 32k) and the fsLR / fsaverage surface
atlases are downloaded by neuromaps the first time the target-state analyses run, into `$NEUROMAPS_DATA`
(default `~/neuromaps-data`). `reference/inputs.sha256` lists them with the prefix `$NEUROMAPS_DATA/`.
