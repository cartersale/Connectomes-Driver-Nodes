# Hub Inputs Funnel Control into the Connectome's Dominant Mode

Analysis code for:

> Carter Sale, Marco Coraggio, Mengsen Zhang, Michael J. Richardson.
> *Hub Inputs Funnel Control into the Connectome's Dominant Mode.*

## Overview

We model 70 human structural connectomes, at three parcellation scales (68, 114 and 219 cortical regions), as
linear dynamical systems driven at a few regions, and compare driver sets chosen by degree strength, eigenvector
centrality, average and modal controllability, betweenness, participation coefficient, and cycle and triangle
participation from persistent homology. The paper shows that:

1. average control energy does not distinguish sparse driver sets, because most Gramian eigenvalues sit at the
   regularization floor;
2. combining hubs concentrates controllability onto a single pattern (high $q_1 = \lambda_{\max}(W)/\mathrm{tr}\,W$);
3. that pattern is the connectome's leading eigenvector $v_1$, a mode of the midline structural core;
4. the next slowest modes divide the core and fade faster by the spectral gap, so damping and the gap control
   how strongly hub inputs concentrate; and
5. strong links among core regions (edge-weighted strength assortativity) give human connectomes a larger
   spectral gap than null networks that preserve strength, geometry or topology.

## Installation

Requires [uv](https://docs.astral.sh/uv/). The environment is locked (`uv.lock`) to the package versions that
produced the paper's results; uv installs Python 3.12 if needed.

```bash
git clone https://github.com/cartersale/Connectomes-Driver-Nodes.git && cd Connectomes-Driver-Nodes
uv sync --frozen --all-extras
uv run pytest                 # unit tests, about a minute
```

## Data

The structural connectomes are not redistributed here. See [docs/data.md](docs/data.md) for where to get each
input and where to put it. In short:

| Input | Location |
| --- | --- |
| Structural connectomes (Griffa et al., 2019; 70 subjects) | `data/raw/Individual_Connectomes.mat` |
| Cammoun (2012) parcellation, fsaverage5 annotations | `data/external/atlas_cache/atl-cammoun2012/fsaverage5/` |
| Yeo (2011) 7- and 17-network fsaverage5 annotations | `data/external/yeo2011_fsaverage5/` |
| Margulies (2016) gradients and surface atlases | downloaded by neuromaps on first use into `$NEUROMAPS_DATA` (default `~/neuromaps-data`) |

Check every input against the checksums the paper's results were computed from:

```bash
uv run python scripts/reproduce.py --check-inputs
```

## Reproducing the paper

```bash
uv run python scripts/reproduce.py --run paper
```

This runs every analysis, figure, table and number of the paper, in dependency order, into `runs/paper/`, and
validates the result against the release reference in `reference/`. It can be interrupted and resumed: finished
steps are skipped, and a step whose script or configuration has changed since it ran is reported instead of
silently reused. The full run takes several days on one machine, almost all of it in the null-network step
(`null_models/01_run_nulls`); see [docs/reproduction.md](docs/reproduction.md) for runtimes, options
(`--list`, `--dry-run`, `--only`, `--force`, `--reuse`) and how results are checked.

Without the data, the same pipeline runs on synthetic connectomes in a few minutes:

```bash
uv run python scripts/reproduce.py --smoke     # every step that does not need external atlas data
uv run pytest -m slow                          # the same, compared with tests/golden/synthetic.json
```

## Results

The figures are written to `runs/paper/figures/` (PDF, SVG and PNG). Everything else the manuscript uses ends up in
`runs/paper/paper/`:

| Path | Contents |
| --- | --- |
| `tables/` | The full tables as LaTeX (`S4_1.tex`, ...), with `notes/` describing how each was computed, and the tables the Supporting Information prints (`C_*.tex`, `S_canonical.tex`) |
| `key_results.json` | Every number quoted in the paper, unrounded, with its source file and what it refers to |
| `numbers.csv` | The same as a table, with the presentation text and the text the manuscript prints |
| `numbers.tex` | One LaTeX macro per number (`\newcommand{\MQOneHubTwoFiveMin}{67}`) |
| `validation_report.md` | Comparison with the release reference and with the manuscript text |
| `provenance.json` | Configuration, seeds, input checksums, package versions and git commits of the run |

The intermediate outputs of each analysis are in `runs/paper/<group>/` (for example `runs/paper/concentration/`).
The full tables are also published in [`supplementary/`](supplementary/) as CSV files with an index, together with
analyses that are not part of the paper ([`supplementary/additional_analyses.md`](supplementary/additional_analyses.md)).
[docs/paper_map.md](docs/paper_map.md) maps every figure panel and table of the manuscript to the script that
makes it and the outputs it reads.

## Repository layout

```text
config/default.toml        every analysis setting (config/smoke.toml: the synthetic smoke run)
src/hoc/                   library: preprocessing, control, topology, rankings, targets, run manifests
  analysis/                helpers shared by the analysis steps (null models, sensitivity, target states, ...)
  figstyle.py              shared figure style and brain-surface drawing
scripts/reproduce.py       runs the whole pipeline
scripts/analysis/          one folder per analysis, scripts numbered in run order
  primary/                 preprocessing, node rankings, driver sets, energy and geometry, robustness
  spectra/                 Gramian spectra of driver sets, random and matched comparison sets
  sensitivity/             damping, time horizon, regularization floor, driver-set size, nctpy equivalence
  modal_controllability/   set-level concentration against node-level modal and average controllability
  concentration/           leading-mode share q1, its projection, decay with damping, spectral gap
  mechanisms/              hub loading on v1, anatomy of v1-v3, model networks, Gramian alignment, core cuts
  null_models/             strength-, geometry- and topology-preserving null networks
  target_states/           named and single-parcel target states, spin tests, network enrichment
  lesions/                 reciprocal hub lesions
  topology/                stability of cycle participation across representatives and perturbations
scripts/figures/           Figures 1-5
scripts/paper/             tables, number registry, validation, assembly, manuscript sync, CSV publication
supplementary/             every full table as CSV, with an index and the methods of the additional analyses
reference/                 release reference: input checksums, paper numbers, table checksums
tests/                     unit tests and the synthetic-pipeline regression test
docs/                      data, reproduction, pipeline and figure/table map, limitations
```

## Citation

If you use this code, please cite the article (see [CITATION.cff](CITATION.cff)):

```bibtex
@article{sale2026hubinputs,
  title  = {Hub Inputs Funnel Control into the Connectome's Dominant Mode},
  author = {Sale, Carter and Coraggio, Marco and Zhang, Mengsen and Richardson, Michael J.},
  year   = {2026},
}
```

## License

MIT (see [LICENSE](LICENSE)). The connectome data and atlases are distributed by their authors under their own
terms.
