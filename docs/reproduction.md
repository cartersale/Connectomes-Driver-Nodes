# Reproducing the paper

```bash
uv sync --frozen --all-extras
uv run python scripts/reproduce.py --check-inputs     # the external data match the paper's inputs
uv run python scripts/reproduce.py --run paper        # everything, into runs/paper/
```

`scripts/reproduce.py` runs every step listed in [pipeline.md](pipeline.md), in order, with
`config/default.toml`. At the end, `runs/paper/paper/` holds the figures, tables, number registry, validation
report and provenance record (see the README). The step `paper/validate` compares every number and table with
the release reference in `reference/` and the run fails if anything differs.

## Runtime

Measured on a 14-core Apple M4 Pro:

| Step | Time |
| --- | --- |
| `null_models/01_run_nulls` (50 null networks × 3 models × 70 subjects × 3 scales) | several days |
| `lesions/01_run_reciprocal_lesion` | ~65 min |
| `primary/01_primary_analysis` | ~50 min |
| `topology/run_perturbation_stability` | ~26 min |
| `primary/02_robustness` | ~24 min |
| all other steps together | ~45 min |

Steps use up to 12 worker processes (`HOC_N_JOBS` overrides the number, see `src/hoc/utils.py`). Each worker
pins its BLAS library to one thread: at 68 regions some set-level metrics differ at about $10^{-6}$ (relative)
when BLAS uses several threads, enough to move a percentile by one rank, so results are bit-reproducible only
with the default `HOC_BLAS_THREADS=1`.

## Resuming, staleness and reuse

The run directory holds `manifest.json`, which records for each step its status, the checksum of its script,
the configuration digest, random seeds, arguments, git state and package versions of every start and resume,
and the checksum of every output. The runner uses it to decide what to do:

| Status (`--list`) | Meaning | What `reproduce.py` does |
| --- | --- | --- |
| `done` | finished with the current script and config, after everything it depends on | skips it |
| `running` | interrupted | reruns it; long steps (null networks, lesions) resume from their saved per-subject chunks |
| `stale` | finished, but its script or config changed since, or a step it depends on finished later | stops and names it |
| `pending` | not run yet | runs it |
| `reused` | outputs taken from another run with `--reuse` | skips it |

Useful options:

```bash
uv run python scripts/reproduce.py --run paper --list            # status of every step
uv run python scripts/reproduce.py --run paper --dry-run         # what would run
uv run python scripts/reproduce.py --run paper --only figures    # figures and everything they need
uv run python scripts/reproduce.py --run paper --force concentration/03_decay_collapse   # rerun a step and its dependents
uv run python scripts/reproduce.py --run paper --rerun-stale     # rerun every stale step and its dependents
```

`--reuse STAGE=RUN` copies a step's outputs from another run instead of computing them. Every file is checked
against the sha256 recorded in that run's manifest, the input data of the two runs must have the same
checksums, a config difference is recorded as a diff, and the reuse is listed in the validation report and
`provenance.json`. It exists for the null-network step:

```bash
uv run python scripts/reproduce.py --run paper --reuse null_models/01_run_nulls=<earlier run>
```

Individual steps can also be run directly, e.g.
`uv run python scripts/analysis/concentration/01_compute_q1.py --run paper`; each script checks that the steps
it needs have finished in that run.

## Validation

```bash
uv run python scripts/paper/validate.py --run paper                                  # what paper/validate does
uv run python scripts/paper/validate.py --run paper --manuscript <manuscript folder>  # also check the manuscript
uv run pytest -m data                                                                 # same check, as a test
```

The release reference (`reference/paper_numbers.json`, `reference/tables.sha256`) was written from a validated
full run with `validate.py --write-reference`. It is never updated implicitly: replacing it needs `--reason`,
which is stored in the file together with the run and commit it came from.

`--manuscript <folder>` (with `--main <file>`, default `main.tex`) compares the run with the main text and
`supp_mat.tex` in that folder: tables the supplement inputs that
the run does not generate, number macros the text uses that the registry does not define, and numbers whose
computed text differs from what the documents print. `scripts/paper/sync.py --manuscript <folder>` lists (and
with `--apply` copies) the generated tables and `numbers.tex` into the manuscript folder, and the figures with
`--figures`; see
[paper_map.md](paper_map.md) for the table layout and `scripts/paper/publish_tables.py`.

## Smoke run

`config/smoke.toml` points the pipeline at synthetic connectomes, labels and gradients
(`src/hoc/synthetic.py`): eight subjects at all three scales, with small ensembles (`[sampling]`). It needs no
downloads and runs every step except those that need external atlas data or the real cortical anatomy (target
states, the core-cut analysis, the brain-surface figures, and the paper steps that depend on them):

```bash
uv run python scripts/reproduce.py --smoke      # into runs/smoke/
uv run pytest -m slow                           # the same, fingerprinted against tests/golden/synthetic.json
```

If a change to the code is meant to change synthetic results, regenerate the fingerprints with
`uv run python tests/test_synthetic_pipeline.py --update` and explain the difference in the commit.
