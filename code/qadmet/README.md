# qadmet

TDC ADMET multitask training research platform. **Recipe (YAML) → Spec → Runner → Artifact** flow
declares "what experiment to run" separated from the code. backbone / fusion / head / loss /
strategy / feature builder / calibrator are all registry-based — a new modality·loss·strategy is
registered with just one file + one decorator line.

This directory is the core released as the training·evaluation engine for the paper (*When and Why to Fuse Molecular Representations for ADMET Prediction*).
The recipes and run order to reproduce the paper's 10 TDC base models and 5 Polaris ensemble members
are organized in [`../TRAINING.md`](../TRAINING.md).

> **This engine is not needed to reproduce figures·tables.** Figures/Tables are reproduced from precomputed predictions·leaderboards
> (`../../data/recompute_data/`) alone, in the 6 section notebooks under `../../notebooks/`
> (`00_pre_analysis.ipynb` + `31`–`35_*.ipynb`), without GPU·network. This engine is
> used when you want to retrain those predictions·leaderboards **from scratch**.

## Installation

```bash
conda activate admet-mtl   # environment with heavy dependencies (torch/lightning/chemprop/transformers/rdkit/tdc) installed
cd code/qadmet
pip install -e .           # register qadmet CLI (entry point: qadmet.cli.main:main)
```

The `[project] dependencies` in `pyproject.toml` are minimal (`pyyaml` / `numpy` / `pandas`) —
heavy dependencies like torch / lightning / chemprop / transformers / peft / rdkit / tdc
are the responsibility of the conda env (`admet-mtl`). See [`../TRAINING.md`](../TRAINING.md) for reproduction environment details.

## CLI

| Subcommand | Role |
|---|---|
| `qadmet precompute --recipe <yaml>` | Precompute feature parquet (e.g. MoLFormer token embeddings). |
| `qadmet train --recipe <yaml>` | MTL / STL training. Saves both val + test splits to `predictions.parquet` immediately after training. |
| `qadmet evaluate --recipe <yaml>` | 7-column metric table (`dataset,metric,ours,ours_std,rank,n_total,Top,Top_model`) + leaderboard ranking → `leaderboard.csv`. |
| `qadmet recipe validate <path...>` | Validate recipe YAML schema. |
| `qadmet artifacts list [--kind <kind>]` | List saved artifacts. |
| `qadmet lineage show <kind> <name>` | Artifact dependency graph (graphviz dot). |

There is no separate `predict` subcommand — the train runner writes
`predictions.parquet` for both val + test immediately after training, so that result is itself the PredictionSet.

Basic flow (paper reproduction — see [`../TRAINING.md`](../TRAINING.md) for the full order):

```bash
# 1) precompute (MoLFormer backbone only, one-time)
qadmet precompute --recipe recipes/precompute/molformer_tokens.yaml

# 2) train (e.g. D-MPNN·MTL base model)
qadmet train --recipe recipes/train/mtl_dmpnn_robust_causal6_kendall_epoch100.yaml

# 3) evaluate → leaderboard.csv (e.g. for the above model)
qadmet evaluate --recipe recipes/evaluate/leaderboard_mtl_dmpnn_epoch100_s01234.yaml
```

CLI override (dot-notation, repeatable):

```bash
qadmet train --recipe recipes/train/mtl_dmpnn_robust_causal6_kendall_epoch100.yaml \
    --override trainer.max_epochs=2 --override trainer.seeds='[0]'
```

`--force` deletes and regenerates the existing artifact directory for the same run_id (= `sha1(recipe + git_sha)[:12]`).

## Directory

| Path | Description |
|---|---|
| [src/qadmet/](src/qadmet/) | Main package. core (registry/spec/recipe/artifact/lineage/runner) + tasks/datasets/features/modules/training/ensembles/evaluation/runners/cli. |
| [recipes/](recipes/) | Paper reproduction recipes (38 total). Per-kind directories + [`_base/`](recipes/_base/) composable base. |
| `LICENSE` | Apache-2.0. |

`artifacts/`, `data/`, `checkpoints*/`, `lightning_logs/`, `logs/` are generated at runtime and
are not included in the release.

## Recipes (38 total, paper reproduction set)

| kind | count | content |
|---|---|---|
| `train/` | 15 | 10 TDC-22 base models (5 MTL + 5 STL) + 5 Polaris ensemble members. |
| `evaluate/` | 10 | Recipes generating leaderboard.csv for the 10 TDC-22 models. |
| `_base/` | 9 | dataset / model / loss / trainer fragments `!include`d by kept recipes. |
| `groupings/` | 2 | `tdc_admet_22_causal6` (MTL 6-group) + `tdc_admet_22` (STL 22-task TaskSet registration). |
| `precompute/` | 2 | MoLFormer token embeddings (TDC / Polaris). |

See the table in [`../TRAINING.md`](../TRAINING.md) for the exact train / evaluate recipe filename mapping of each model.
The 5 Polaris models have no `evaluate/` recipe since Polaris Hub scores them.

## Recipe Validation

```bash
qadmet recipe validate recipes/train/*.yaml recipes/evaluate/*.yaml \
                       recipes/precompute/*.yaml
```

## MoLFormer HuggingFace revision pin

`ibm/MoLFormer-XL-both-10pct` uses custom modelling code (`trust_remote_code=True`), and
subsequent hub commits require functions not present in transformers 4.57.x, causing `ImportError`. The backbone
pins a working revision (in `src/qadmet/modules/backbones/molformer_lora.py`'s
`MOLFORMER_REVISION`). Keep this pin for reproducibility.
