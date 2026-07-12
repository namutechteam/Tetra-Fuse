# Model Training — Reproduction Guide

This documents how the trained models behind the paper were produced. Regenerating the
figures/tables does **not** require any of this (use `../notebooks/`); this section is for
reproducing the underlying model checkpoints and predictions.

## Environment

- Python 3.12, conda environment `admet-mtl`.
- 2× NVIDIA RTX A6000 (24 GB each) used for training; a single 24 GB GPU is sufficient per model.
- Core stack: PyTorch + PyTorch Lightning, RDKit, HuggingFace Transformers 4.57.x, chemprop
  (D-MPNN), and the pretrained backbone weights (CheMeleon `chemeleon_mp.pt`,
  GraphMVP `graphmvp_hybrid.pth`, MoLFormer-XL `ibm/MoLFormer-XL-both-10pct`).
- Training is driven by the `qadmet` recipe/registry CLI: `python -m qadmet.cli.main train --recipe <recipe.yaml>`.

## The five models behind the ensemble

The final k=5 late-fusion ensemble combines two multi-task learners and three single-task learners,
all sharing identical backbone hyperparameters between the TDC-22 primary benchmark and the
Polaris external benchmark (only the dataset and task grouping differ).

All recipe paths below are relative to the `qadmet/` core directory (`code/qadmet/`).

| Model | Regime | Backbone | TDC-22 recipe | Polaris recipe |
|---|---|---|---|---|
| D-MPNN | MTL | chemprop message-passing | `recipes/train/mtl_dmpnn_robust_causal6_kendall_epoch100_tdcpxr25.yaml` | `recipes/train/mtl_dmpnn_robust_polaris_fang.yaml` |
| MoLFormer | MTL | MoLFormer-XL + LoRA (r=16, α=32) | `recipes/train/mtl_molformer_robust_causal6_kendall_epoch100_tdcpxr25.yaml` | `recipes/train/mtl_molformer_robust_polaris_fang.yaml` |
| CheMeleon | STL | AdapterGNN (bottleneck 128) | (STL variant of `mtl_*_tdcpxr25`) | `recipes/train/stl_chemeleon_paper_polaris_fang.yaml` |
| GraphMVP | STL | AdapterGNN (bottleneck 64) | `recipes/train/mtl_graphmvp_robust_causal6_kendall_epoch100_tdcpxr25.yaml` | `recipes/train/stl_graphmvp_paper_polaris_fang.yaml` |
| MAP | STL | 4-modality Modality-Attention-Pooling | `recipes/train/mtl_4modality_robust_causal6_nounimol2_epoch100_tdcpxr25.yaml` | `recipes/train/stl_map_robust_polaris_fang.yaml` |

Full recipe hyperparameters (LoRA rank/alpha/dropout, AdapterGNN bottlenecks, MAP entropy/tau
annealing, optimizer/lr/batch/epochs/scheduler) are in the paper's Supplementary **Table S5**
and in the recipe YAML files themselves.

## IMPORTANT — MoLFormer HuggingFace revision pin

`ibm/MoLFormer-XL-both-10pct` ships custom modelling code (`trust_remote_code=True`). A later
hub commit requires a `transformers` function (`create_bidirectional_mask`) not present in
transformers 4.57.x, causing an `ImportError`. The backbone therefore pins the working revision:

```python
# qadmet/src/qadmet/modules/backbones/molformer_lora.py
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
...
AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, revision=MOLFORMER_REVISION)
```

All paper results use this pinned revision. Without the pin, training fails to import; with a
different revision, the tokenizer/model code may differ. Keep the pin for reproducibility.

## Polaris ADME-Fang integration (external benchmark)

The 6-endpoint external benchmark is integrated into the qadmet core via two modules:

- `qadmet/src/qadmet/tasks/polaris_fang.py`
- `qadmet/src/qadmet/datasets/polaris_fang.py`

The raw data (already fetched from the Polaris Hub) is in `data_polaris_adme_fang/`
(`<endpoint>_train_val.csv`, `<endpoint>_test.csv`, `summary.json`). All six endpoints are
trained under a **single shared task group** (`polaris_fang_6.yaml`) — unlike the TDC-22
setup which uses six mechanism-informed groups — because all six are physicochemical/ADME
properties on overlapping compound sets.

### Run order (see `polaris_fang_run_all.sh`)

```bash
conda activate admet-mtl
cd code/qadmet          # the core engine directory in this release

# 0) stage the raw Polaris CSVs where the dataset module expects them, e.g.
#    cp -r ../data_polaris_adme_fang data/polaris_adme_fang
#    (see qadmet/src/qadmet/datasets/polaris_fang.py for the expected path)

# 1) precompute MoLFormer token embeddings (once)
python -m qadmet.cli.main precompute --recipe recipes/precompute_polaris_fang_molformer_tokens.yaml

# 2) train the five models (5 seeds each)
python -m qadmet.cli.main train --recipe recipes/mtl_dmpnn_robust_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/mtl_molformer_robust_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/stl_chemeleon_paper_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/stl_graphmvp_paper_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/stl_map_robust_polaris_fang.yaml
```

Each run writes per-seed checkpoints and a long-format prediction parquet
(`seed, task, split, smiles, y_true, y_pred`). Polaris hides test labels, so the local
"leaderboard metrics" block reports NaN for the test split — this is expected, not an error;
scoring is done against the official Polaris test labels.

## Late-fusion ensemble and submission

The ensemble is a **uniform (equal-weight) mean** of the five models' predictions, averaged
across the five random seeds (seed-averaged / cross-seed track). See
`../data/polaris_all_model_test_preds.csv` for the exact per-molecule predictions submitted
to the Polaris Hub as `LateFusion-k5`. No fitting is involved in the fusion step.

## Data splits

- **TDC-22**: scaffold split via `GroupKFold` on raw SMILES (5 folds if ≥50 unique SMILES groups,
  else 3), 5 seeds. 21 tasks are scored (Bioavailability_Ma is trained as an MTL auxiliary but
  excluded from benchmark scoring due to an unreliable leaderboard rank; see Supplementary Note S1).
- **Polaris ADME-Fang**: the benchmark's own random split, 5 seeds, Pearson r metric.
