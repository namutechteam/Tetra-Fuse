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

## Pretrained backbone weights

Two backbones load **frozen pretrained weights that are not shipped in this repository** and
must be downloaded once into `code/qadmet/data/weights/` (the exact paths the recipes expect).
MoLFormer-XL is fetched automatically from HuggingFace at the pinned revision (see below), and
D-MPNN is trained from scratch (no external weights).

| Backbone | Expected path (under `code/qadmet/`) | Source |
|---|---|---|
| CheMeleon | `data/weights/chemeleon_mp.pt` | https://github.com/JacksonBurns/chemeleon · weights on Zenodo https://zenodo.org/records/15460715 |
| GraphMVP | `data/weights/graphmvp/graphmvp_hybrid.pth` | https://github.com/chao1224/GraphMVP · pretrained checkpoints https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6 (see note below) |

**GraphMVP checkpoint — which one and how to prepare it.** From the Drive folder use
`pretraining_model/GraphMVP_hybrid/` — the **Hybrid (Eq. 8) variant trained with the OGB
"complete" atom/bond features** (9-d atom / 3-d bond), *not* the chirality-only classification
weights. This choice is deliberate: our 22 ADMET tasks mix 10 regression + 12 classification
endpoints, and the OGB-feature weights are what our GIN encoder (OGB `AtomEncoder`/`BondEncoder`)
loads. The raw Drive checkpoint is a dict `{model, model_3D, AE_2D_3D, AE_3D_2D}`; **extract only
the `model` key** (the 2D-GIN state_dict — the other keys are SSL auxiliaries not used
downstream) and save that as `graphmvp_hybrid.pth`:

```python
import torch
ckpt = torch.load("GraphMVP_hybrid/<downloaded>.pth", map_location="cpu")
torch.save(ckpt["model"], "data/weights/graphmvp/graphmvp_hybrid.pth")
```

```bash
cd code/qadmet
mkdir -p data/weights/graphmvp
# CheMeleon: download chemeleon_mp.pt from the Zenodo record → data/weights/chemeleon_mp.pt
# GraphMVP:  download pretraining_model/GraphMVP_hybrid/ (OGB-feature regression weights) from
#            the Drive folder, extract its "model" key (see snippet above)
#            → data/weights/graphmvp/graphmvp_hybrid.pth
```

If a weight file is absent, the corresponding backbone falls back to random-init "smoke" mode
(`checkpoint_path=None`), which will **not** reproduce the paper's results — the download is
required for any `train` run of the CheMeleon or GraphMVP models (and of MAP, which uses the
CheMeleon graph encoder). `rdkit2d`, `xtb`, and `unimol2` descriptors/embeddings are **not**
used by any of the 10 paper models, so no such files are needed.

## The base models and the ensemble

All recipe paths below are relative to the `qadmet/` core directory (`code/qadmet/`), i.e.
`code/qadmet/recipes/…`. Backbone hyperparameters are shared between the TDC-22 primary
benchmark and the Polaris external benchmark; only the dataset and task grouping differ
(TDC-22 = six mechanism-informed groups for MTL / 22-task set for STL, Polaris = one shared
6-endpoint group).

The paper evaluates **10 base models** on the TDC-22 benchmark — four backbones
(D-MPNN, CheMeleon, GraphMVP, MoLFormer) and the joint-fusion MAP model, each under both
MTL and STL. All 10 are needed to reproduce Figure 1 and Table 3; the final k=5 late-fusion
ensemble (marked ★) combines **two MTL learners** (D-MPNN·MTL, MoLFormer·MTL) and
**three STL learners** (CheMeleon·STL, GraphMVP·STL, MAP·STL).

### TDC-22 base models — train + evaluate recipes (all under `recipes/`)

| Model | Regime | Ensemble | Train recipe | Evaluate recipe |
|---|---|:---:|---|---|
| D-MPNN | MTL | ★ | `train/mtl_dmpnn_robust_causal6_kendall_epoch100.yaml` | `evaluate/leaderboard_mtl_dmpnn_epoch100_s01234.yaml` |
| CheMeleon | MTL | | `train/mtl_chemeleon_robust_causal6_kendall_epoch100.yaml` | `evaluate/leaderboard_chemeleon_robust_causal6_kendall.yaml` |
| GraphMVP | MTL | | `train/mtl_graphmvp_robust_causal6_kendall_epoch100.yaml` | `evaluate/leaderboard_graphmvp_robust_causal6_kendall.yaml` |
| MoLFormer | MTL | ★ | `train/mtl_molformer_robust_causal6_kendall_epoch100.yaml` | `evaluate/leaderboard_molformer_robust_causal6_kendall.yaml` |
| MAP | MTL | | `train/mtl_4modality_robust_causal6_nounimol2_epoch100.yaml` | `evaluate/leaderboard_4modality_robust_causal6_nounimol2.yaml` |
| D-MPNN | STL | | `train/stl_dmpnn_paper_gradclip_seed01234_epoch100.yaml` | `evaluate/leaderboard_stl_dmpnn_seed01234.yaml` |
| CheMeleon | STL | ★ | `train/stl_chemeleon_adaptivegnn_paper.yaml` | `evaluate/leaderboard_stl_chemeleon.yaml` |
| GraphMVP | STL | ★ | `train/stl_graphmvp_adaptivegnn_paper.yaml` | `evaluate/leaderboard_stl_graphmvp_seed01234.yaml` |
| MoLFormer | STL | | `train/stl_molformer_lora.yaml` | `evaluate/leaderboard_stl_molformer_seed01234.yaml` |
| MAP | STL | ★ | `train/stl_4modality_robust_nounimol2_seed01234.yaml` | `evaluate/leaderboard_stl_4modality_robust_nounimol2_seed01234.yaml` |

### Polaris ADME-Fang — train recipes (the 5 ensemble members only; scored by the Polaris Hub, no `evaluate` step)

| Model | Regime | Train recipe |
|---|---|---|
| D-MPNN | MTL | `train/mtl_dmpnn_robust_polaris_fang.yaml` |
| MoLFormer | MTL | `train/mtl_molformer_robust_polaris_fang.yaml` |
| CheMeleon | STL | `train/stl_chemeleon_paper_polaris_fang.yaml` |
| GraphMVP | STL | `train/stl_graphmvp_paper_polaris_fang.yaml` |
| MAP | STL | `train/stl_map_robust_polaris_fang.yaml` |

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
python -m qadmet.cli.main precompute --recipe recipes/precompute/polaris_fang_molformer_tokens.yaml

# 2) train the five models (5 seeds each)
python -m qadmet.cli.main train --recipe recipes/train/mtl_dmpnn_robust_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/train/mtl_molformer_robust_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_chemeleon_paper_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_graphmvp_paper_polaris_fang.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_map_robust_polaris_fang.yaml
```

Each run writes per-seed checkpoints and a long-format prediction parquet
(`seed, task, split, smiles, y_true, y_pred`). Polaris hides test labels, so the local
"leaderboard metrics" block reports NaN for the test split — this is expected, not an error;
scoring is done against the official Polaris test labels.

## End-to-end pipeline (TDC-22 primary benchmark)

The full order — from raw data to the figures/tables — is:

```bash
conda activate admet-mtl
cd code/qadmet
pip install -e .                       # registers the `qadmet` CLI (entry point qadmet.cli.main)

# 1) precompute  (only needed for the MoLFormer backbone — token embeddings, run once)
#    the TDC-22 token recipe; for Polaris use recipes/precompute/polaris_fang_molformer_tokens.yaml
python -m qadmet.cli.main precompute --recipe recipes/precompute/molformer_tokens.yaml

# 2) train the 10 base models (5 seeds each).  There is NO separate `predict` step:
#    each train run writes its own predictions.parquet (seed,task,split,smiles,y_true,y_pred)
#    for both the validation and the held-out-test split, right after training.
#    (exact filenames are in the TDC-22 table above, all under recipes/train/)
# --- 5 MTL ---
python -m qadmet.cli.main train --recipe recipes/train/mtl_dmpnn_robust_causal6_kendall_epoch100.yaml
python -m qadmet.cli.main train --recipe recipes/train/mtl_chemeleon_robust_causal6_kendall_epoch100.yaml
python -m qadmet.cli.main train --recipe recipes/train/mtl_graphmvp_robust_causal6_kendall_epoch100.yaml
python -m qadmet.cli.main train --recipe recipes/train/mtl_molformer_robust_causal6_kendall_epoch100.yaml
python -m qadmet.cli.main train --recipe recipes/train/mtl_4modality_robust_causal6_nounimol2_epoch100.yaml
# --- 5 STL ---
python -m qadmet.cli.main train --recipe recipes/train/stl_dmpnn_paper_gradclip_seed01234_epoch100.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_chemeleon_adaptivegnn_paper.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_graphmvp_adaptivegnn_paper.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_molformer_lora.yaml
python -m qadmet.cli.main train --recipe recipes/train/stl_4modality_robust_nounimol2_seed01234.yaml

# 3) evaluate each model → writes leaderboard.csv next to its predictions.parquet.
#    leaderboard.csv columns: dataset,metric,ours,ours_std,rank,n_total,Top,Top_model
#    where rank/n_total come from qadmet.evaluation.leaderboard.get_rank against the
#    TDC public-leaderboard snapshot (the rank-normalization ruler).
# --- 5 MTL ---
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_mtl_dmpnn_epoch100_s01234.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_chemeleon_robust_causal6_kendall.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_graphmvp_robust_causal6_kendall.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_molformer_robust_causal6_kendall.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_4modality_robust_causal6_nounimol2.yaml
# --- 5 STL ---
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_stl_dmpnn_seed01234.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_stl_chemeleon.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_stl_graphmvp_seed01234.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_stl_molformer_seed01234.yaml
python -m qadmet.cli.main evaluate --recipe recipes/evaluate/leaderboard_stl_4modality_robust_nounimol2_seed01234.yaml
```

Both `train` and `evaluate` write into each model's artifact directory
(`artifacts/<...>/<run>/`): `predictions.parquet` from `train`, `leaderboard.csv` from
`evaluate`. In the public baseline layout these live at
`artifacts/baselines/<model>/baseline/{predictions.parquet,leaderboard.csv}`.

## How the notebook inputs are produced

The figure/table notebook `../notebooks/regenerate_paper_materials.ipynb` reads raw predictions
and leaderboards from `../data/recompute_data/`. To wire a fresh
training run into the notebooks, copy the two per-model outputs there:

```bash
# for each of the 10 models produced by train + evaluate:
cp artifacts/<...>/<model>/.../predictions.parquet  ../../data/recompute_data/predictions/<model>.parquet
cp artifacts/<...>/<model>/.../leaderboard.csv      ../../data/recompute_data/leaderboards/<model>.csv
```

The notebooks then recompute **everything** from these files: solo/ensemble norm_scores
(the uniform-mean late fusion), the per-backbone and MAP-aggregation tables, the k-sweep,
LOO/Shapley, correlations, and the Wilcoxon statistics. The formats match exactly — the same
`leaderboard.csv` schema and the same `norm = (n_total - rank + 1) / n_total` rank
normalization are used by `qadmet evaluate` and by the notebook recompute; verified
identical across all released model×task rows. The two ruler/sweep inputs
(`tdc_leaderboard_snapshot.json`, `clean_stage1_current.csv`, `all1023.csv`) are shipped in
`recompute_data/` as-is.

## Late-fusion ensemble and submission

The ensemble is a **uniform (equal-weight) mean** of the five models' predictions, averaged
across the five random seeds (seed-averaged / cross-seed track). It has no fitted parameters,
so it is computed directly in the notebooks from the five `predictions.parquet` files — there
is no separate ensemble-training step. See `../data/polaris_all_model_test_preds.csv` for the
exact per-molecule predictions submitted to the Polaris Hub as `LateFusion-k5`.

## Data splits

- **TDC-22**: scaffold split via `GroupKFold` on raw SMILES (5 folds if ≥50 unique SMILES groups,
  else 3), 5 seeds. 21 tasks are scored (Bioavailability_Ma is trained as an MTL auxiliary but
  excluded from benchmark scoring due to an unreliable leaderboard rank; see Supplementary Note S1).
- **Polaris ADME-Fang**: the benchmark's own random split, 5 seeds, Pearson r metric.
