# Tetra-Fuse: Late Fusion Preserves Task-Specific Strengths of Heterogeneous Molecular Encoders for ADMET Prediction

Code and weights for the paper *"Tetra-Fuse: Late Fusion Preserves Task-Specific
Strengths of Heterogeneous Molecular Encoders for ADMET Prediction."*

The paper compares representation- and prediction-level fusion across five molecular
foundation models on 21 TDC ADMET tasks, and validates the resulting uniform-mean
late-fusion ensemble (**Tetra-Fuse**, k=5) on the independent Polaris ADME-Fang
benchmark (6 regression endpoints; submitted to the Polaris Hub as `LateFusion-k5`).

---

## Which axis do you want?

This release serves two different needs, and they share almost nothing in terms of
setup cost. Pick one:

| | **Axis A — reproduce the paper** | **Axis B — predict new molecules** |
|---|---|---|
| Question | "Do the figures and tables come out of the data?" | "What does Tetra-Fuse say about *my* compound?" |
| Needs a GPU | no | no (faster with one) |
| Needs network | no | first run only (MoLFormer download) |
| Extra download | none — inputs are in `data/` | 12 GB of weights + 47 MB of backbones |
| Entry point | `reproduce.sh`, or the six `notebooks/` | `qadmet predict` / `code/predict_smiles.py` |
| Runtime | minutes | seconds per molecule per endpoint |
| Read next | [Axis A](#axis-a--reproduce-the-papers-figures-and-tables) | [Axis B](#axis-b--predict-new-molecules) |

Axis A works from frozen per-seed predictions, so it never touches a model. Axis B
loads the actual trained weights. A third, heavier path — retraining the models from
scratch on a GPU — is documented separately in [`code/TRAINING.md`](code/TRAINING.md).

---

## Axis A — reproduce the paper's figures and tables

No GPU, no network, no model weights. Every figure and table is recomputed from the
raw per-seed model predictions and leaderboard entries in `data/recompute_data/` —
there are no hard-coded summary numbers.

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, jupyter
bash reproduce.sh                   # one command, writes figures/ and tables/
```

`reproduce.sh` detects which ingredients are already present and runs only the stages
still needed. Since the shipped `data/recompute_data/` already contains the 10
prediction parquets, the leaderboards, and `all1023.csv`, it skips training and the
subset sweep and just runs the six notebooks — no GPU, no network, ~10 minutes.
`bash reproduce.sh -h` lists the flags for forcing a stage (`--figures-only`,
`--sweep`, `--train`).

Or run the notebooks directly. Regeneration is split into six of them — one for the
intro/methods tables, one per Results subsection. **Each runs independently** (each
carries the same setup cell that recomputes everything from source), so any order,
or just the one you care about:

| Notebook | Produces |
|---|---|
| `notebooks/00_pre_analysis.ipynb` | Setup + Table 1, Table S5, Table 2, Table S7 |
| `notebooks/31_complementarity.ipynb` (§3.1) | Figure 1, Table S1 |
| `notebooks/32_fusion_comparison.ipynb` (§3.2) | Figure 2, Figures S1–S3, Table S2, Figure 3, Table S3 |
| `notebooks/33_heldout_test.ipynb` (§3.3) | Table 3, Figures S6/S7, Table 4, Table S8, Figure 4 |
| `notebooks/34_gain_analysis.ipynb` (§3.4) | Figure 5, Table 5, Figures S4/S5, Table S4 |
| `notebooks/35_polaris_external.ipynb` (§3.5) | Figure 6, Table S6 |

```bash
for nb in notebooks/00_pre_analysis notebooks/3?_*; do
    jupyter nbconvert --to notebook --execute --inplace "${nb%.ipynb}.ipynb"
done
jupyter notebook notebooks/31_complementarity.ipynb   # or interactively
```

The shared setup cell recomputes every value — solo and ensemble norm_scores, the
uniform-mean ensemble, LOO/Shapley, inter-member correlations, the k-sweep, and the
Wilcoxon statistics — from the raw files, and `assert` blocks check each result
against the manuscript before anything is drawn (the k=5 composition, the 3-of-6
Polaris wins, and so on). Output lands in `figures/` and `tables/`.

### What gets produced

| Output | What it shows |
|---|---|
| Figure 1 | No single molecular encoder dominates (matched-seed solo test norm_score, 10 models) |
| Figure 2 | MAP joint fusion raises the floor but rarely tops the best single backbone; fusion-level contrast |
| Figure 3 | Ensemble-size sweep; diminishing marginal gain, elbow at k=5 |
| Figure 4 | Statistical validation: paired Wilcoxon forest plot of the ensemble gain vs strong single models and the per-task oracle |
| Figure 5 | Where the gain comes from: validation→test reliability over all 1,023 subsets (post-hoc) + per-member task-win counts |
| Figure 6 | LateFusion-k5 vs every public submission on the 6 Polaris ADME-Fang leaderboards |
| Figures S1–S7 | Per-backbone and aggregation heatmaps (S1–S3), inter-member correlation (S4), member roles by LOO and Shapley (S5), per-task box plots vs TDC leaderboard best (S6), leaderboard coverage vs mean rank (S7) |
| Tables 1–5 | Positioning vs prior fusion work; mechanism grouping; ensemble summary; per-task ensemble; k=5 member roles |
| Tables S1–S8 | Per-backbone detail; representation aggregation; k-sweep; inter-member correlation; reproducibility; Polaris datasets; dataset provenance; ensemble-vs-baseline significance |

---

## Axis B — predict new molecules

Loads the actual trained weights and runs the k=5 ensemble on arbitrary SMILES.

**Setup** (once): follow [`weights/README.md`](weights/README.md). Two things are
needed, because they carry different licences — the fine-tuned PEFT weights we
release (12 GB, on Zenodo) and the three pretrained backbones, which you download
from their original authors (CheMeleon 34 MB, GraphMVP 13 MB, MoLFormer fetched
automatically from Hugging Face).

```bash
qadmet predict --smiles "CCO" "CC(=O)Oc1ccccc1C(=O)O"
qadmet predict --input molecules.csv --output predictions.csv --device cuda
python code/predict_smiles.py --input molecules.csv     # same thing, no install needed
```

Output is one row per (molecule, endpoint):

```
smiles,task,task_type,tetrafuse,mtl_dmpnn,mtl_molformer,stl_chemeleon,stl_graphmvp,stl_map
CCO,Caco2_Wang,regression,-4.4715,-4.1637,-4.6159,-4.5362,-4.3084,-4.7334
```

`tetrafuse` is the k=5 uniform-mean ensemble — the model the paper reports. The five
member columns give each member's own seed-averaged prediction, so a surprising
ensemble value can be traced to the member behind it. Regression endpoints come back
in their original units (the target scaler and any `log1p` are inverted);
classification endpoints are probabilities.

All 22 TDC endpoints are predicted by default. `Bioavailability_Ma` is included
because the multitask members were trained on it, but it is excluded from every
benchmark result in the paper (its leaderboard ranking is unreliable).

### Is it really the published model?

Verified, not assumed. `code/verify_inference.py` rebuilds each member from the
released weights, predicts the TDC test split, and compares against the same
prediction parquets axis A consumes:

- four of five members agree to floating-point tolerance (≤1.2e-05)
- the **k=5 ensemble matches at 3 decimal places on every endpoint checked**
- MAP·STL carries a small documented residual (~1% of each endpoint's range)

This check caught two silent bugs that would otherwise have shipped — a
checkpoint-selection ambiguity and a fusion-temperature value absent from
`state_dict`. Both are written up, with the evidence, in
[`verification/inference_parity.md`](verification/inference_parity.md).

---

## Package layout

```
release/
├── README.md                     ← this file
├── reproduce.sh                  ← AXIS A: one-command driver (auto-detects what to skip)
├── CITATION.cff                  ← citation metadata
├── requirements.txt              ← minimal deps for the notebooks (axis A)
├── environment.yml               ← conda env for the notebooks
├── LICENSE                       ← MIT
│
├── data/                         ← AXIS A inputs (see DATA_MANIFEST.md)
│   ├── recompute_data/           ← raw source the notebooks recompute from:
│   │   ├── predictions/{model}.parquet   10 models; seed,task,split,smiles,y_true,y_pred
│   │   ├── leaderboards/{model}.csv      10 models; dataset,metric,ours,ours_std,rank,n_total,Top,Top_model
│   │   ├── all1023.csv                   1,023-subset uniform-mean sweep (code/make_all1023.py regenerates it)
│   │   ├── tdc_leaderboard_snapshot.json, task_registry.json, task_metadata.json
│   │   └── polaris_leaderboard_full.json Polaris public submissions (Figure 6 input)
│   └── polaris_external/         ← Polaris audit tables (not notebook inputs; see its README.md)
├── notebooks/                    ← AXIS A: six independent regeneration notebooks
├── figures/  tables/             ← AXIS A outputs (PNG / CSV)
│
├── weights/                      ← AXIS B: released PEFT weights (see its README.md)
│   ├── README.md                 ← backbone downloads + directory layout + scalers
│   ├── weights_manifest.csv      ← 340 files with sha256   ┐ tracked in git, so the
│   ├── target_scalers.json       ← regression scalers      │ layout and checksums are
│   ├── checkpoint_map.json       ← checkpoint provenance   ┘ readable from a clone
│   └── tetrafuse/<member>/<task>/seed_<n>/peft.pt   ← 12 GB, from Zenodo (gitignored)
├── verification/                 ← AXIS B: parity report vs the released predictions
│
└── code/
    ├── TRAINING.md               ← retraining from scratch (GPU): weights, splits, pipeline
    ├── predict_smiles.py         ← axis B standalone predictor
    ├── verify_inference.py       ← axis B parity check
    ├── export_weights.py         ← rebuild weights/ from training checkpoints
    ├── export_scalers.py         ← rebuild target_scalers.json
    ├── select_checkpoints.py     ← resolve ambiguous checkpoints by measurement
    ├── audit_recipes.py          ← check shipped recipes against what was trained
    ├── make_all1023.py           ← regenerate all1023.csv from the prediction parquets
    ├── make_tdc_submission.py    ← build TDC leaderboard submission scores
    ├── tdc_submission/           ← the resulting scores (21 benchmarks, evaluate_many format)
    ├── qadmet/                   ← the training/evaluation/inference engine
    │   ├── src/qadmet/           ← core, datasets, features, modules/backbones, ensembles,
    │   │                            evaluation, precompute, runners, tasks, training,
    │   │                            inference (axis B), cli
    │   ├── recipes/              ← 38 recipe YAMLs: train/ 15 (TDC-22: 5 MTL + 5 STL;
    │   │                            Polaris: 5), evaluate/ 10, _base/ 9, groupings/ 2, precompute/ 2
    │   └── pyproject.toml        ← CLI entry point qadmet.cli.main
    ├── polaris_fang_run_all.sh   ← end-to-end training run order
    └── data_polaris_adme_fang/   ← raw Polaris ADME-Fang CSVs (train/val + test) + summary
```

## Data sources

- **TDC** — Therapeutics Data Commons (<https://tdcommons.ai>), ADMET benchmark group.
  The 22 splits are downloaded at runtime by the `tdc` package, deterministically per seed.
- **Polaris ADME-Fang** — Biogen ADME-Fang benchmark on the Polaris Hub
  (<https://polarishub.io>), benchmarks `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1`.
  Our ensemble was submitted as `LateFusion-k5`. Test labels are withheld by the Hub,
  so those scores come from Hub-side evaluation.

## License

MIT (see `LICENSE`), including the `qadmet` engine under `code/qadmet/`. The
pretrained backbones are **not** covered by it — they are redistributed by their own
authors under their own terms, which is why axis B asks you to download them directly.

## Notes

- `data/` holds no model weights. Axis A needs none; axis B gets them from
  `weights/README.md`; retraining is covered by `code/TRAINING.md`.
- `rdkit2d`, `xtb`, and `unimol2` descriptor/embedding files are **not** required —
  none of the 10 paper models use them. The two joint-fusion MAP models use only the
  CheMeleon graph, MoLFormer tokens, and a GraphMVP encoder built on the fly from SMILES.
- The shipped training recipes are checked against the recipe each released run
  actually recorded (`code/audit_recipes.py`), so retraining from this repository
  reproduces the published configuration rather than a drifted one.
