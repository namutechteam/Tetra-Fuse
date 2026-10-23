# Tetra-Fuse: Encoder Heterogeneity Improves ADMET Prediction at Matched Ensemble Size

Code, data and weights for the paper *"Tetra-Fuse: Encoder Heterogeneity Improves ADMET
Prediction at Matched Ensemble Size"* (An, Han, Lee, Yeom).

Combining several molecular encoders usually also changes how many models are averaged, at
what level they are fused and whether scores are averaged over runs. The paper separates
these effects on 21 TDC ADMET tasks, using the per-molecule predictions of 10 base models
(four encoders — CheMeleon, MoLFormer, GraphMVP, D-MPNN — and the feature-fusion model MAP,
each single- and multi-task) over five independent runs. At equal member count, ensembles of
different encoders outperform multi-run ensembles of one encoder, including the strongest
same-encoder ensemble chosen on validation data; once member count is matched, prediction-
and feature-level fusion do not differ detectably. The validation-selected five-member
uniform mean (**Tetra-Fuse**) is also evaluated on the independent Polaris ADME-Fang
benchmark (6 regression endpoints; submitted to the Polaris Hub as `LateFusion-k5`).

An earlier version of this work was posted as ChemRxiv 10.26434/chemrxiv.15007185/v1
(release 1.1.0 of this repository). Release 2.0.0 adds the controlled comparisons
(notebook 36, `code/followup/`) and renumbers figures and tables to the current manuscript;
see *Figure and table numbering* below.

---

## Which axis do you want?

This release serves two different needs, and they share almost nothing in terms of
setup cost. Pick one:

| | **Axis A — reproduce the paper** | **Axis B — predict new molecules** |
|---|---|---|
| Question | "Do the figures and tables come out of the data?" | "What does Tetra-Fuse say about *my* compound?" |
| Needs a GPU | no | no (faster with one) |
| Needs network | notebook 36 only: TDC SMILES, once | first run only (MoLFormer download) |
| Extra download | none — inputs are in `data/` | 12 GB of weights + 47 MB of backbones |
| Entry point | `reproduce.sh`, or the seven `notebooks/` | `qadmet predict` / `code/predict_smiles.py` |
| Runtime | minutes | seconds per molecule per endpoint |
| Read next | [Axis A](#axis-a--reproduce-the-papers-figures-and-tables) | [Axis B](#axis-b--predict-new-molecules) |

Axis A works from frozen per-seed predictions, so it never touches a model. Axis B
loads the actual trained weights. A third, heavier path — retraining the models from
scratch on a GPU — is documented separately in [`code/TRAINING.md`](code/TRAINING.md).

---

## Axis A — reproduce the paper's figures and tables

No GPU and no model weights. Every figure and table is recomputed from the raw per-seed
model predictions and leaderboard entries in `data/recompute_data/` — there are no
hard-coded summary numbers.

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, jupyter, rdkit, PyTDC
bash reproduce.sh                   # one command, writes figures/ and tables/
```

`reproduce.sh` detects which ingredients are already present and runs only the stages
still needed. Since the shipped `data/recompute_data/` already contains the 10
prediction parquets, the leaderboards, and `all1023.csv`, it skips training and the
subset sweep and just runs the seven notebooks — no GPU, ~20 minutes on CPU.
`bash reproduce.sh -h` lists the flags for forcing a stage (`--figures-only`,
`--sweep`, `--train`).

Notebook 36 has one external input: the TDC ADMET training/validation SMILES, used to
place each test molecule in a similarity quintile (Note S3.4, S3.6). It reads them from
`$TDC_ADMET_GROUP` or `data/tdc_admet_group/` if present and otherwise downloads them once
with PyTDC. Everything else runs offline.

Or run the notebooks directly. **Each runs independently** (notebooks 00 and 31–35 carry
the same setup cell that recomputes everything from source; notebook 36 uses
`code/followup/`), so any order, or just the one you care about:

| Notebook | Produces |
|---|---|
| `notebooks/00_pre_analysis.ipynb` | Setup + Table 1, Table S5, Table S11, Table S7 |
| `notebooks/31_complementarity.ipynb` (§3.1) | Figure 1, Table S1 |
| `notebooks/36_controlled_comparisons.ipynb` (§2.7, §3.2–3.4) | Figure 2, Table 2, Tables S10a–c (incl. STL-only and MTL-only pools), Figures S7–S9, Supplementary Note S3 values |
| `notebooks/32_fusion_comparison.ipynb` (§3.3–3.4) | Figure 3 (fusion level at matched member count), Figures S1–S3, Table S2, Table S3 |
| `notebooks/33_heldout_test.ipynb` (§3.4) | Table 3 (per-task, with summary rows), Figures S4/S5, Table S8, Figure 4, Table S9 |
| `notebooks/34_gain_analysis.ipynb` (§3.4–3.5) | Figure 5, Table S4, Figure S6 |
| `notebooks/35_polaris_external.ipynb` (§3.6) | Figure 6, Table S6 |

```bash
for nb in notebooks/00_pre_analysis notebooks/3?_*; do
    jupyter nbconvert --to notebook --execute --inplace "${nb%.ipynb}.ipynb"
done
jupyter notebook notebooks/36_controlled_comparisons.ipynb   # or interactively
```

The shared setup cell recomputes every value — solo and ensemble norm_scores, the
uniform-mean ensemble, LOO/Shapley, inter-member correlations, the k-sweep, and the
Wilcoxon statistics — from the raw files, and `assert` blocks check each result
against the manuscript before anything is drawn (the k=5 composition, the 3-of-6
Polaris wins, and so on). Notebook 36 does the same for every number of Section 3.2,
Table 3, Table S10 and Note S3. Output lands in `figures/` and `tables/`; intermediate
tables of notebook 36 go to `results/controlled_comparisons/`.

### Controlled comparisons (`code/followup/`)

| File | Role |
|---|---|
| `tfcc.py` | Scoring in the manuscript convention (raw metric per realization → mean → normalize once), the member-count × encoder-heterogeneity grid S(m, d), validation-only selection of same-encoder and same-model multi-run controls, the fusion-level comparison at matched member count, the seed-averaging effect, Table 3 and Table S10 |
| `figs.py` | Figure 2 and Figure 3 |
| `analysis_B_decomposition.py` | Note S3.3, Figure S7 and Figure 5C: specialization vs beyond-best-member decomposition of the loss reduction |
| `analysis_C_similarity.py` | Note S3.4, Figure S8: gain by similarity of test molecules to the training data |
| `analysis_2_uncertainty.py` | Note S3.6, Figure S9: member disagreement as an error ranker, heterogeneous vs same-encoder ensembles |
| `tf_common.py` | Shared loaders and the TDC metric / leaderboard normalization (same logic as `code/make_all1023.py`) |

Intervals are task-bootstrap 95% percentile intervals (2,000 resamples of the 21 tasks,
seed 20261002).

### What gets produced

| Output | What it shows |
|---|---|
| Figure 1 | No single base model is best across ADMET endpoints (matched-seed solo test norm_score, 10 models) |
| Figure 2 | At equal member count, ensembles of different encoders outperform multi-run ensembles of one encoder, against average, validation-selected and oracle controls |
| Figure 3 | Fusion level at matched member count: MAP, Late-3, MAP×3, MAP×5 and Tetra-Fuse |
| Figure 4 | Tetra-Fuse vs deployable single models: paired Wilcoxon forest plot |
| Figure 5 | Validation→test transfer over all 1,023 subsets, endpoint-specific member strengths and the loss decomposition (S, O, S − O) |
| Figure 6 | LateFusion-k5 vs every public submission on the 6 Polaris ADME-Fang leaderboards |
| Figures S1–S8 | Per-backbone and aggregation heatmaps (S1–S3), inter-member correlation (S4), member roles by LOO and Shapley (S5), per-task box plots vs TDC leaderboard best (S6), leaderboard coverage vs mean rank (S7), same-encoder five-run control (S8) |
| Figures S9–S12 | Error decomposition (S9), gain by similarity quintile (S10), uncertainty quality (S11), MAP vs its constituent encoders (S12) |
| Tables 1–3 | Comparison with prior fusion and ensemble studies; effects of model combination (Table 2); per-task results with summary rows (Table 3) |
| Tables S1–S10 | Per-backbone detail; representation aggregation; k-sweep; inter-member correlation; reproducibility; Polaris datasets; dataset provenance; ensemble-vs-baseline significance; same-encoder five-run control; controlled comparisons (S10a per-task, S10b contrasts incl. STL-only/MTL-only pools, S10c S(m, d) grid); mechanism-informed task grouping (S11) |

### Figure and table numbering

Release 1.1.0 (the ChemRxiv version) numbered the figures differently. Mapping from the
1.1.0 files to this release: Figure 2A, B → Figure S12 (redrawn on the Table S2 basis),
Figure 2C → Figure 3 (redrawn at matched member count), Figure 3 → 4, Figure 4 → 5,
Figure 5 → 6, Figure 6 → 7; Table 2 → Table S11; Tables 3 and 4 → Table 3 (summary rows
appended to the per-task table); Table 5 → Table 4. Figure 2, Table 2, Figures S9–S11 and
Table S10 are new, and Table 1 was rewritten to the current manuscript. Ensemble norm_scores
entering the significance tests are now compared with the baselines at full precision
(Table S8, Fig. 5, Table S9).

Manuscript v3-1 renumbering (this commit): Figure 4 → Figure S3; Figures 5–7 → 4–6, with Figure 5 gaining panel C (loss decomposition); Table 4 → Table S4 (the former correlation summary Table S4 is now `tables/extra_inter_model_corr.csv`); Figures S12, S2, S6, S7, S4, S9, S10 and S11 → Figures S1–S9 in that order. The former Figures S1, S3, S5 and S8 duplicate Table S2, Figure S2, Table S4 and Figure 2C/Table S9 and are still written, as `figures/extra_*.png`. Figure 1B now shows the four encoder values as points.

---

## Axis B — predict new molecules

Loads the actual trained weights and runs the k=5 ensemble on arbitrary SMILES.

**Setup** (once): follow [`weights/README.md`](weights/README.md). Two things are
needed, because they carry different licences — the fine-tuned PEFT weights we
release (12 GB, on Zenodo: [10.5281/zenodo.21802319](https://doi.org/10.5281/zenodo.21802319))
and the three pretrained backbones, which you download
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
├── LICENSE                       ← Apache-2.0
│
├── data/                         ← AXIS A inputs (see DATA_MANIFEST.md)
│   ├── recompute_data/           ← raw source the notebooks recompute from:
│   │   ├── predictions/{model}.parquet   10 models; seed,task,split,smiles,y_true,y_pred
│   │   ├── leaderboards/{model}.csv      10 models; dataset,metric,ours,ours_std,rank,n_total,Top,Top_model
│   │   ├── all1023.csv                   1,023-subset uniform-mean sweep (code/make_all1023.py regenerates it)
│   │   ├── tdc_leaderboard_snapshot.json, task_registry.json, task_metadata.json
│   │   └── polaris_leaderboard_full.json Polaris public submissions (Figure 6 input)
│   └── polaris_external/         ← Polaris audit tables (not notebook inputs; see its README.md)
├── notebooks/                    ← AXIS A: seven independent regeneration notebooks
├── figures/  tables/             ← AXIS A outputs (PNG / CSV)
├── results/controlled_comparisons/ ← AXIS A: intermediate tables of notebook 36
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
    ├── followup/                 ← AXIS A: controlled comparisons (notebook 36, Figure 3)
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

Apache-2.0 (see `LICENSE`), including the `qadmet` engine under `code/qadmet/`. The
pretrained backbones are **not** covered by it — they are redistributed by their own
authors under their own terms, which is why axis B asks you to download them directly.

## Notes

- `data/` holds no model weights. Axis A needs none; axis B gets them from
  `weights/README.md`; retraining is covered by `code/TRAINING.md`.
- `rdkit2d`, `xtb`, and `unimol2` descriptor/embedding files are **not** required —
  none of the 10 paper models use them. The two joint-fusion MAP models use only the
  CheMeleon graph, MoLFormer tokens, and a GraphMVP encoder built on the fly from SMILES.
- **GraphMVP chirality embedding.** The 2021 GraphMVP checkpoint was trained with a
  4-entry chirality vocabulary; current OGB featurizers use 5 (they append `misc`). The
  backbone therefore cannot load `atom_encoder.atom_embedding_list.1.weight` from the
  checkpoint. In every published GraphMVP and MAP model this (4→5, 300) table kept its
  random initialization and stayed frozen; the trained table is stored in the released
  PEFT weights and restored on loading, so predictions are unaffected
  (`code/verify_inference.py`). The backbone option `vocab_fill` controls the behaviour:
  `"random"` (default, the published setting) or `"pretrained_rows"`, which copies the four
  checkpoint rows and initializes only the extra row — recommended for new training. On
  the 46,202 TDC ADMET molecules, OGB chirality indices 3 and 4 never occur, so with
  `"pretrained_rows"` every chirality row actually used comes from the checkpoint.
- The shipped training recipes are checked against the recipe each released run
  actually recorded (`code/audit_recipes.py`), so retraining from this repository
  reproduces the published configuration rather than a drifted one.
