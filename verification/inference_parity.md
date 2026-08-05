# Inference parity: released weights vs. released predictions

This note records whether the model-usage axis (`qadmet predict` /
`code/predict_smiles.py`) reproduces the predictions the paper's analysis is
built on. The test rebuilds each member from the released PEFT weights, predicts
the TDC held-out test split, and compares against
`data/recompute_data/predictions/<member>.parquet`.

Reproduce with:

```bash
python code/verify_inference.py --out verification/inference_parity.csv
# full 5-seed sweep:
python code/verify_inference.py --seeds 0 1 2 3 4 --tasks <all 22>
```

Raw results: [`inference_parity.csv`](inference_parity.csv) — 20 rows
(5 members × 4 endpoints, seed 0). The four endpoints span both task types and
include a log1p-transformed regression target (VDss_Lombardo), so the bundled
target scalers and the `expm1` inversion are exercised.

## Result

| Member | max abs. diff | max diff / prediction range | Pearson r |
|---|---|---|---|
| D-MPNN·MTL | 1.2e-07 – 1.4e-06 | ≤ 1.8e-07 | 1.0000000 |
| MoLFormer·MTL | 4.2e-07 – 2.5e-06 | ≤ 2.6e-06 | 1.0000000 |
| CheMeleon·STL | 5.4e-07 – 9.5e-06 | ≤ 1.3e-06 | 1.0000000 |
| GraphMVP·STL | 8.9e-07 – 1.2e-05 | ≤ 2.8e-06 | 1.0000000 |
| MAP·STL | 4.5e-03 – 1.1e-01 | 3.7e-03 – 1.4e-02 | ≥ 0.9999872 |

Four of the five members agree with the released predictions to floating-point
tolerance. Predictions on new molecules therefore come from the same models that
produced the published numbers.

## Two bugs this check caught

Both were silent — the weights loaded cleanly and the predictions looked
plausible — and neither would have been found without comparing against the
released parquets.

**1. Checkpoint selection (CheMeleon·STL).** `ModelCheckpoint(filename="best")`
writes `best-v1.ckpt` rather than overwriting when the name already exists, and
all 110 CheMeleon·STL seed directories hold both files (the other four members
hold only `best.ckpt`). Exporting `best.ckpt` gave errors of 3.7e-02 – 1.6e-01.
Neither the filename nor the recorded `best_model_score` identifies the right
one — the scores order the two files inconsistently across tasks — so
`code/select_checkpoints.py` decides it by measurement, scoring every candidate
against the released predictions. `best-v1.ckpt` won 110/110. The released files
are all named `peft.pt`; `meta["source_ckpt"]` and the `source_ckpt` column of
`weights/weights_manifest.csv` record which checkpoint each came from.

**2. Fusion temperature (MAP·STL).** `RobustAttentionFusion.tau` is a plain
Python float, so it is absent from `state_dict` and a freshly built module starts
at `tau_init = 3.0`. Training anneals it to `tau_min` and `trainer.predict` runs
after training ends, so the published predictions were made at `tau_min = 0.5`
(confirmed from the CSV logger's `train/fusion_tau`, which reaches 0.5 well
before the final epoch). Leaving tau at its initial value left the gate softmax
far too smooth: max abs. error 4.0e-01. Restoring `tau_min` at load time
(`_restore_fusion_temperature`) reduced it to 1.2e-02.

## Documented residual: MAP·STL

MAP·STL still differs by 4.5e-03 – 1.1e-01 in absolute terms — worst case
1.062e-01 on VDss_Lombardo, whose predictions span roughly 0 – 17, i.e.
3.7e-03 – 1.4e-02 of each endpoint's prediction range (r ≥ 0.99999). The
following were each measured and ruled out:

| Candidate cause | Evidence against |
|---|---|
| Wrong or corrupted weights | All 704 model tensors match the source checkpoint exactly (max diff 0.0), including the 207 base MoLFormer tensors and all 16 fusion tensors |
| Our data pipeline | The training loader (`prepare_fold_loaders`) and our inference loader agree with each other to 2.4e-07 — and both differ from the release by the same amount |
| MoLFormer tokenization | On-the-fly tokenization matches the precomputed `molformer_tokens.parquet` exactly (max diff 0.0) |
| A different training run | The run named by `artifacts/baselines/stl_map/baseline/manifest.yaml` (`a57008843b9b`) is the one used; its own `predictions.parquet` is md5-identical to the released file. The only other run of this recipe is further off (8.5e-01) |
| Code drift since training | The training manifest's `git_sha` (`dde87f96`) is the current HEAD, with no changes to `modules/fusion`, `modules/backbones`, or `datasets/loaders.py` |
| Mixed precision (`bf16-mixed`) | Predicting through a Lightning `Trainer` at the recipe's own `precision="bf16-mixed"` — the same call the runner makes — leaves the gap unchanged (1.16e-02 at `32-true`, 1.25e-02 at `bf16-mixed`). The released values are also *not* on the bf16 grid: of the 30 HIA_Hou probabilities that are not exactly 1.0, zero are exact bf16 values |
| Storing predictions in bf16 | Same evidence as above. (An earlier reading of "74% of HIA_Hou probabilities are exact bf16 values" was an artifact of the 87 probabilities that are exactly 1.0, which are representable in any format) |
| GPU vs CPU | Identical to the digit |
| Batch size | 16 / 32 / 182 give identical results |
| A different tau value | A fine scan over 0.46–0.54 leaves 0.5 with the lowest mean error |
| Missing per-sample presence mask | The training batch's `modality_present` is all-True for these tasks, so the mask does not alter the gate; and our batches agree with the training loader's to 2.4e-07 regardless |

All 704 model tensors are bit-identical to the checkpoint and the inputs match
the training loader, so the cause lies in some piece of prediction-time state
that `state_dict` does not carry — the same class of problem as the tau bug, and
`RobustAttentionFusion` is the only member with such state. The specific
remaining candidate is the runtime configuration under which the released
`predictions.parquet` was written (e.g. device count during `trainer.predict`).

### What the residual costs in practice

Recomputing each endpoint's TDC metric from our predictions versus the released
ones (seed 0):

| Endpoint | Metric | Released | Ours | Δ | At 3 dp |
|---|---|---|---|---|---|
| Caco2_Wang | MAE | 0.414384 | 0.414463 | 7.9e-05 | 0.414 = 0.414 |
| VDss_Lombardo | Spearman | 0.700291 | 0.701037 | 7.5e-04 | 0.700 vs 0.701 |
| HIA_Hou | AUROC | 0.957202 | 0.956173 | 1.0e-03 | 0.957 vs 0.956 |
| BBB_Martins | AUROC | 0.891143 | 0.891104 | 3.9e-05 | 0.891 = 0.891 |

Metric differences are 4e-05 – 1e-03. At the 3 decimals the paper reports, two of
the four endpoints are identical (Caco2_Wang, BBB_Martins) and two differ in the
last digit (VDss_Lombardo 0.700 vs 0.701, HIA_Hou 0.957 vs 0.956). No reported
conclusion turns on a difference of that size, but
MAP predictions from this release should be treated as reproducing the published
model to about 1% of each endpoint's range (up to ~0.11 in VDss_Lombardo's own
units) rather than to floating-point tolerance. The other four members — and so
4/5 of the ensemble — are exact.
