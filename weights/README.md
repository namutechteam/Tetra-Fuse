# Tetra-Fuse released weights

This directory holds the weights needed to **predict new molecules** with
Tetra-Fuse (axis B of the release). To reproduce the paper's figures and tables
from the shipped prediction parquets you need none of this — see the top-level
`README.md`.

Two pieces have to come together, because they have different licences:

| Piece | Ships here? | Where it comes from |
|---|---|---|
| Fine-tuned PEFT weights (LoRA, AdapterGNN, fusion, heads, target scalers) | **yes** — Zenodo archive | this release |
| Pretrained backbones (CheMeleon, GraphMVP, MoLFormer) | **no** | download from the original authors |

Everything we trained is released; the frozen backbones are redistributed by
their own authors under their own terms, so you fetch those directly.

---

## 1. Download the pretrained backbones

Place them under `code/qadmet/data/weights/` exactly as shown. The recipes name
these paths, so the names matter.

```
code/qadmet/data/weights/
├── chemeleon_mp.pt                 34 MB   sha256 c376624d3407204e…
└── graphmvp/
    └── graphmvp_hybrid.pth         13 MB   sha256 077fc1d36822ef3f…
```

**CheMeleon** — `chemeleon_mp.pt`
- Zenodo: <https://zenodo.org/records/15460715>
- Source: <https://github.com/JacksonBurns/chemeleon>
- Save the checkpoint unchanged. Our loader reads its `hyper_parameters` and
  `state_dict` keys directly.

**GraphMVP** — `graphmvp_hybrid.pth`
- Google Drive: <https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6>
- Source: <https://github.com/chao1224/GraphMVP>
- Take `pretraining_model/GraphMVP_hybrid/` — the **regression** variant trained
  on OGB complete atom features, not the chirality-classification variant.
- Save the file **as downloaded**. It is a dict of several sub-models
  (`model`, `model_3D`, …); our loader extracts the `model` key and shape-filters
  it itself, so do not pre-extract anything.

**MoLFormer** — downloaded automatically
- Hugging Face `ibm/MoLFormer-XL-both-10pct` (~180 MB cached on first use).
  Needs network access on the first prediction; `trust_remote_code=True` is
  required by that repository and our loaders pass it.
- The loaders do **not** pin a revision — they resolve `main` at download time.
  The runs behind this release used commit `361063d0ad524ef77cf39b08469f6be770dc550f`.
  If you need bit-exact reproduction and upstream `main` has since moved, fetch that
  commit explicitly:

  ```bash
  huggingface-cli download ibm/MoLFormer-XL-both-10pct \
      --revision 361063d0ad524ef77cf39b08469f6be770dc550f
  ```

  In practice this matters only for the frozen encoder's weights; the fine-tuned
  parameters we release are unaffected.

A note on GraphMVP: loading it prints a warning that
`atom_encoder.atom_embedding_list.1.weight` is skipped for a shape mismatch
(4×300 in the checkpoint, 5×300 in our featurization) and randomly initialised.
This is expected — that row is then restored from our released weights, which
were trained with it. The warning is not a sign of a broken download.

---

## 2. Unpack the PEFT weights

The Zenodo record (DOI in the top-level `CITATION.cff`) holds one archive per
ensemble member, so you can take only the members you need:

| Archive | Size | Files |
|---|---|---|
| `tetrafuse_weights_mtl_dmpnn.tar` | 10 MB | 5 |
| `tetrafuse_weights_mtl_molformer.tar` | 25 MB | 5 |
| `tetrafuse_weights_stl_chemeleon.tar` | 4.0 GB | 110 |
| `tetrafuse_weights_stl_graphmvp.tar` | 240 MB | 110 |
| `tetrafuse_weights_stl_map.tar` | 7.0 GB | 110 |

Predicting with the full k=5 ensemble needs all five. The archives are
uncompressed `.tar` — the payload is dense fp32 tensors, where gzip buys about 8%
at roughly 30× the wall time.

```bash
sha256sum -c SHA256SUMS                        # verify the downloads first
for t in tetrafuse_weights_*.tar; do tar -xf "$t" -C weights/; done
```

Paths inside each archive begin with `tetrafuse/`, so extracting into `weights/`
gives exactly:

```
weights/
├── README.md                      ← this file
├── weights_manifest.csv           one row per file, with sha256
├── target_scalers.json            regression target scalers (see below)
├── checkpoint_map.json            which training checkpoint each file came from
└── tetrafuse/
    ├── mtl_dmpnn/_multitask/seed_{0..4}/peft.pt        5 files    9.9 MB
    ├── mtl_molformer/_multitask/seed_{0..4}/peft.pt    5 files     24 MB
    ├── stl_chemeleon/<task>/seed_{0..4}/peft.pt      110 files    4.1 GB
    ├── stl_graphmvp/<task>/seed_{0..4}/peft.pt       110 files    230 MB
    └── stl_map/<task>/seed_{0..4}/peft.pt            110 files    7.1 GB
```

Total 340 files, 12 GB. The two multitask (MTL) members carry one model per seed
covering all endpoints; the three single-task (STL) members carry one model per
(endpoint, seed) — 22 endpoints × 5 seeds. `seed_<n>` is the TDC fold seed `n`,
the value the training runner passes to `load_fold`.

`<task>` is one of the 22 TDC endpoint names:

```
AMES  BBB_Martins  Bioavailability_Ma  Caco2_Wang  Clearance_Hepatocyte_AZ
Clearance_Microsome_AZ  CYP2C9_Substrate_CarbonMangels  CYP2C9_Veith
CYP2D6_Substrate_CarbonMangels  CYP2D6_Veith  CYP3A4_Substrate_CarbonMangels
CYP3A4_Veith  DILI  Half_Life_Obach  hERG  HIA_Hou  LD50_Zhu
Lipophilicity_AstraZeneca  Pgp_Broccatelli  PPBR_AZ  Solubility_AqSolDB
VDss_Lombardo
```

`Bioavailability_Ma` is included because the MTL members were trained on it, but
it is excluded from every benchmark and ensemble result in the paper (its
leaderboard ranking is unreliable — see the manuscript, Section 2.2).

Verify an unpacked tree against the manifest:

```bash
cd weights && awk -F, 'NR>1 {gsub(/\r/,""); print $9"  "$4}' weights_manifest.csv \
  | sha256sum -c - | grep -v ': OK$'      # prints nothing when every file matches
```

### What is inside a `peft.pt`

```python
{"state_dict": {...},                    # only the tensors we trained
 "meta": {"member": "stl_map", "task": "Caco2_Wang", "seed": 0,
          "recipe": "stl_4modality_robust_nounimol2", "run_id": "a57008843b9b",
          "pretrained": ["chemeleon", "molformer", "graphmvp"],
          "source_ckpt": "best.ckpt", "source_ckpt_sha256": "2e2d487a…",
          "n_tensors": 282, "n_params": 17136564, "dtype": "fp32"}}
```

Frozen backbone tensors are **excluded** — which is why you download the
backbones separately, and why the archive is far smaller than the raw training
checkpoints (12 GB versus ~78 GB). Which tensors were trained was decided by
measurement, not by name patterns: tensors that differ between seed 0 and seed 1
are learned; those identical across seeds and matching a declared pretrained
backbone in both key *and* shape are frozen. (The shape test matters — D-MPNN
names its message-passing tensors identically to CheMeleon but with different
dimensions.)

`meta["source_ckpt"]` records which training checkpoint the weights came from.
Released files are uniformly named `peft.pt`; the provenance lives inside the
file and in the manifest's `source_ckpt` column. For CheMeleon·STL this reads
`best-v1.ckpt` — Lightning wrote a second checkpoint rather than overwriting the
first, and the second is the one that reproduces the published predictions. See
`verification/inference_parity.md`.

### `target_scalers.json`

Regression heads were trained on standardised targets, so raw model output has to
be inverted to get real units. Lightning does not persist the scaler, so we
refit it from the training split exactly as the dataloader does and ship the
parameters:

```json
{"stl_chemeleon": {"Caco2_Wang": {"0": {"mean": -5.206575701434837,
                                        "scale": 0.7980829285610342,
                                        "log1p": false}}}}
```

9 regression endpoints × 5 seeds × 5 members. The inference code applies
`y = output × scale + mean`, then `expm1` where `log1p` is true. Refitting was
verified against the live training pipeline: with a float32 fit (chemprop stores
targets as float32) the STL scalers match to exactly 0, and MTL to 3.6e-15.
Regenerate with `python code/export_scalers.py --out weights/`.
Classification endpoints need no scaler — the head output is a probability.

---

## 3. Predict

```bash
# whole ensemble, all 22 endpoints
qadmet predict --smiles "CCO" "CC(=O)Oc1ccccc1C(=O)O"

# or without installing the CLI
python code/predict_smiles.py --input molecules.csv --output predictions.csv

# a subset, on GPU
qadmet predict --input molecules.csv --tasks Caco2_Wang HIA_Hou --device cuda
```

Output has one row per (molecule, endpoint): a `tetrafuse` column — the k=5
uniform-mean ensemble, which is the model the paper reports — plus one column per
member holding that member's own seed-averaged prediction, so an unexpected
ensemble value can be traced to the member behind it.

Defaults average all five seeds, which loads 5 models per endpoint per member.
Pass `--seeds 0` for a quick single-seed run at some cost in accuracy; note that
single-seed output is *not* the paper's model.

### Does this reproduce the published predictions?

Yes, for four of the five members, to floating-point tolerance (≤1.2e-05); the
k=5 ensemble itself matches at 3 decimal places on every endpoint checked.
MAP·STL carries a small documented residual. Full numbers, and the two bugs this
check caught, are in `verification/inference_parity.md`; re-run it yourself with
`python code/verify_inference.py`.
