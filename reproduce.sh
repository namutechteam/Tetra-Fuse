#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — one-command, end-to-end reproduction driver for Tetra-Fuse.
#
# Just run it:
#
#     bash reproduce.sh
#
# It auto-detects which ingredients are already present and runs only the
# stages that are still needed, then regenerates every figure and table.
#
#   * If data/recompute_data/all1023.csv AND the 10 predictions/leaderboards
#     are present  ->  it SKIPS training and the sweep, and only runs the
#     six notebooks (figures + tables).  No GPU, no network, ~10 min.
#
#   * If the 10 predictions are present but all1023.csv is missing/stale
#     ->  it regenerates all1023.csv from the predictions (make_all1023.py,
#     ~40 min, CPU only) and then runs the notebooks.
#
#   * If the predictions are NOT present  ->  it trains all 10 base models
#     from scratch (needs a GPU and the pretrained backbone weights, see
#     "BACKBONE WEIGHTS" below), copies their outputs into recompute_data/,
#     regenerates all1023.csv, and runs the notebooks.
#
# You can also force a level explicitly:
#     bash reproduce.sh --figures-only     # notebooks only (assume inputs exist)
#     bash reproduce.sh --sweep            # regenerate all1023.csv + notebooks
#     bash reproduce.sh --train            # full train -> evaluate -> sweep -> notebooks
#
# -----------------------------------------------------------------------------
# BACKBONE WEIGHTS (only needed for --train / from-scratch)
# -----------------------------------------------------------------------------
# Two pretrained backbones are NOT shipped (license). Download them ONCE into:
#
#     code/qadmet/data/weights/chemeleon_mp.pt
#         from  https://zenodo.org/records/15460715
#     code/qadmet/data/weights/graphmvp/graphmvp_hybrid.pth
#         the GraphMVP_hybrid (OGB-feature) checkpoint, saved raw, from
#         https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6
#
# MoLFormer-XL is fetched automatically from HuggingFace; D-MPNN needs no weights.
# If a weight file is missing, --train aborts with a message pointing here.
#
# -----------------------------------------------------------------------------
# BRING YOUR OWN parquet / csv (skip training, skip the sweep)
# -----------------------------------------------------------------------------
# To skip training: drop your 10 trained-model outputs here (filenames must be
# the model keys below), then run `bash reproduce.sh` (auto-detects -> figures):
#
#     data/recompute_data/predictions/<model>.parquet     # cols: seed,task,split,smiles,y_true,y_pred
#     data/recompute_data/leaderboards/<model>.csv        # cols: dataset,metric,ours,ours_std,rank,n_total,Top,Top_model
#
#   model keys: mtl_dmpnn mtl_chemeleon mtl_graphmvp mtl_molformer mtl_map
#               stl_dmpnn stl_chemeleon stl_graphmvp stl_molformer stl_map
#
# To ALSO skip the sweep: drop your sweep result here as well, then run it:
#
#     data/recompute_data/all1023.csv                     # cols: combo_key,k,val_within,val_cross,test_within,test_cross,val_comb,test_comb
#
# (All three are already shipped in this repo, so a fresh clone runs figures-only
#  out of the box.)
# =============================================================================
set -euo pipefail

# --- locate the release root (this script lives at the repo root) ------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA="data/recompute_data"
PRED="$DATA/predictions"
LB="$DATA/leaderboards"
NB="notebooks"
QCORE="code/qadmet"
PYBIN="${PYTHON:-python}"          # override with PYTHON=/path/to/python
NBTIMEOUT="${NBTIMEOUT:-1800}"

MODELS=(mtl_dmpnn mtl_chemeleon mtl_graphmvp mtl_molformer mtl_map \
        stl_dmpnn stl_chemeleon stl_graphmvp stl_molformer stl_map)

# TDC-22 train / evaluate recipes, indexed by model key (used only by --train)
declare -A TRAIN_RECIPE=(
  [mtl_dmpnn]=mtl_dmpnn_robust_causal6_kendall_epoch100
  [mtl_chemeleon]=mtl_chemeleon_robust_causal6_kendall_epoch100
  [mtl_graphmvp]=mtl_graphmvp_robust_causal6_kendall_epoch100
  [mtl_molformer]=mtl_molformer_robust_causal6_kendall_epoch100
  [mtl_map]=mtl_4modality_robust_causal6_nounimol2_epoch100
  [stl_dmpnn]=stl_dmpnn_paper_gradclip_seed01234_epoch100
  [stl_chemeleon]=stl_chemeleon_adaptivegnn_paper
  [stl_graphmvp]=stl_graphmvp_adaptivegnn_paper
  [stl_molformer]=stl_molformer_lora
  [stl_map]=stl_4modality_robust_nounimol2_seed01234
)
declare -A EVAL_RECIPE=(
  [mtl_dmpnn]=leaderboard_mtl_dmpnn_epoch100_s01234
  [mtl_chemeleon]=leaderboard_chemeleon_robust_causal6_kendall
  [mtl_graphmvp]=leaderboard_graphmvp_robust_causal6_kendall
  [mtl_molformer]=leaderboard_molformer_robust_causal6_kendall
  [mtl_map]=leaderboard_4modality_robust_causal6_nounimol2
  [stl_dmpnn]=leaderboard_stl_dmpnn_seed01234
  [stl_chemeleon]=leaderboard_stl_chemeleon
  [stl_graphmvp]=leaderboard_stl_graphmvp_seed01234
  [stl_molformer]=leaderboard_stl_molformer_seed01234
  [stl_map]=leaderboard_stl_4modality_robust_nounimol2_seed01234
)

say(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
have_predictions(){ for m in "${MODELS[@]}"; do [ -s "$PRED/$m.parquet" ] && [ -s "$LB/$m.csv" ] || return 1; done; return 0; }
have_sweep(){ [ -s "$DATA/all1023.csv" ]; }

# --- decide the level --------------------------------------------------------
MODE="auto"
case "${1:-}" in
  --figures-only) MODE="figures" ;;
  --sweep)        MODE="sweep" ;;
  --train)        MODE="train" ;;
  "" )            MODE="auto" ;;
  -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1  (use --figures-only | --sweep | --train | -h)"; exit 2 ;;
esac

if [ "$MODE" = "auto" ]; then
  if   have_predictions && have_sweep; then MODE="figures"
  elif have_predictions;               then MODE="sweep"
  else                                      MODE="train"; fi
  say "auto-detected level: $MODE"
fi

# --- STAGE 1: train (only for --train / no predictions present) --------------
if [ "$MODE" = "train" ]; then
  say "STAGE 1/4 — train 10 base models from scratch (GPU required)"
  CW="$QCORE/data/weights"
  MISS=0
  [ -s "$CW/chemeleon_mp.pt" ]            || { echo "MISSING: $CW/chemeleon_mp.pt"; MISS=1; }
  [ -s "$CW/graphmvp/graphmvp_hybrid.pth" ] || { echo "MISSING: $CW/graphmvp/graphmvp_hybrid.pth"; MISS=1; }
  if [ "$MISS" = 1 ]; then
    echo
    echo "Pretrained backbone weights are not shipped. Download them into the paths above:"
    echo "  CheMeleon : https://zenodo.org/records/15460715"
    echo "  GraphMVP  : https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6"
    echo "              (use pretraining_model/GraphMVP_hybrid/, save the raw checkpoint unchanged)"
    echo "Then re-run:  bash reproduce.sh --train"
    exit 1
  fi
  ( cd "$QCORE" && $PYBIN -m pip install -e . -q )
  ( cd "$QCORE" && $PYBIN -m qadmet.cli.main precompute --recipe recipes/precompute/molformer_tokens.yaml )
  for m in "${MODELS[@]}"; do
    say "train $m"
    ( cd "$QCORE" && $PYBIN -m qadmet.cli.main train    --recipe "recipes/train/${TRAIN_RECIPE[$m]}.yaml" )
    say "evaluate $m"
    ( cd "$QCORE" && $PYBIN -m qadmet.cli.main evaluate --recipe "recipes/evaluate/${EVAL_RECIPE[$m]}.yaml" )
    # copy this model's outputs into recompute_data/ (public baseline layout)
    SRC="$QCORE/artifacts/baselines/$m/baseline"
    mkdir -p "$PRED" "$LB"
    cp "$SRC/predictions.parquet" "$PRED/$m.parquet"
    cp "$SRC/leaderboard.csv"     "$LB/$m.csv"
  done
  MODE="sweep"   # fall through to regenerate the sweep from fresh predictions
fi

# --- STAGE 2: ensemble-size sweep (regenerate all1023.csv) --------------------
if [ "$MODE" = "sweep" ]; then
  say "STAGE 2/4 — regenerate all1023.csv from predictions (CPU, ~40 min)"
  have_predictions || { echo "ERROR: 10 predictions/leaderboards not found under $DATA. See header."; exit 1; }
  $PYBIN code/make_all1023.py
  MODE="figures"
fi

# --- STAGE 3: figures + tables (six notebooks) -------------------------------
say "STAGE 3/4 — regenerate figures + tables (six notebooks, CPU only)"
have_predictions || { echo "ERROR: inputs missing under $DATA. See the header of this script."; exit 1; }
have_sweep       || { echo "ERROR: $DATA/all1023.csv missing. Run 'bash reproduce.sh --sweep'."; exit 1; }
for nb in 00_pre_analysis 31_complementarity 32_fusion_comparison 33_heldout_test 34_gain_analysis 35_polaris_external; do
  say "run $nb.ipynb"
  ( cd "$NB" && $PYBIN -m jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout="$NBTIMEOUT" "$nb.ipynb" )
done

# --- STAGE 4: report ---------------------------------------------------------
say "STAGE 4/4 — done"
echo "Figures written to:  figures/    ($(ls figures/*.png 2>/dev/null | wc -l | tr -d ' ') PNG)"
echo "Tables written to:   tables/     ($(ls tables/*.csv 2>/dev/null | wc -l | tr -d ' ') CSV)"
echo "All figures and tables in the paper have been regenerated from $DATA/."
