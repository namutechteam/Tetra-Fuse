# Polaris ADME-Fang 6-task — script to run 5 models (original paper STL/MTL configuration)
# Server: VPN_Node02 | conda: admet-mtl | repo: /nas/tk_ahn/ADMET-MTL/qadmet
# Training scheme per model:  D-MPNN=MTL, MoLFormer=MTL, CheMeleon=STL, GraphMVP=STL, MAP=STL
# All recipe schemas validated OK. Weight files (chemeleon_mp.pt, graphmvp_hybrid.pth) confirmed.

cd /nas/tk_ahn/ADMET-MTL/qadmet
conda activate admet-mtl
PY=python

# =====================================================================
# Group A — feature precompute not required (graph backbone = synthetic featurize)
# =====================================================================

# A1. D-MPNN  MTL   (from-scratch, grouped head, Kendall UW)
$PY -m qadmet.cli.main train --recipe recipes/train/mtl_dmpnn_robust_polaris_fang.yaml --devices 1

# A2. CheMeleon  STL  (AdapterGNN paper r=128, bottleneck head)   [chemeleon_mp.pt ✓]
$PY -m qadmet.cli.main train --recipe recipes/train/stl_chemeleon_paper_polaris_fang.yaml --devices 1

# A3. GraphMVP  STL  (AdapterGNN paper r=64, bottleneck head)     [graphmvp_hybrid.pth ✓]
$PY -m qadmet.cli.main train --recipe recipes/train/stl_graphmvp_paper_polaris_fang.yaml --devices 1

# =====================================================================
# Group B — molformer_tokens parquet must be precomputed (MoLFormer / MAP)
# =====================================================================

# B0. (one-time) Precompute MoLFormer tokens for Polaris SMILES
$PY -m qadmet.cli.main precompute --recipe recipes/precompute/polaris_fang_molformer_tokens.yaml

# B0b. Symlink output parquet → path referenced by dataset yaml
HASH_DIR=$(ls -d artifacts/precompute/precompute_polaris_fang_molformer_tokens/*/ | head -1)
ln -sf "${HASH_DIR}data/molformer_tokens.parquet" data/embeddings/polaris_fang_molformer_tokens.parquet
ls -la data/embeddings/polaris_fang_molformer_tokens.parquet

# B1. MoLFormer  MTL  (LoRA r=16, grouped head)                  [MoLFormer-XL 84M base]
$PY -m qadmet.cli.main train --recipe recipes/train/mtl_molformer_robust_polaris_fang.yaml --devices 1

# B2. MAP  STL  (joint 3-modality: CheMeleon+MoLFormer+GraphMVP, robust_attention, bottleneck head)
#     ※ joint-fusion STL does not support DDP → sequential on single GPU (6 task × 5 seed = 30 models)
$PY -m qadmet.cli.main train --recipe recipes/train/stl_map_robust_polaris_fang.yaml --devices 1

# =====================================================================
# Option — DDP (2-GPU): applies only to MTL backbone (dmpnn / molformer). STL uses single GPU.
# =====================================================================
# torchrun --nproc_per_node=2 -m qadmet.cli.main train \
#   --recipe recipes/train/mtl_molformer_robust_polaris_fang.yaml --devices 2 --strategy ddp

# =====================================================================
# Option — smoke test (1 seed, 2 epoch)
# =====================================================================
# $PY -m qadmet.cli.main train --recipe recipes/train/mtl_dmpnn_robust_polaris_fang.yaml \
#   -o trainer.max_epochs=2 -o trainer.seeds='[0]' --force
