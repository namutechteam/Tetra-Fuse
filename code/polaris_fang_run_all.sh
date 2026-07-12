# Polaris ADME-Fang 6-task — 5개 모델 실행 스크립트 (원 논문 STL/MTL 구성)
# 서버: VPN_Node02 | conda: admet-mtl | repo: /nas/tk_ahn/ADMET-MTL/qadmet
# 모델별 학습 방식:  D-MPNN=MTL, MoLFormer=MTL, CheMeleon=STL, GraphMVP=STL, MAP=STL
# 모든 recipe schema 검증 OK. weight 파일(chemeleon_mp.pt, graphmvp_hybrid.pth) 확인 완료.

cd /nas/tk_ahn/ADMET-MTL/qadmet
conda activate admet-mtl
PY=python

# =====================================================================
# 그룹 A — feature precompute 불필요 (graph backbone = synthetic featurize)
# =====================================================================

# A1. D-MPNN  MTL   (from-scratch, grouped head, Kendall UW)
$PY -m qadmet.cli.main train --recipe recipes/train/mtl_dmpnn_robust_polaris_fang.yaml --devices 1

# A2. CheMeleon  STL  (AdapterGNN paper r=128, bottleneck head)   [chemeleon_mp.pt ✓]
$PY -m qadmet.cli.main train --recipe recipes/train/stl_chemeleon_paper_polaris_fang.yaml --devices 1

# A3. GraphMVP  STL  (AdapterGNN paper r=64, bottleneck head)     [graphmvp_hybrid.pth ✓]
$PY -m qadmet.cli.main train --recipe recipes/train/stl_graphmvp_paper_polaris_fang.yaml --devices 1

# =====================================================================
# 그룹 B — molformer_tokens parquet 선행 필요 (MoLFormer / MAP)
# =====================================================================

# B0. (한 번만) Polaris SMILES 의 MoLFormer 토큰 precompute
$PY -m qadmet.cli.main precompute --recipe recipes/precompute/polaris_fang_molformer_tokens.yaml

# B0b. 출력 parquet → dataset yaml 참조 경로로 symlink
HASH_DIR=$(ls -d artifacts/precompute/precompute_polaris_fang_molformer_tokens/*/ | head -1)
ln -sf "${HASH_DIR}data/molformer_tokens.parquet" data/embeddings/polaris_fang_molformer_tokens.parquet
ls -la data/embeddings/polaris_fang_molformer_tokens.parquet

# B1. MoLFormer  MTL  (LoRA r=16, grouped head)                  [MoLFormer-XL 84M base]
$PY -m qadmet.cli.main train --recipe recipes/train/mtl_molformer_robust_polaris_fang.yaml --devices 1

# B2. MAP  STL  (joint 3-modality: CheMeleon+MoLFormer+GraphMVP, robust_attention, bottleneck head)
#     ※ joint-fusion STL 은 DDP 미지원 → 단일 GPU 순차 (6 task × 5 seed = 30 모델)
$PY -m qadmet.cli.main train --recipe recipes/train/stl_map_robust_polaris_fang.yaml --devices 1

# =====================================================================
# 옵션 — DDP (2-GPU) : MTL backbone (dmpnn / molformer) 만 해당. STL 은 단일 GPU.
# =====================================================================
# torchrun --nproc_per_node=2 -m qadmet.cli.main train \
#   --recipe recipes/train/mtl_molformer_robust_polaris_fang.yaml --devices 2 --strategy ddp

# =====================================================================
# 옵션 — smoke 테스트 (1 seed, 2 epoch)
# =====================================================================
# $PY -m qadmet.cli.main train --recipe recipes/train/mtl_dmpnn_robust_polaris_fang.yaml \
#   -o trainer.max_epochs=2 -o trainer.seeds='[0]' --force
