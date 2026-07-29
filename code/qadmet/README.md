# qadmet

TDC ADMET 멀티태스크 학습 연구 플랫폼. **Recipe (YAML) → Spec → Runner → Artifact** 로
"어떤 실험을 돌릴지" 를 코드와 분리해 선언한다. backbone / fusion / head / loss /
strategy / feature builder / calibrator 모두 registry 기반 — 새 modality·loss·strategy 는
파일 하나 + 데코레이터 한 줄로 등록된다.

이 디렉토리는 논문(*When and Why to Fuse Molecular Representations for ADMET Prediction*)의
학습·평가 엔진으로 공개되는 코어다. 논문의 10개 TDC 기본 모델과 5개 Polaris 앙상블 멤버를
재현하는 recipe 와 실행 순서는 [`../TRAINING.md`](../TRAINING.md) 에 정리돼 있다.

> **그림·표 재현에는 이 엔진이 필요 없다.** Figure/Table 은 사전 계산된 예측·리더보드
> (`../../data/recompute_data/`) 만으로 `../../notebooks/` 의 6개 섹션 노트북
> (`00_pre_analysis.ipynb` + `31`–`35_*.ipynb`) 에서 GPU·네트워크 없이 재현된다. 이 엔진은
> 그 예측·리더보드를 **처음부터 다시 학습**해 만들고 싶을 때 쓴다.

## 설치

```bash
conda activate admet-mtl   # 무거운 의존(torch/lightning/chemprop/transformers/rdkit/tdc)이 깔린 환경
cd code/qadmet
pip install -e .           # qadmet CLI 등록 (entry point: qadmet.cli.main:main)
```

`pyproject.toml` 의 `[project] dependencies` 는 최소 (`pyyaml` / `numpy` / `pandas`) —
torch / lightning / chemprop / transformers / peft / rdkit / tdc 같은 무거운 의존은
conda env (`admet-mtl`) 책임이다. 재현 환경 상세는 [`../TRAINING.md`](../TRAINING.md).

## CLI

| 서브커맨드 | 역할 |
|---|---|
| `qadmet precompute --recipe <yaml>` | feature parquet 사전계산 (MoLFormer token embeddings 등). |
| `qadmet train --recipe <yaml>` | MTL / STL 학습. 학습 직후 val + test split 모두 `predictions.parquet` 로 저장. |
| `qadmet evaluate --recipe <yaml>` | 7-컬럼 metric 표 (`dataset,metric,ours,ours_std,rank,n_total,Top,Top_model`) + 리더보드 순위 → `leaderboard.csv`. |
| `qadmet recipe validate <path...>` | recipe YAML schema 검증. |
| `qadmet artifacts list [--kind <kind>]` | 저장된 artifact 목록. |
| `qadmet lineage show <kind> <name>` | artifact 의존 그래프 (graphviz dot). |

별도의 `predict` 서브커맨드는 없다 — train runner 가 학습 즉시 val + test 모두에 대한
`predictions.parquet` 를 작성하므로 그 결과가 곧 PredictionSet 이다.

기본 흐름 (논문 재현 — 전체 순서는 [`../TRAINING.md`](../TRAINING.md)):

```bash
# 1) precompute (MoLFormer backbone 전용, 한 번만)
qadmet precompute --recipe recipes/precompute/molformer_tokens.yaml

# 2) train (예: D-MPNN·MTL 기본 모델)
qadmet train --recipe recipes/train/mtl_dmpnn_robust_causal6_kendall_epoch100.yaml

# 3) evaluate → leaderboard.csv (예: 위 모델)
qadmet evaluate --recipe recipes/evaluate/leaderboard_mtl_dmpnn_epoch100_s01234.yaml
```

CLI override (dot-notation, 반복 가능):

```bash
qadmet train --recipe recipes/train/mtl_dmpnn_robust_causal6_kendall_epoch100.yaml \
    --override trainer.max_epochs=2 --override trainer.seeds='[0]'
```

`--force` 는 같은 run_id (= `sha1(recipe + git_sha)[:12]`) 의 기존 artifact 디렉토리를 삭제 후 재생성.

## 디렉토리

| 경로 | 설명 |
|---|---|
| [src/qadmet/](src/qadmet/) | 메인 패키지. core (registry/spec/recipe/artifact/lineage/runner) + tasks/datasets/features/modules/training/ensembles/evaluation/runners/cli. |
| [recipes/](recipes/) | 논문 재현 recipe (38개). kind 별 디렉토리 + [`_base/`](recipes/_base/) 컴포저블 base. |
| `LICENSE` | MIT. |

`artifacts/`, `data/`, `checkpoints*/`, `lightning_logs/`, `logs/` 는 실행 시 생성되며
릴리스에는 포함하지 않는다.

## Recipes (38개, 논문 재현 세트)

| kind | 개수 | 내용 |
|---|---|---|
| `train/` | 15 | TDC-22 기본 모델 10개 (5 MTL + 5 STL) + Polaris 앙상블 멤버 5개. |
| `evaluate/` | 10 | TDC-22 10개 모델의 leaderboard.csv 생성 recipe. |
| `_base/` | 9 | kept recipe 가 `!include` 하는 dataset / model / loss / trainer 조각. |
| `groupings/` | 2 | `tdc_admet_22_causal6` (MTL 6-group) + `tdc_admet_22` (STL 22-task TaskSet 등록). |
| `precompute/` | 2 | MoLFormer token embeddings (TDC / Polaris). |

각 모델의 정확한 train / evaluate recipe 파일명 매핑은 [`../TRAINING.md`](../TRAINING.md) 표 참조.
Polaris 5개는 Polaris Hub 가 채점하므로 `evaluate/` recipe 가 없다.

## Recipe 검증

```bash
qadmet recipe validate recipes/train/*.yaml recipes/evaluate/*.yaml \
                       recipes/precompute/*.yaml
```

## MoLFormer HuggingFace revision pin

`ibm/MoLFormer-XL-both-10pct` 는 custom modelling code (`trust_remote_code=True`) 를 쓰며,
이후 hub commit 은 transformers 4.57.x 에 없는 함수를 요구해 `ImportError` 를 낸다. backbone 이
동작 revision 을 고정한다 (`src/qadmet/modules/backbones/molformer_lora.py` 의
`MOLFORMER_REVISION`). 재현을 위해 이 pin 을 유지할 것.
