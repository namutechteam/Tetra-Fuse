# qadmet

TDC ADMET 멀티태스크 학습 연구 플랫폼. **Recipe (YAML) → Spec → Runner → Artifact** 로
"어떤 실험을 돌릴지" 를 코드와 분리해 선언한다. backbone / fusion / head / loss /
strategy / feature builder / calibrator 모두 registry 기반 — 새 modality·loss·strategy 는
파일 하나 + 데코레이터 한 줄로 등록된다.

설계 근거: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). 도메인 컨텍스트 / 연구 방향:
[CLAUDE.md](CLAUDE.md).

## 설치

```bash
conda activate admet-mtl   # 사전 구성된 admet-mtl 환경 사용
cd /nas/tk_ahn/ADMET-MTL/qadmet
pip install -e .
```

`pyproject.toml` 의 `[project] dependencies` 는 최소 (`pyyaml` / `numpy` / `pandas`) —
torch / lightning / chemprop / transformers / peft / xgboost / rdkit / tdc 같은 무거운 의존은
conda env (`admet-mtl`) 책임.

## CLI

| 서브커맨드 | 역할 |
|---|---|
| `qadmet precompute --recipe <yaml>` | feature parquet 사전계산 (rdkit2d / xtb / molformer / molformer_tokens / unimol2). |
| `qadmet train --recipe <yaml>` | MTL / STL 학습. 학습 직후 val + test split 모두 `predictions.parquet` 로 저장. |
| `qadmet stack --recipe <yaml>` | 학습된 base artifact 들을 ensembles strategy 로 묶음. |
| `qadmet evaluate --recipe <yaml>` | 7-컬럼 metric 표 + W/L/T 비교 + (옵션) 리더보드 순위. |
| `qadmet recipe validate <path...>` | YAML schema 검증. |
| `qadmet artifacts list [--kind <kind>]` | 저장된 artifact 목록. |
| `qadmet lineage show <kind> <name>` | artifact 의존 그래프 (graphviz dot). |

별도의 `predict` 서브커맨드는 두지 않는다 — train runner 가 학습 즉시 val + test 모두에 대한
`predictions.parquet` 를 작성하므로 그 결과가 곧 PredictionSet 이다 (Step 4.2 부터).

기본 흐름:

```bash
# 1) precompute (한 번만; 결과는 data/ 안의 parquet)
qadmet precompute --recipe recipes/precompute/rdkit2d.yaml

# 2) train (MTL + STL — base artifact 두 개)
qadmet train --recipe recipes/train/mtl_dmpnn_grouped.yaml
qadmet train --recipe recipes/train/stl_dmpnn_rdkit.yaml

# 3) stack (옛 phase3/4 대체)
qadmet stack --recipe recipes/ensemble/stack_average_70_30.yaml      # 옛 v1
qadmet stack --recipe recipes/ensemble/stack_hybrid_blend.yaml       # 옛 v6 (alpha=0.5)

# 4) evaluate (target vs others 비교, legacy CSV 도 비교 가능)
qadmet evaluate --recipe recipes/evaluate/leaderboard_diff.yaml
```

CLI override (dot-notation, 반복 가능):

```bash
qadmet train --recipe recipes/train/mtl_dmpnn_grouped.yaml \
    --override trainer.max_epochs=2 --override trainer.seeds='[0]'
```

`--force` 는 같은 run_id (= `sha1(recipe + git_sha)[:12]`) 의 기존 artifact 디렉토리를 삭제 후 재생성.

## 디렉토리

| 경로 | 설명 |
|---|---|
| [src/qadmet/](src/qadmet/) | 메인 패키지. core (registry/spec/recipe/artifact/lineage/runner) + tasks/datasets/features/modules/training/ensembles/evaluation/runners/cli. |
| [recipes/](recipes/) | YAML recipe. kind 별 디렉토리 + [`_base/`](recipes/_base/) 컴포저블 base + [`ablations/`](recipes/ablations/) sweep. |
| [scripts/](scripts/) | 얇은 shell wrapper (logic 없음, `qadmet` CLI 호출만). |
| [scripts-old/](scripts-old/) | 옛 ~8,000 lines 코드. parity 검증용 read-only — **수정 금지**. |
| [tests/](tests/) | pytest 단위 테스트 (425 passed). |
| [parity/](parity/) | scripts-old ↔ qadmet 결과 동등성 검증 (W/L/T 모드). |
| [docs/](docs/) | 활성 문서 (~7개) + [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). 시간정지 산출물은 [docs/archive/](docs/archive/) — 명시 요청 시에만 참조. |
| [artifacts/](artifacts/) | runner 산출물 (manifest.yaml + data/). |
| `data/`, `checkpoints*/`, `lightning_logs/`, `logs/` | 데이터/체크포인트/로그 — 손대지 않음. |

## Ablations

`recipes/ablations/<group>/<variant>.yaml` 패턴으로 한 축만 변동시키는 sweep 을 표현.
default 와 동일한 변형은 두지 않는다 (default 는 `recipes/ensemble/` 또는 `recipes/train/` 가
보유). 자세한 패턴은 [recipes/ablations/README.md](recipes/ablations/README.md).

현재 그룹:
- `hybrid_alpha_sweep/` — alpha ∈ {0.3, 0.7} (default=0.5).
- `stack_avg_weights_sweep/` — weights ∈ {[0.5,0.5], [0.8,0.2]} (default=[0.7,0.3]).

## 테스트

```bash
pytest tests/ -q                                                # 425 passed
qadmet recipe validate recipes/{train,precompute,ensemble,evaluate}/*.yaml \
                       recipes/ablations/*/*.yaml
```

## Parity (scripts-old ↔ qadmet)

Step 3.7 4종 모두 PASS (성능 수준 비교 모드, tie_eps=0.005, 22 task):

| 비교 | net |
|---|---|
| `mtl_dmpnn_grouped` vs `phase1.1_grouped` | +6 |
| `stl_dmpnn_rdkit` vs `phase2_stl_summary` | -7 (데이터 경로 통일 영향) |
| `stack_average_70_30` vs `phase3_stack_v1_avg70` | -1 |
| `stack_hybrid_blend` vs `phase4_stack_v6_a0.5` | -3 |

scripts-old/ 는 보존 (사용자 결정).


---

## Note for the paper code release

This is the `qadmet` research platform — a recipe + registry + lineage framework for
TDC ADMET multi-task learning and multi-representation fusion. It is included here as the
core training/evaluation engine behind the paper.

- Model training is driven by recipe YAMLs under `recipes/` via
  `python -m qadmet.cli.main train --recipe <recipe.yaml>`.
- The five backbones behind the paper's ensemble and their recipes are documented in
  `../TRAINING.md`.
- `src/qadmet/modules/backbones/molformer_lora.py` pins the working MoLFormer-XL HuggingFace
  revision (`MOLFORMER_REVISION`); keep this pin for reproducibility.
- Figures and tables do **not** require this engine — see `../../notebooks/` and `../../data/`.

Dependencies (RDKit, PyTorch, Lightning, Transformers, chemprop, pretrained backbone weights)
are installed in the `admet-mtl` conda environment described in `../TRAINING.md`.
