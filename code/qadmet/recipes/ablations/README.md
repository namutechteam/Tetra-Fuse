# Ablation recipes

코드에 phase 번호 박는 대신, **변형 그룹 단위 디렉토리** 로 sweep 을 표현하는 자리.
한 그룹 안의 recipe 들은 *한 가지 축* 만 바꾼다 (alpha, weights, fusion type 등).
default 와 동일한 변형은 두지 않는다 (`recipes/ensemble/` / `recipes/train/` 의 기본 recipe 가 그 자리를 메운다).

## 현재 그룹

### `hybrid_alpha_sweep/`

`stack_hybrid_blend` (옛 v6) 의 ``alpha`` 만 변동. base / sub-strategy / sub-kwargs 고정.

| recipe | alpha | 의미 |
|---|---|---|
| (default) `recipes/ensemble/stack_hybrid_blend.yaml` | 0.5 | 옛 v6 default. |
| `a03.yaml` | 0.3 | secondary(=average 70/30) 우세. |
| `a07.yaml` | 0.7 | primary(=smart_fallback) 우세. |

실행:

```bash
qadmet stack --recipe recipes/ensemble/stack_hybrid_blend.yaml          # alpha=0.5
qadmet stack --recipe recipes/ablations/hybrid_alpha_sweep/a03.yaml     # alpha=0.3
qadmet stack --recipe recipes/ablations/hybrid_alpha_sweep/a07.yaml     # alpha=0.7
```

비교는 `qadmet evaluate` 의 `compare_to` 또는 `compare_kinds` 로 묶어 한 번에:

```yaml
# recipes/evaluate/hybrid_alpha_sweep_diff.yaml (예시 — 직접 작성)
name: hybrid_alpha_sweep_diff
kind: evaluate
evaluation:
  benchmark: tdc_admet
  aggregate: mean
ensemble:
  bases: []
  strategy: average
evaluation:
  compare_to: [stack_hybrid_blend, stack_hybrid_blend_a03, stack_hybrid_blend_a07]
  kwargs:
    target: stack_hybrid_blend
    target_kind: stack
    compare_kinds:
      stack_hybrid_blend_a03: stack
      stack_hybrid_blend_a07: stack
```

### `stack_avg_weights_sweep/`

`stack_average_70_30` 의 ``weights`` 만 변동. base 고정 (mtl_dmpnn_grouped + stl_dmpnn_rdkit).

| recipe | weights | 의미 |
|---|---|---|
| `w50_50.yaml` | [0.5, 0.5] | 균등 평균. |
| (default) `recipes/ensemble/stack_average_70_30.yaml` | [0.7, 0.3] | 옛 best ensemble — MTL 우세. |
| `w80_20.yaml` | [0.8, 0.2] | MTL 강조. |

## 패턴

새 그룹 만들 때:

1. `recipes/ablations/<group>/` 디렉토리 생성.
2. 각 변형마다 `recipe.name` 이 서로 달라야 함 (artifact 키). 같은 recipe (= 같은 hash) 는 한 artifact 로 합쳐지므로 sweep 효과 없음.
3. `recipe.kind` 는 train / stack / evaluate 중 하나. 같은 그룹 안에서 kind 섞지 않기.
4. `notes` 에 비교 대상과 default 위치를 명시 — 후속 작업이 어떤 축이 변동했는지 한눈에 파악.
5. validate: `qadmet recipe validate recipes/ablations/<group>/*.yaml`.
