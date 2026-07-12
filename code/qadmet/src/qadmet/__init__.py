"""qadmet — TDC ADMET 멀티태스크 학습 연구 플랫폼.

핵심 개념:
- ExperimentSpec: 실험 한 건을 직교 분해된 dataclass(spec) 들의 합으로 표현.
- Recipe: YAML 한 장이 ExperimentSpec 으로 직렬화된다.
- Registry: backbone/fusion/head/loss/strategy/feature builder 를 데코레이터로 등록.
- Artifact + Lineage: 모든 산출물에 manifest.yaml 동봉(recipe + 입력 해시 + git SHA).
- Runner: spec → 실행 → artifact 생성.

자세한 설계는 docs/ARCHITECTURE.md 참조.
"""

__version__ = "0.1.0"
