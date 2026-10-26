"""qadmet — TDC ADMET multi-task learning research platform.

Core concepts:
- ExperimentSpec: Represents a single experiment as a sum of orthogonally decomposed dataclasses (specs).
- Recipe: A single YAML file is serialized into an ExperimentSpec.
- Registry: Registers backbone/fusion/head/loss/strategy/feature builder via decorators.
- Artifact + Lineage: Every output is accompanied by a manifest.yaml (recipe + input hash + git SHA).
- Runner: spec → execution → artifact generation.

See docs/ARCHITECTURE.md for detailed design.
"""

__version__ = "0.1.0"
