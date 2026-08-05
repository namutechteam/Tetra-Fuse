#!/usr/bin/env python
"""Check the shipped training recipes against what was actually trained.

Every released run records the *fully resolved* recipe it ran under, inside
`artifacts/baselines/<model>/baseline/manifest.yaml`. That manifest is the ground
truth: if a shipped recipe disagrees with it, then someone who re-trains from this
repository gets a different model from the one the paper reports.

This script compares the two, field by field, for all 10 TDC base models. It is
how a drift was caught once already — four recipes shipped with shorter training
budgets and the wrong `trainer.seeds` than the runs that produced the released
predictions (see TRAINING.md, "Recipe/manifest audit").

    python code/audit_recipes.py                       # exits 1 on any mismatch
    python code/audit_recipes.py --verbose             # print every compared field

Requires the training artifacts (`artifacts/baselines/`), which are not part of
the public release; run it on the machine that holds them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Fields whose disagreement changes the trained model. `notes` and the various
# path kwargs are prose/environment and are compared only under --verbose.
CRITICAL = (
    "trainer.max_epochs", "trainer.patience", "trainer.seeds", "trainer.precision",
    "trainer.batch_size", "trainer.lr", "trainer.lr_lora", "trainer.lr_fusion",
    "trainer.early_stopping", "trainer.predict_ckpt", "trainer.head_warmup_epochs",
    "dataset.name", "features.missing_policy", "features.graph",
    "model.type", "model.backbone.type", "model.fusion.type", "model.head.type",
    "loss.type", "loss.classification", "loss.regression",
)


def _flat(obj, prefix=""):
    """Flatten nested dicts/dataclasses to {dotted.key: scalar-or-json}."""
    out = {}
    if hasattr(obj, "__dataclass_fields__"):
        for f in obj.__dataclass_fields__:
            out.update(_flat(getattr(obj, f), f"{prefix}.{f}" if prefix else f))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(list(obj), sort_keys=True, default=str)
    else:
        out[prefix] = obj
    return out


def _same(a, b) -> bool:
    """Compare two flattened values, tolerating list-vs-JSON-string encoding."""
    def norm(v):
        if isinstance(v, str):
            try:
                return json.dumps(json.loads(v), sort_keys=True)
            except Exception:
                return v
        return v
    return norm(a) == norm(b)


def main() -> int:
    here = Path(__file__).resolve().parent
    src = here / "qadmet" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines",
                    default="/nas/tk_ahn/ADMET-MTL/qadmet_public/artifacts/baselines",
                    help="directory of per-model baseline manifests")
    ap.add_argument("--recipes", default=str(here / "qadmet" / "recipes" / "train"))
    ap.add_argument("--verbose", action="store_true",
                    help="also report non-critical fields (notes, path kwargs)")
    args = ap.parse_args()

    import yaml

    from qadmet.core.recipe import load_recipe

    base = Path(args.baselines)
    if not base.is_dir():
        print(f"ERROR: baselines dir not found: {base}\n"
              f"       (training artifacts are not shipped; run this where they live)",
              file=sys.stderr)
        return 2

    # Index the shipped recipes by the recipe name they resolve to, so a rename
    # of the file does not break the mapping.
    specs = {}
    for fn in sorted(os.listdir(args.recipes)):
        if not fn.endswith(".yaml"):
            continue
        try:
            specs[fn] = load_recipe(Path(args.recipes) / fn)
        except Exception as exc:  # a malformed recipe is itself a finding
            print(f"UNLOADABLE {fn}: {type(exc).__name__}: {exc}")

    models = sorted(d for d in os.listdir(base) if (base / d).is_dir())
    problems = 0
    for model in models:
        man_path = base / model / "baseline" / "manifest.yaml"
        if not man_path.exists():
            continue
        recipe = yaml.safe_load(open(man_path))["recipe"]
        name = recipe["name"]
        hits = [(fn, sp) for fn, sp in specs.items() if getattr(sp, "name", None) == name]
        if not hits:
            hits = [(fn, sp) for fn, sp in specs.items() if fn.startswith(name)]
        if not hits:
            print(f"MISSING  {model:16} no shipped recipe resolves to {name!r}")
            problems += 1
            continue
        fn, spec = hits[0]
        trained, shipped = _flat(recipe), _flat(spec)
        keys = CRITICAL if not args.verbose else sorted(set(trained) & set(shipped))
        bad = [(k, trained[k], shipped[k]) for k in keys
               if k in trained and k in shipped and not _same(trained[k], shipped[k])]
        if bad:
            problems += len(bad)
            print(f"MISMATCH {model:16} ({fn})")
            for k, t, s in bad:
                print(f"           {k:30} trained={t!r}  shipped={s!r}")
        else:
            print(f"OK       {model:16} ({fn})")

    print(f"\n{len(models)} models checked, {problems} mismatching field(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
