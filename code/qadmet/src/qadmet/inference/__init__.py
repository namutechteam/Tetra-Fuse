"""SMILES -> 22 ADMET predictions with the released Tetra-Fuse weights.

This is the *model-usage* axis of the release: it loads the published PEFT
weights (plus the pretrained backbones you download separately) and predicts
new molecules. The paper-reproduction axis lives in ``notebooks/`` and works
from the shipped prediction parquets instead.

    from qadmet.inference import predict_smiles
    df = predict_smiles(["CCO", "c1ccccc1"], weights_root="weights",
                        recipes_root="code/qadmet/recipes")

See ``weights/README.md`` for the directory layout and backbone downloads.
"""
from .loader import ENSEMBLE, LoadedMember, load_member
from .predict import predict_smiles

__all__ = ["ENSEMBLE", "LoadedMember", "load_member", "predict_smiles"]
