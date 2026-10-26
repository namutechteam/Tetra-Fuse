"""SMILES → PyG Data converter (OGB 9-d atom / 3-d bond features).

GraphMVP/src_regression/datasets_complete_feature/molecule_datasets.py:18-59's
``mol_to_graph_data_obj_simple`` ported as-is. Input feature spec
(OGB ``atom_to_feature_vector`` / ``bond_to_feature_vector``) must exactly match the pretrained weights'
``AtomEncoder`` / ``BondEncoder`` for the weight to be meaningful.

graphmvp backbone's input is generated on-the-fly from SMILES at the collate stage,
``torch_geometric.data.Batch`` (a bundle of Data objects). Since there's no heavy 3D conversion like ETKDG,
the on-the-fly conversion cost is small enough — unlike UniMol2, no LMDB precompute needed.
"""

from __future__ import annotations

import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch_geometric.data import Data

# RDKit rarely produces a mol with 0 atoms, but defensively — in the graphmvp_data slot,
# such mols are marked as None, and collate handles them with a placeholder + presence mask.

_BOND_FEAT_DIM = 3
_ATOM_FEAT_DIM = 9


def _empty_data() -> Data:
    """Atom 0 / Bond 0 placeholder Data — for filling invalid sample slots during collate.

    Blocked by presence mask in the actual forward pass, so the output is not used. To protect BatchNorm statistics,
    a dummy with at least 1 atom (all feature indices 0) is inserted.
    """
    x = torch.zeros((1, _ATOM_FEAT_DIM), dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, _BOND_FEAT_DIM), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_pyg_data(smi: str) -> Data | None:
    """SMILES → PyG Data with OGB 9-d atom / 3-d bond features.

    Returns:
        ``Data(x=(N, 9), edge_index=(2, E), edge_attr=(E, 3))`` (long).
        Invalid / atom-less SMILES is ``None``.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    atoms = mol.GetAtoms()
    if len(atoms) == 0:
        return None
    atom_features = [atom_to_feature_vector(a) for a in atoms]
    x = torch.tensor(np.asarray(atom_features), dtype=torch.long)

    if len(mol.GetBonds()) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, _BOND_FEAT_DIM), dtype=torch.long)
    else:
        edges: list[tuple[int, int]] = []
        edge_features: list[list[int]] = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            f = bond_to_feature_vector(bond)
            edges.append((i, j))
            edge_features.append(f)
            edges.append((j, i))
            edge_features.append(f)
        edge_index = torch.tensor(np.asarray(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_batch_to_pyg(smiles: list[str]) -> tuple["object", torch.Tensor]:
    """SMILES list → (PyG Batch, present_mask).

    Returns:
        - batch: ``torch_geometric.data.Batch`` — invalid samples are 1-atom placeholders.
        - present_mask: ``(B,) bool`` — False on RDKit parse / atom 0 validation failure.

    Why placeholder:
        Batch.from_data_list allows 0-node Data, but BatchNorm statistics can become unstable
        in some cases, so a 1-atom dummy is safer. Since the presence mask in fusion
        blocks the corresponding modality alpha anyway.
    """
    from torch_geometric.data import Batch

    data_list: list[Data] = []
    flags: list[bool] = []
    for smi in smiles:
        d = smiles_to_pyg_data(smi)
        if d is None:
            data_list.append(_empty_data())
            flags.append(False)
        else:
            data_list.append(d)
            flags.append(True)
    batch = Batch.from_data_list(data_list)
    return batch, torch.tensor(flags, dtype=torch.bool)
