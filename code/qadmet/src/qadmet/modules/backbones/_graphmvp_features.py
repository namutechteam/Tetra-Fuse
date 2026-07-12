"""SMILES → PyG Data converter (OGB 9-d atom / 3-d bond features).

GraphMVP/src_regression/datasets_complete_feature/molecule_datasets.py:18-59 의
``mol_to_graph_data_obj_simple`` 을 그대로 옮긴 형태. 입력 feature 스펙
(OGB ``atom_to_feature_vector`` / ``bond_to_feature_vector``) 이 사전학습 가중치의
``AtomEncoder`` / ``BondEncoder`` 와 정확히 일치해야 weight 가 의미를 가진다.

graphmvp backbone 의 입력은 collate 단계에서 SMILES 로부터 즉석으로 생성된
``torch_geometric.data.Batch`` (Data 들의 묶음). ETKDG 같은 무거운 3D 변환이 없어
on-the-fly 변환 비용이 충분히 작다 — UniMol2 와 달리 LMDB precompute 불필요.
"""

from __future__ import annotations

import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch_geometric.data import Data

# RDKit 이 atom 0 개 mol 을 만드는 경우는 거의 없지만 방어적으로 — graphmvp_data 슬롯에서
# 이런 mol 은 None 으로 표시되고 collate 가 placeholder + presence mask 로 처리한다.

_BOND_FEAT_DIM = 3
_ATOM_FEAT_DIM = 9


def _empty_data() -> Data:
    """Atom 0 / Bond 0 placeholder Data — collate 시 invalid sample 자리채움용.

    실제 forward 에서는 presence mask 로 차단되므로 출력 사용 X. BatchNorm 통계 보호를
    위해 최소 1 atom (모든 feature index 0) 으로 더미를 넣는다.
    """
    x = torch.zeros((1, _ATOM_FEAT_DIM), dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, _BOND_FEAT_DIM), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_pyg_data(smi: str) -> Data | None:
    """SMILES → PyG Data with OGB 9-d atom / 3-d bond features.

    Returns:
        ``Data(x=(N, 9), edge_index=(2, E), edge_attr=(E, 3))`` (long).
        Invalid / atom-less SMILES 는 ``None``.
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
    """SMILES 리스트 → (PyG Batch, present_mask).

    Returns:
        - batch: ``torch_geometric.data.Batch`` — invalid 샘플은 1-atom placeholder.
        - present_mask: ``(B,) bool`` — RDKit parse / atom 0 검증 실패는 False.

    Why placeholder:
        Batch.from_data_list 가 0-node Data 를 허용하지만, BatchNorm 통계가 unstable
        한 경우가 있어 1-atom 더미가 안전하다. 어차피 fusion 에서 presence 마스크로
        해당 modality alpha 는 차단된다.
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
