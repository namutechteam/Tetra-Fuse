"""GFN2-xTB 4-dim QC descriptor builder.

옛 ``scripts-old/compute_xtb_descriptors.py`` 의 SMILES → ETKDG 3D → MMFF/UFF →
xTB single-point → JSON 파싱 파이프라인을 ``PrecomputeBase`` 위에 재구성.

출력 (모두 scalar):
    - dipole          : 쌍극자 모멘트 크기 (Debye)
    - homo_lumo_gap   : HOMO-LUMO gap (eV)
    - electrons       : 총 전자 수
    - total_energy    : 총 에너지 (Hartree)

실패 시 (3D embed 실패, xTB SCF 발산, 타임아웃) 모든 채널 NaN.
하류 missing_policy 가 mask/drop/impute 로 처리.

Note:
    xtb 바이너리(conda-forge) + RDKit 가 conda env 에 있어야 한다. import 시점에는
    rdkit 가 필요 없도록 worker 함수 안에서 lazy import.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

ETKDG_SEED = 42
XTB_KEY_ALIASES: dict[str, list[str]] = {
    "total_energy": ["total energy"],
    "homo_lumo_gap": ["HOMO-LUMO gap/eV", "HOMO-LUMO gap / eV", "HOMO-LUMO Gap"],
    "dipole": ["dipole", "total dipole moment", "dipole moment"],
    "electrons": ["number of electrons", "total number of electrons"],
}
DESCRIPTOR_COLS: tuple[str, ...] = ("dipole", "homo_lumo_gap", "electrons", "total_energy")
NAN_VEC = np.full(len(DESCRIPTOR_COLS), np.nan, dtype=np.float64)


def _smiles_to_xyz(smiles: str, *, largest_fragment: bool) -> str | None:
    """SMILES → ETKDGv3 embedding → MMFF/UFF optimize → XYZ block. 실패 시 None."""
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore

    if largest_fragment:
        mol0 = Chem.MolFromSmiles(smiles)
        if mol0 is None:
            return None
        chooser = rdMolStandardize.LargestFragmentChooser()
        largest = chooser.choose(mol0)
        if largest is None:
            return None
        smiles = Chem.MolToSmiles(largest)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = ETKDG_SEED
    params.useRandomCoords = False
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=500) != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            return None
    return Chem.MolToXYZBlock(mol)


def _lookup(d: dict, aliases: list[str]):
    for k in aliases:
        if k in d:
            return d[k]
    return None


def _parse_xtb_json(data: dict) -> dict[str, float]:
    raw_dipole = _lookup(data, XTB_KEY_ALIASES["dipole"])
    if isinstance(raw_dipole, (list, tuple)) and len(raw_dipole) >= 3:
        # xtb 6.5: [dx, dy, dz] (a.u.). 일부 버전: [dx, dy, dz, |d|].
        vec = np.asarray(raw_dipole[:3], dtype=float)
        dipole = float(np.linalg.norm(vec))
    elif isinstance(raw_dipole, (int, float)):
        dipole = float(raw_dipole)
    else:
        dipole = float("nan")
    gap = _lookup(data, XTB_KEY_ALIASES["homo_lumo_gap"])
    gap = float(gap) if isinstance(gap, (int, float)) else float("nan")
    ne = _lookup(data, XTB_KEY_ALIASES["electrons"])
    ne = float(ne) if isinstance(ne, (int, float)) else float("nan")
    te = _lookup(data, XTB_KEY_ALIASES["total_energy"])
    te = float(te) if isinstance(te, (int, float)) else float("nan")
    return {"dipole": dipole, "homo_lumo_gap": gap, "electrons": ne, "total_energy": te}


def _run_xtb(xyz_str: str, timeout: int) -> dict[str, float]:
    """XYZ → xtb single-point → descriptor dict. 실패는 NaN row."""
    from pathlib import Path

    nan_row = {c: float("nan") for c in DESCRIPTOR_COLS}
    with tempfile.TemporaryDirectory(prefix="xtb_") as tmp:
        xyz_path = Path(tmp) / "mol.xyz"
        xyz_path.write_text(xyz_str)
        try:
            subprocess.run(
                ["xtb", xyz_path.name, "--gfn", "2", "--sp", "--json", "--parallel", "1"],
                cwd=tmp,
                timeout=timeout,
                capture_output=True,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return nan_row
        json_path = Path(tmp) / "xtbout.json"
        if not json_path.exists():
            return nan_row
        try:
            with json_path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return nan_row
        return _parse_xtb_json(data)


def _process_one(args: tuple[str, int, bool]) -> tuple[str, np.ndarray]:
    """(smi, timeout, largest_fragment) → (smi, 4-dim float64 vector)."""
    smi, timeout, largest_fragment = args
    try:
        xyz = _smiles_to_xyz(smi, largest_fragment=largest_fragment)
        if xyz is None:
            return smi, NAN_VEC.copy()
        result = _run_xtb(xyz, timeout=timeout)
        vec = np.array(
            [result["dipole"], result["homo_lumo_gap"], result["electrons"], result["total_energy"]],
            dtype=np.float64,
        )
        return smi, vec
    except Exception:
        return smi, NAN_VEC.copy()


@FEATURE_BUILDERS.register("xtb")
class XtbBuilder(PrecomputeBase):
    """ETKDG 3D + xTB single-point. ``ProcessPoolExecutor`` 로 병렬화."""

    name = "xtb"
    column_prefix = ""  # 사용 안 함 — columns 직접 지정.
    output_dim = len(DESCRIPTOR_COLS)
    default_batch_size = 512
    default_checkpoint_every = 500

    def __init__(
        self,
        n_jobs: int = 8,
        timeout: int = 180,
        largest_fragment: bool = False,
    ) -> None:
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.largest_fragment = largest_fragment

    @property
    def columns(self) -> list[str]:
        return list(DESCRIPTOR_COLS)

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        # 외부 멀티프로세싱 시 xTB 내부 OMP 스레드 제한 (경합 방지).
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        out: dict[str, np.ndarray] = {}
        items = [(s, self.timeout, self.largest_fragment) for s in smiles]
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            futures = {ex.submit(_process_one, it): it[0] for it in items}
            for fut in as_completed(futures):
                smi, vec = fut.result()
                out[smi] = vec
        return out
