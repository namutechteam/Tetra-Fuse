"""Uni-Mol2 LMDB feature dict builder (LoRA 학습용).

옛 ``scripts-old/precompute_unimol2_features.py`` 를 신규 builder 시스템으로 재구성.
frozen CLS embedding parquet (``unimol2.py``) 과 달리 이 builder 는
``mol2unimolv2()`` 가 반환하는 feature dict (atom_feat, edge_feat, coords, attn_bias 등)
를 LMDB 에 저장한다 — LoRA fine-tuning 시 매 forward 에서 ETKDG 재실행을 피하기 위함.

출력:
    - ``.lmdb``:  key = smiles.encode('utf-8'), value = pickle.dumps(feature_dict)
    - ``<stem>_status.parquet``: smiles, parse_ok, etkdg_ok, featurize_ok
      (분석/디버깅 전용 — 모델 입력 아님)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import signal
from pathlib import Path
from typing import Iterable

import lmdb
import pandas as pd
from tqdm import tqdm

from qadmet.features.registry import FEATURE_BUILDERS

logger = logging.getLogger(__name__)

_LMDB_MAP_SIZE_GB = 20
_KEY_MAX_BYTES = 480  # LMDB 실제 limit ~511 — 여유분 확보
_DEFAULT_BATCH_SIZE = 512
_DEFAULT_CHECKPOINT_EVERY = 1000


def _featurize_one(
    smi_idx: tuple[int, str],
) -> tuple[int, str, object, bool, bool, bool]:
    """Worker: SMILES → (idx, smi, feat_or_None, parse_ok, etkdg_ok, featurize_ok)."""
    idx, smi = smi_idx
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from qadmet.modules.backbones._vendor import unimol_v2 as uv2

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return idx, smi, None, False, False, False

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 30
        if AllChem.EmbedMolecule(mol, params) < 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) < 0:
                return idx, smi, None, True, False, False

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass

        try:
            feat = uv2.mol2unimolv2(mol)
            return idx, smi, feat, True, True, True
        except Exception:
            return idx, smi, None, True, True, False

    except Exception:
        return idx, smi, None, False, False, False


def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@FEATURE_BUILDERS.register("unimol2_lmdb")
class UniMol2LmdbBuilder:
    """Uni-Mol2 LMDB feature dict builder (LoRA 학습용).

    ``PrecomputeBase`` 를 상속하지 않는다 — 출력이 parquet 이 아니라 LMDB 이므로.
    ``run(smiles, output_path, *, resume=True)`` 인터페이스는 runner 와 호환.
    """

    name = "unimol2_lmdb"
    output_format = "lmdb"
    default_output_relpath = "data/embeddings/unimol2_features.lmdb"

    def __init__(
        self,
        n_workers: int = 8,
        map_size_gb: int = _LMDB_MAP_SIZE_GB,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        self.n_workers = n_workers
        self.map_size_gb = map_size_gb
        self.checkpoint_every = checkpoint_every

    # ----------------------------------------------------------------- public

    def run(
        self,
        smiles: Iterable[str],
        output_path: str | Path,
        *,
        resume: bool = True,
    ) -> Path:
        """SMILES 전체 → LMDB + status sidecar 저장.

        Returns:
            Path to the .lmdb file.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        status_path = out_path.with_name(out_path.stem + "_status.parquet")

        smiles_list = sorted(set(smiles))

        if resume and out_path.exists():
            done = self._existing_smiles(out_path)
            before = len(smiles_list)
            smiles_list = [s for s in smiles_list if s not in done]
            logger.info(
                "resume: %d already done, %d remaining", before - len(smiles_list), len(smiles_list)
            )

        if not smiles_list:
            logger.info("nothing to compute — exit.")
            return out_path

        # LMDB key 크기 초과 SMILES 사전 필터
        ok_smiles: list[str] = []
        skipped: list[str] = []
        for s in smiles_list:
            if len(s.encode("utf-8")) > _KEY_MAX_BYTES:
                skipped.append(s)
                logger.warning("skip SMILES >%d bytes: %s…", _KEY_MAX_BYTES, s[:60])
            else:
                ok_smiles.append(s)
        if skipped:
            logger.info("skipped %d SMILES due to key size limit", len(skipped))

        logger.info(
            "computing unimol2_lmdb for %d SMILES (n_workers=%d)", len(ok_smiles), self.n_workers
        )

        status_rows: list[dict] = []
        env = lmdb.open(
            str(out_path),
            map_size=self.map_size_gb * 1024**3,
            subdir=False,
            lock=False,
        )

        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.n_workers, initializer=_init_worker) as pool:
                it = pool.imap_unordered(
                    _featurize_one,
                    list(enumerate(ok_smiles)),
                    chunksize=8,
                )
                pending: list[tuple[str, bytes]] = []

                def _flush_lmdb() -> None:
                    if not pending:
                        return
                    with env.begin(write=True) as txn:
                        for key_str, val in pending:
                            txn.put(key_str.encode("utf-8"), val)
                    pending.clear()

                with tqdm(total=len(ok_smiles), desc="unimol2_lmdb") as pbar:
                    for _gi, smi, feat, parse_ok, etkdg_ok, featurize_ok in it:
                        pbar.update(1)
                        status_rows.append(
                            {
                                "smiles": smi,
                                "parse_ok": parse_ok,
                                "etkdg_ok": etkdg_ok,
                                "featurize_ok": featurize_ok,
                            }
                        )
                        if feat is not None:
                            pending.append((smi, pickle.dumps(feat)))
                        if len(pending) >= self.checkpoint_every:
                            _flush_lmdb()
                    _flush_lmdb()
        finally:
            env.close()

        self._write_status(status_path, status_rows, skipped)
        return out_path

    # --------------------------------------------------------------- private

    @staticmethod
    def _existing_smiles(lmdb_path: Path) -> set[str]:
        """기존 LMDB 에서 이미 저장된 SMILES 키를 읽어 resume 에 사용."""
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
        try:
            with env.begin() as txn:
                return {k.decode("utf-8") for k, _ in txn.cursor()}
        finally:
            env.close()

    @staticmethod
    def _write_status(
        status_path: Path,
        rows: list[dict],
        skipped: list[str],
    ) -> None:
        """status sidecar parquet 기록 (기존 있으면 append + dedupe)."""
        for s in skipped:
            rows.append(
                {"smiles": s, "parse_ok": False, "etkdg_ok": False, "featurize_ok": False}
            )
        df_new = pd.DataFrame(
            rows,
            columns=["smiles", "parse_ok", "etkdg_ok", "featurize_ok"],
        )
        if status_path.exists():
            df_old = pd.read_parquet(status_path)
            df = (
                pd.concat([df_old, df_new], ignore_index=True)
                .drop_duplicates(subset="smiles", keep="last")
                .reset_index(drop=True)
            )
        else:
            df = df_new
        df.to_parquet(status_path, index=False)
        n_fail = (~df["featurize_ok"]).sum()
        logger.info(
            "status sidecar: %d rows, %d failures (%.1f%%) → %s",
            len(df),
            n_fail,
            100.0 * n_fail / max(1, len(df)),
            status_path,
        )
