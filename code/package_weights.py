#!/usr/bin/env python
"""Package the released PEFT weights into per-member archives for Zenodo.

Zenodo accepts up to 50 GB and 100 files per record. The weight tree is 340 files
totalling ~11.3 GiB, so the files have to be archived — one archive per ensemble
member keeps each piece independently downloadable (a reader who only wants
CheMeleon·STL does not have to pull 11 GB).

Archives are uncompressed `.tar`: the payload is dense fp32 tensors, where gzip
buys about 8% at ~30x the wall time, and every archive ships with a SHA-256 anyway.

    python code/package_weights.py --weights weights/ --out dist/

Writes `dist/tetrafuse_weights_<member>.tar` plus `dist/zenodo_manifest.csv`
(filename, bytes, sha256, n_files) and `dist/SHA256SUMS` for `sha256sum -c`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

MEMBERS = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map"]

# Small files uploaded to Zenodo as-is (they are also tracked in git, so a reader
# can inspect the layout and verify checksums before downloading 11 GB).
METADATA = ["README.md", "weights_manifest.csv", "target_scalers.json",
            "checkpoint_map.json"]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights",
                    help="directory holding tetrafuse/ and the metadata files")
    ap.add_argument("--out", default="dist", help="where to write the archives")
    ap.add_argument("--members", nargs="+", default=MEMBERS,
                    help=f"subset to archive (default: all of {', '.join(MEMBERS)}). "
                         "Note that a partial run still rewrites the manifest, so it "
                         "will describe only the members you passed.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild archives that already exist (default: reuse them, "
                         "since re-taring 11 GiB is slow and the content is identical)")
    args = ap.parse_args()

    root = Path(args.weights).resolve()
    tree = root / "tetrafuse"
    if not tree.is_dir():
        print(f"ERROR: {tree} not found — unpack or export the weights first",
              file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for member in args.members:
        src = tree / member
        if not src.is_dir():
            print(f"SKIP {member}: {src} missing")
            continue
        n_files = sum(1 for _ in src.rglob("peft.pt"))
        archive = out / f"tetrafuse_weights_{member}.tar"
        # Paths inside the archive are `tetrafuse/<member>/...`, so extracting
        # from weights/ recreates exactly the layout weights/README.md documents.
        if archive.exists() and not args.force:
            print(f"reusing existing {archive.name} (pass --force to rebuild)")
        else:
            subprocess.run(["tar", "-cf", str(archive), "-C", str(root),
                            f"tetrafuse/{member}"], check=True)
        digest = sha256_of(archive)
        size = archive.stat().st_size
        rows.append({"filename": archive.name, "bytes": size,
                     "sha256": digest, "n_files": n_files})
        print(f"{archive.name:44} {size/2**30:6.2f} GiB  {n_files:3} files  {digest[:16]}…")

    # Metadata files are copied into the output dir alongside the archives, so the
    # upload set is self-contained and `sha256sum -c SHA256SUMS` runs there.
    for name in METADATA:
        src = root / name
        if not src.exists():
            print(f"SKIP metadata {name}: not found")
            continue
        dst = out / name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        digest = sha256_of(dst)
        rows.append({"filename": name, "bytes": dst.stat().st_size,
                     "sha256": digest, "n_files": 1})
        print(f"{name:44} {dst.stat().st_size/1024:6.0f} KiB    1 file   {digest[:16]}…")

    manifest = out / "zenodo_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "bytes", "sha256", "n_files"])
        w.writeheader()
        w.writerows(rows)

    # `sha256sum -c SHA256SUMS` after downloading, from the directory holding the files
    with open(out / "SHA256SUMS", "w") as fh:
        for r in rows:
            fh.write(f"{r['sha256']}  {r['filename']}\n")

    total = sum(r["bytes"] for r in rows)
    print(f"\n{len(rows)} files, {total/2**30:.2f} GiB -> {out}")
    print(f"Zenodo limits: 100 files / 50 GB per record — "
          f"{'within' if len(rows) <= 100 and total < 50e9 else 'OVER'} limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
