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
    ap.add_argument("--max-part-bytes", type=int, default=0,
                    help="split archives larger than this into .part-aa, .part-ab, … "
                         "(0 = never split). Zenodo shapes each connection to well "
                         "under 1 MB/s, and one file is one HTTP PUT, so a single "
                         "large archive cannot be parallelised; splitting it lets "
                         "several parts upload at once and bounds what a failed "
                         "transfer costs. Readers reassemble with `cat …part-* > x.tar`. "
                         "Suggested: 1000000000 (1 GB).")
    args = ap.parse_args()

    root = Path(args.weights).resolve()
    tree = root / "tetrafuse"
    if not tree.is_dir():
        print(f"ERROR: {tree} not found — unpack or export the weights first",
              file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []            # files that exist in `out` — these go into SHA256SUMS
    reassembled_rows = []  # split archives: manifest-only, never present as one file
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
        size = archive.stat().st_size

        if args.max_part_bytes and size > args.max_part_bytes:
            # `split -b` with a numeric-free suffix so `cat *.part-*` reassembles in
            # lexical order. The whole-archive sha256 stays in the manifest as well,
            # so a reader can verify the reassembled file, not just the parts.
            whole_digest = sha256_of(archive)
            prefix = f"{archive.name}.part-"
            existing = sorted(out.glob(prefix + "*"))
            if not existing or args.force:
                for old in existing:
                    old.unlink()
                subprocess.run(["split", "-b", str(args.max_part_bytes),
                                str(archive), str(out / prefix)], check=True)
                existing = sorted(out.glob(prefix + "*"))
            print(f"  split into {len(existing)} parts of <= "
                  f"{args.max_part_bytes/1e9:.2f} GB "
                  f"(reassemble: cat {prefix}* > {archive.name})")
            for part in existing:
                rows.append({"filename": part.name, "bytes": part.stat().st_size,
                             "sha256": sha256_of(part), "n_files": 0})
            # The whole-archive digest belongs in the manifest (a reader verifies the
            # file they reassemble) but NOT in SHA256SUMS, which `sha256sum -c` runs
            # against files actually present in the upload directory.
            reassembled_rows.append({"filename": archive.name, "bytes": size,
                                     "sha256": whole_digest, "n_files": n_files})
            continue

        digest = sha256_of(archive)
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
        # after the uploadable files, the digest of each reassembled whole archive
        w.writerows(reassembled_rows)
    if reassembled_rows:
        names = ", ".join(r["filename"] for r in reassembled_rows)
        with open(out / "REASSEMBLE.txt", "w") as fh:
            fh.write("Some archives were split so they could be uploaded in "
                     "parallel.\nReassemble before extracting:\n\n")
            for r in reassembled_rows:
                fh.write(f"  cat {r['filename']}.part-* > {r['filename']}\n"
                         f"  sha256sum {r['filename']}   # expect "
                         f"{r['sha256']}\n\n")
            fh.write("Then extract as usual:\n  tar -xf <archive> -C weights/\n")
        print(f"\nsplit archives ({names}) — see REASSEMBLE.txt; "
              f"whole-file digests are in zenodo_manifest.csv")

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
