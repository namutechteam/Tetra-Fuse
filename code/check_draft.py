#!/usr/bin/env python
"""Query the Zenodo draft and compare its file list to the local dist directory.
Reads ZENODO_TOKEN from the environment. Prints what is present / missing / size-mismatched."""
import os, sys, hashlib
from pathlib import Path
import requests

DEP = "21802319"
DIST = Path("/nas/tk_ahn/tetrafuse_dist")
tok = os.environ.get("ZENODO_TOKEN")
if not tok:
    sys.exit("ERROR: set ZENODO_TOKEN")

# files we intended to upload (everything in dist except SHA256SUMS, which the
# uploader does not push, and zenodo_manifest.csv which is local bookkeeping)
SKIP_LOCAL = {"SHA256SUMS", "zenodo_manifest.csv"}  # match upload_zenodo.py --exclude default
local = {p.name: p.stat().st_size for p in sorted(DIST.iterdir())
         if p.is_file() and p.name not in SKIP_LOCAL}

r = requests.get(f"https://zenodo.org/api/deposit/depositions/{DEP}",
                 params={"access_token": tok}, timeout=60)
r.raise_for_status()
dep = r.json()
remote = {f["filename"]: f["filesize"] for f in dep.get("files", [])}

print(f"draft {DEP}: {dep.get('state')}  | remote files: {len(remote)}  local files: {len(local)}")
missing = [n for n in local if n not in remote]
mism = [(n, local[n], remote[n]) for n in local if n in remote and local[n] != remote[n]]
extra = [n for n in remote if n not in local]

print("\n== MISSING (local but not on Zenodo) ==")
print("\n".join("  "+n for n in missing) or "  none")
print("\n== SIZE MISMATCH ==")
print("\n".join(f"  {n}: local={l} remote={rr}" for n,l,rr in mism) or "  none")
print("\n== EXTRA on Zenodo (not in local dist) ==")
print("\n".join("  "+n for n in extra) or "  none")

ok = not missing and not mism
print("\n== VERDICT ==")
print("  ALL FILES PRESENT AND SIZES MATCH" if ok else "  INCOMPLETE — see above")
tot = sum(remote.values())
print(f"  remote total: {tot/1e9:.2f} GB across {len(remote)} files")
