#!/usr/bin/env python
"""Upload the weight archives to Zenodo as a draft record.

Browser drag-and-drop is a poor fit for 11 GiB: a dropped connection restarts the
whole file. This uses Zenodo's bucket API, which streams one file per PUT, so a
failure costs you one archive rather than the batch — rerun and it skips what
already landed.

The script never publishes. It creates (or reuses) a draft, uploads, attaches the
metadata from `.zenodo.json`, and prints the draft URL for you to review and press
Publish yourself. Publishing mints the DOI, and a published record's files cannot
be swapped afterwards, so the last look should be a human's.

    export ZENODO_TOKEN=...          # Applications -> Personal access tokens,
                                     # scopes: deposit:write deposit:actions
    python code/upload_zenodo.py --files-dir dist/ --metadata .zenodo.json

    # rehearse against the sandbox first (separate account and token):
    python code/upload_zenodo.py --files-dir dist/ --sandbox --dry-run

Reruns are safe: pass `--deposition <id>` (printed on the first run) to resume,
and files already present with a matching size are skipped.

Expect this to take hours. Zenodo shapes a single connection to roughly
0.3 MB/s no matter how fast your link is (measured: 0.33 MB/s to Zenodo versus
9.5 MB/s to GitHub from the same host), so several files go up in parallel —
`--jobs` controls how many. Run it detached, because a closed terminal kills it:

    nohup python code/upload_zenodo.py --files-dir dist/ --jobs 4 \\
        > upload.log 2>&1 & disown
    tail -f upload.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Zenodo's documented per-record ceiling. Exceeding either needs a support
# request ("File upload quota increase"), which grants up to 200 GB one time.
MAX_FILES = 100
MAX_BYTES = 50_000_000_000


def api_base(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files-dir", required=True,
                    help="directory of files to upload (e.g. the packaging output)")
    ap.add_argument("--metadata", default=".zenodo.json",
                    help="Zenodo metadata JSON (title, creators, license, ...)")
    ap.add_argument("--deposition", help="resume an existing draft by id")
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org (needs a sandbox token)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate files and metadata, contact nothing")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel upload streams (default 4). Zenodo shapes each "
                         "connection to well under 1 MB/s, so concurrency — not "
                         "your link speed — sets the wall time. Only helps when "
                         "several files remain.")
    ap.add_argument("--exclude", nargs="*", default=["SHA256SUMS", "zenodo_manifest.csv"],
                    help="filenames to keep local (checksum lists are usually "
                         "redundant — Zenodo publishes its own per-file MD5)")
    args = ap.parse_args()

    src = Path(args.files_dir).resolve()
    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.name not in args.exclude)
    if not files:
        print(f"ERROR: no files to upload in {src}", file=sys.stderr)
        return 2

    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total/1e9:.2f} GB ({total/2**30:.2f} GiB) from {src}")
    for p in files:
        print(f"  {p.name:44} {p.stat().st_size/1e9:8.3f} GB")

    problems = []
    if len(files) > MAX_FILES:
        problems.append(f"{len(files)} files exceeds Zenodo's limit of {MAX_FILES}")
    if total > MAX_BYTES:
        problems.append(f"{total/1e9:.1f} GB exceeds the default {MAX_BYTES/1e9:.0f} GB "
                        f"quota — request an increase before uploading")
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        problems.append(f"metadata file {meta_path} not found")
    else:
        meta = json.loads(meta_path.read_text())
        for field in ("title", "upload_type", "description", "creators"):
            if not meta.get(field):
                problems.append(f"metadata is missing required field {field!r}")
    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return 2
    print(f"within Zenodo limits ({MAX_FILES} files / {MAX_BYTES/1e9:.0f} GB per record)")

    if args.dry_run:
        print("\n--dry-run: files and metadata validate; nothing was uploaded.")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN is not set. Create one under Zenodo -> "
              "Applications -> Personal access tokens with scopes "
              "deposit:write and deposit:actions.", file=sys.stderr)
        return 2

    import requests

    base = api_base(args.sandbox)
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    if args.deposition:
        r = session.get(f"{base}/deposit/depositions/{args.deposition}", timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"\nresuming draft {dep['id']}")
    else:
        r = session.post(f"{base}/deposit/depositions", json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"\ncreated draft {dep['id']} — pass --deposition {dep['id']} to resume")

    bucket = dep["links"]["bucket"]
    already = {f["filename"]: f.get("filesize") for f in dep.get("files", [])}

    pending = []
    for p in files:
        size = p.stat().st_size
        if already.get(p.name) == size:
            print(f"  skip {p.name} (already uploaded, {size/1e9:.3f} GB)")
        else:
            pending.append(p)
    if not pending:
        print("  all files already uploaded")

    def put_one(path):
        """One streaming PUT. Own Session per thread — requests Sessions are not
        thread-safe for concurrent use."""
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {token}"
        t0 = time.time()
        with open(path, "rb") as fh:
            # Streaming body: no multipart wrapper, no in-memory copy.
            r = s.put(f"{bucket}/{path.name}", data=fh, timeout=(60, 14400))
        return path, r, time.time() - t0

    # Zenodo shapes a single connection to well under 1 MB/s regardless of your
    # link, so wall time is set by concurrency, not bandwidth. Uploading several
    # files at once is the only lever that helps; it does nothing for a lone
    # large file, since one file is one PUT.
    failed = []
    if args.jobs > 1 and len(pending) > 1:
        print(f"  uploading {len(pending)} files with {args.jobs} parallel streams")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(put_one, p): p for p in pending}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    path, r, secs = fut.result()
                except Exception as exc:                      # noqa: BLE001
                    print(f"  FAILED {p.name}: {exc}", file=sys.stderr)
                    failed.append(p.name)
                    continue
                if not r.ok:
                    print(f"  FAILED {path.name}: {r.status_code} {r.text[:200]}",
                          file=sys.stderr)
                    failed.append(path.name)
                    continue
                mb = path.stat().st_size / 1e6
                print(f"  ok {path.name} ({mb/1024:.2f} GB in {secs/60:.1f} min, "
                      f"{mb/max(secs, 1):.2f} MB/s)  md5 {r.json().get('checksum', '')}",
                      flush=True)
    else:
        for p in pending:
            print(f"  uploading {p.name} ({p.stat().st_size/1e9:.3f} GB) …", flush=True)
            path, r, secs = put_one(p)
            if not r.ok:
                print(f"  FAILED {p.name}: {r.status_code} {r.text[:200]}",
                      file=sys.stderr)
                failed.append(p.name)
                continue
            mb = p.stat().st_size / 1e6
            print(f"    ok  ({mb/max(secs, 1):.2f} MB/s)  md5 "
                  f"{r.json().get('checksum', '')}", flush=True)

    if failed:
        print(f"\n{len(failed)} file(s) failed: {', '.join(failed)}", file=sys.stderr)
        print(f"Rerun to retry just those:\n"
              f"  python {sys.argv[0]} --files-dir {src} "
              f"--deposition {dep['id']}", file=sys.stderr)
        return 1

    r = session.put(f"{base}/deposit/depositions/{dep['id']}",
                    json={"metadata": meta}, timeout=60)
    if not r.ok:
        print(f"ERROR attaching metadata: {r.status_code} {r.text[:500]}",
              file=sys.stderr)
        return 1
    print("metadata attached")

    host = "sandbox.zenodo.org" if args.sandbox else "zenodo.org"
    print(f"\nDraft ready — NOT published:\n  https://{host}/uploads/{dep['id']}\n"
          "Review the files and metadata there, then press Publish to mint the DOI.\n"
          "Afterwards put that DOI into CITATION.cff and weights/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
