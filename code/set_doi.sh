#!/usr/bin/env bash
# Fill the Zenodo DOI into the release once the record is published.
#
#   bash code/set_doi.sh 10.5281/zenodo.1234567
#
# Publishing on Zenodo mints the DOI, so this cannot run earlier. It edits the
# two places a reader looks — the citation metadata and the weights README —
# then shows a diff for you to check before committing.
set -euo pipefail

DOI="${1:-}"
if [[ -z "$DOI" ]]; then
    echo "usage: bash code/set_doi.sh 10.5281/zenodo.XXXXXXX" >&2
    exit 2
fi
if [[ ! "$DOI" =~ ^10\.5281/zenodo\.[0-9]+$ ]]; then
    echo "ERROR: '$DOI' does not look like a Zenodo DOI (10.5281/zenodo.NNNNNNN)" >&2
    exit 2
fi
REC="${DOI##*.}"                       # record id = digits after the last dot
URL="https://doi.org/${DOI}"

cd "$(dirname "$0")/.."                # repository root

# 1) CITATION.cff — uncomment the placeholder line and set the real DOI
if grep -q '^# doi: 10\.5281/zenodo\.XXXXXXX' CITATION.cff; then
    sed -i "s|^# doi: 10\.5281/zenodo\.XXXXXXX.*|doi: ${DOI}|" CITATION.cff
elif grep -qE '^doi: 10\.5281/zenodo\.[0-9]+' CITATION.cff; then
    sed -i "s|^doi: 10\.5281/zenodo\.[0-9]*|doi: ${DOI}|" CITATION.cff
else
    echo "WARNING: no DOI line found in CITATION.cff — add 'doi: ${DOI}' by hand" >&2
fi

# 2) weights/README.md — state the DOI where the reader is standing, and drop the
#    "being uploaded / may 404" caveat that only applied before publication
python3 - "$DOI" "$REC" <<'PY'
import sys
doi, rec = sys.argv[1], sys.argv[2]
p = "weights/README.md"
lines = open(p).read().split("\n")

# Replace the whole paragraph that starts with "The Zenodo record" — from that
# line up to the following blank line. Paragraph-scoped rather than
# sentence-scoped, so it also drops the pre-publication "may 404" caveat, and
# re-running it (a new version DOI, say) is harmless.
start = next((i for i, l in enumerate(lines)
              if l.startswith("The Zenodo record")), None)
if start is None:
    sys.exit("ERROR: no line starting 'The Zenodo record' in weights/README.md "
             "— edit it by hand")
end = start
while end < len(lines) and lines[end].strip():
    end += 1

lines[start:end] = [
    f"The Zenodo record <https://zenodo.org/records/{rec}> (DOI {doi}) holds one",
    "archive per ensemble member, so you can take only the members you need:",
]
open(p, "w").write("\n".join(lines))
PY

echo "=== diff ==="
git --no-pager diff -- CITATION.cff weights/README.md
cat <<EOF

Record:  https://zenodo.org/records/${REC}
DOI URL: ${URL}

Check the diff, then:
  git add CITATION.cff weights/README.md
  git commit -m "Add Zenodo DOI ${DOI} for the released weights"
  git push

Still to do by hand — the manuscript's Code Availability section:
  it currently defers the DOI to "upon publication" and does not mention the
  model weights at all. See the replacement text in the release notes.
EOF
