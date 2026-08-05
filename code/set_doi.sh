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

# 2) weights/README.md — give the record URL directly rather than sending the
#    reader via CITATION.cff, since this file is where they are standing
sed -i "s|The Zenodo record (DOI in the top-level \`CITATION.cff\`) holds|The Zenodo record <https://zenodo.org/records/${REC}> (DOI ${DOI}) holds|" \
    weights/README.md

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
