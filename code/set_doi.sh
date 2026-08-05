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
import re, sys
doi, rec = sys.argv[1], sys.argv[2]
p = "weights/README.md"
s = open(p).read()
new = (f"The Zenodo record <https://zenodo.org/records/{rec}> (DOI {doi}) holds "
       "one archive per\nensemble member, so you can take only the members you "
       "need:")
# matches both the pre-publication wording and an already-filled one, so the
# script is safe to re-run (e.g. after minting a new version DOI)
pat = re.compile(
    r"The Zenodo record[^\n]*\n(?:[^\n]*\n)*?"
    r"(?:need\.|need:)(?:[^\n]*\n)*?(?=\n\| Archive)", re.M)
s2, n = pat.subn(new + "\n", s, count=1)
if n != 1:
    sys.exit("ERROR: could not locate the Zenodo record paragraph in "
             "weights/README.md — edit it by hand")
open(p, "w").write(s2)
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
