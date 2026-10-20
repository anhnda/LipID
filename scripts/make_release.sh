#!/usr/bin/env bash
# make_release.sh — REP-01: package the public artifact (code at a fixed commit + configs + outputs + checksums).
#
#   bash scripts/make_release.sh                  # code, configs, protocols, locks, results, figures, logs, run summaries
#   INCLUDE_RAW=1 bash scripts/make_release.sh    # + row-level parquet tables (records, diag, pairs, inst) — large
#
# Refuses to run on a dirty tree. Output: release/lipid_v1.2_<commit>.tar.gz and release/SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then   # untracked outputs are expected and packaged
  echo "tracked files modified: the code in the release must equal one commit"; git status --porcelain --untracked-files=no; exit 1
fi
COMMIT="$(git rev-parse --short=12 HEAD)"
NAME="lipid_v1.2_${COMMIT}"
STAGE="release/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE"
git archive HEAD | tar -x -C "$STAGE"                         # exactly the committed code
for d in results results_v12 figures figures_v12 logs protocols; do
  [[ -d "$d" ]] && tar -cf - --exclude 'matched' --exclude 'scaling_cache' "$d" | tar -xf - -C "$STAGE" || true
done
mkdir -p "$STAGE/runs"
for r in runs/*/; do
  [[ -d "$r" ]] || continue
  t="$STAGE/$r"; mkdir -p "$t"
  find "$r" -maxdepth 1 -type f \( -name '*.json' -o -name '*.yaml' -o -name '*.csv' \) -exec cp {} "$t" \;
  if [[ "${INCLUDE_RAW:-0}" == "1" ]]; then
    find "$r" -maxdepth 1 -type f \( -name '*.parquet' -o -name '*.csv.gz' \) -exec cp {} "$t" \;
  fi
done
python -m pip freeze > "$STAGE/pip_freeze.txt" 2>/dev/null || true
(cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "release/$NAME.tar.gz" -C release "$NAME"
(cd release && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "wrote release/$NAME.tar.gz ($(du -h "release/$NAME.tar.gz" | cut -f1)); checksums in $STAGE/SHA256SUMS"
echo "Upload the tarball to an archival host (e.g. Zenodo) and put the DOI + commit in the paper and the README."