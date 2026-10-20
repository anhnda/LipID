#!/usr/bin/env bash
# clean_env_reproduce.sh — REP-01: rerun a declared subset of the pipeline in a FRESH environment and compare.
#
#   bash scripts/clean_env_reproduce.sh                       # fresh venv + copy of the committed tree, CPU
#   REPRO_STEPS="tests jnt01 metric real" bash scripts/clean_env_reproduce.sh
#   USE_DOCKER=1 bash scripts/clean_env_reproduce.sh          # same inside the Dockerfile image (no host packages)
#
# 1. exports the COMMITTED tree (git archive HEAD) into $WORK (default repro_check/src) — uncommitted edits are excluded
# 2. creates a fresh virtualenv ($WORK/venv) and installs requirements.txt (or Docker image lipid-v12)
# 3. runs the requested steps there (scripts/reproduce_main.sh and scripts/reproduce_revision.sh step names)
# 4. compares regenerated tables with the archived ones (scripts/verify_release.py) and writes the scope to the report
# The full P0'' replay (explore/confirm) is long; it is included only if listed in REPRO_STEPS.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WORK="${WORK:-$ROOT/repro_check}"
REPRO_STEPS="${REPRO_STEPS:-tests fig2 metric real tests_v12 jnt01 jnt01_diag}"
DEVICE="${DEVICE:-cpu}"
mkdir -p "$WORK"
rm -rf "$WORK/src"
mkdir -p "$WORK/src"
git archive HEAD | tar -x -C "$WORK/src"
COMMIT="$(git rev-parse HEAD)"
echo "[clean-env] commit $COMMIT -> $WORK/src ; steps: $REPRO_STEPS"

MAIN_STEPS=""; REV_STEPS=""
for s in $REPRO_STEPS; do
  case "$s" in
    env|tests|explore|confirm|fig2|metric|scaling|real|build|checksums) MAIN_STEPS="$MAIN_STEPS $s" ;;
    *) REV_STEPS="$REV_STEPS $s" ;;
  esac
done

if [[ "${USE_DOCKER:-0}" == "1" ]]; then
  docker build -t lipid-v12 -f "$WORK/src/Dockerfile" "$WORK/src"
  docker run --rm -v "$WORK/src:/work" -w /work -e DEVICE="$DEVICE" lipid-v12 bash -lc \
    "git init -q && git add -A && git -c user.email=r@x -c user.name=repro commit -qm repro >/dev/null; \
     ${MAIN_STEPS:+STEPS='$MAIN_STEPS' bash scripts/reproduce_main.sh;} \
     ${REV_STEPS:+STEPS='$REV_STEPS' bash scripts/reproduce_revision.sh;} true"
else
  python3 -m venv "$WORK/venv"
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  if [[ -f "$WORK/src/requirements-lock.txt" ]]; then
    python -m pip install -r "$WORK/src/requirements-lock.txt"
  else
    python -m pip install -r "$WORK/src/requirements.txt"
  fi
  (cd "$WORK/src" && git init -q && git add -A && git -c user.email=r@x -c user.name=repro commit -qm repro >/dev/null)
  if [[ -n "$MAIN_STEPS" ]]; then (cd "$WORK/src" && STEPS="$MAIN_STEPS" DEVICE="$DEVICE" bash scripts/reproduce_main.sh); fi
  if [[ -n "$REV_STEPS" ]]; then (cd "$WORK/src" && STEPS="$REV_STEPS" DEVICE="$DEVICE" bash scripts/reproduce_revision.sh); fi
  deactivate || true
fi

python scripts/verify_release.py --check-manifests \
  --compare results "$WORK/src/results" \
  --compare results_v12 "$WORK/src/results_v12" \
  --scope "clean environment ($([[ "${USE_DOCKER:-0}" == "1" ]] && echo Docker || echo venv), device $DEVICE), commit $COMMIT, steps: $REPRO_STEPS"
