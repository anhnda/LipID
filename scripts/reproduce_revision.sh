#!/usr/bin/env bash
# reproduce_revision.sh — LipID v1.2 revision pipeline (new experiments only; v1.1 steps stay in reproduce_main.sh).
#
#   bash scripts/reproduce_revision.sh                                   # P0 theory + JNT + P1 package (no STAT-01)
#   STEPS="lock_stat01" bash scripts/reproduce_revision.sh               # 1) lock the confirmatory protocol (clean tree!)
#   STEPS="stat01_nested stat01_pairs stat01_analyze jnt02_final" bash scripts/reproduce_revision.sh   # 2) run it
#   DEVICE=cpu WORKERS=8 bash scripts/reproduce_revision.sh
#
# Steps (default order):
#   tests_v12 jnt01 pairs_explore pairs_confirm jnt02 jnt03 met02 scl02_pilot scl02_eval scl02_analyze cert02 build_v12
# Confirmatory STAT-01 (run separately, after committing and locking):
#   lock_stat01 stat01_nested stat01_pairs stat01_analyze jnt02_final build_v12
# Release / verification:
#   verify  release  clean_env
# Every step logs to logs/v12_<step>.log; logs/v12_timeline.log records start times; set -e stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$($PY -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
fi
WORKERS="${WORKERS:-$($PY -c 'import os; print(max(1, (os.cpu_count() or 2) - 1))')}"
STEPS="${STEPS:-tests_v12 jnt01 pairs_explore pairs_confirm jnt02 jnt03 met02 scl02_pilot scl02_eval scl02_analyze cert02 build_v12}"
STAT_SEEDS="$($PY -c 'import json; print(" ".join(map(str, json.load(open("protocols/v12_preregistration.json"))["experiments"]["STAT-01"]["seeds"])))')"
LOCK="protocols/locks/STAT-01.lock.json"
mkdir -p logs runs results_v12 figures_v12 protocols/locks
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
has() { [[ " $STEPS " == *" $1 "* ]]; }
stamp() { echo "[$(date -Iseconds)] $*" | tee -a logs/v12_timeline.log; }
run() { local name="$1"; shift; stamp "$name"; "$@" 2>&1 | tee "logs/v12_${name}.log"; }

stamp "start DEVICE=$DEVICE WORKERS=$WORKERS STEPS='$STEPS' commit=$(git rev-parse HEAD 2>/dev/null || echo none) dirty=$(git status --porcelain 2>/dev/null | wc -l)"

has tests_v12      && run tests_v12 $PY -m pytest -q -rfEs tests/test_v12_pairwise.py tests/test_v12_joint.py \
                        tests/test_v12_mc_certdiag.py tests/test_v12_instances.py tests/test_v12_stats_lock.py \
                        --junitxml=logs/v12_pytest_junit.xml
if has jnt01; then
  # exit code 1 = pre-specified verdict NOT accepted; it is recorded, the pipeline (e.g. a clean-env rerun) continues
  run jnt01 $PY scripts/jnt01_validate_pairwise.py --workers "$WORKERS" \
    || stamp "jnt01 exited non-zero: verdict NOT accepted (results_v12/jnt01_manifest.json); recorded, continuing"
fi
has pairs_explore  && run pairs_explore $PY scripts/run_pairwise.py --seeds 1 2 3 4 --tag explore --out runs/pairs_explore --device "$DEVICE"
has pairs_confirm  && run pairs_confirm $PY scripts/run_pairwise.py --seeds 8 9 10 11 --tag confirm --out runs/pairs_confirm --device "$DEVICE"
has jnt02          && run jnt02 $PY scripts/jnt02_analyze.py --run explore=runs/pairs_explore --run confirm=runs/pairs_confirm --main confirm
has jnt03          && run jnt03 $PY scripts/jnt03_absolute_lp.py --device "$DEVICE"

if has lock_stat01; then
  if [[ -f "$LOCK" ]]; then stamp "lock exists: $LOCK (not overwritten)"; else
    run lock_stat01 $PY scripts/lock_protocol.py lock --experiment STAT-01 --out "$LOCK"
    stamp "NEXT: copy $LOCK to the machine that can push, commit + push it, then on this server:"
    stamp "      mv $LOCK $LOCK.server && git pull && cmp $LOCK $LOCK.server && rm $LOCK.server"
    stamp "      run stat01_* only after the pushed lock commit exists on the remote (its timestamp precedes outputs)"
  fi
fi
if has stat01_nested; then
  $PY scripts/lock_protocol.py verify --lock "$LOCK"
  # shellcheck disable=SC2086
  run stat01_nested $PY scripts/run_nested.py --seeds $STAT_SEEDS --tag stat01 --out runs/stat01 --device "$DEVICE"
fi
# shellcheck disable=SC2086
has stat01_pairs   && run stat01_pairs $PY scripts/run_pairwise.py --seeds $STAT_SEEDS --tag stat01 --out runs/pairs_stat01 \
                        --device "$DEVICE" --require-lock "$LOCK"
has stat01_analyze && run stat01_analyze $PY scripts/stat01_analyze.py --partition stat01:runs/stat01:runs/pairs_stat01 \
                        --partition confirm:runs/confirm:runs/pairs_confirm --partition explore:runs/explore:runs/pairs_explore \
                        --lock "$LOCK"
has jnt02_final    && run jnt02_final $PY scripts/jnt02_analyze.py --run explore=runs/pairs_explore \
                        --run confirm=runs/pairs_confirm --run stat01=runs/pairs_stat01 --main stat01

has met02          && run met02 $PY scripts/met02_scale_sensitivity.py --device "$DEVICE"
if has scl02_pilot; then
  if [[ -f results_v12/scl02_pilot.json ]]; then stamp "scl02 pilot exists (run once): kept"; else
    run scl02_pilot $PY scripts/scl02_nonsaturated.py pilot --device "$DEVICE"; fi
fi
has scl02_eval     && run scl02_eval $PY scripts/scl02_nonsaturated.py eval --device "$DEVICE"
has scl02_analyze  && run scl02_analyze $PY scripts/scl02_nonsaturated.py analyze
has cert02         && run cert02 $PY scripts/cert02_diagnostics.py --device "$DEVICE"
has build_v12      && run build_v12 $PY scripts/build_revision.py
# order matters: the clean-environment rerun first (it writes the verification report with its scope),
# then the on-disk integrity check, then packaging
has clean_env      && run clean_env bash scripts/clean_env_reproduce.sh
has verify         && run verify $PY scripts/verify_release.py --check-manifests --out results_v12/integrity \
                        --scope "${VERIFY_SCOPE:-on-disk integrity only (manifests and checksums); no independent rerun in this step}"
has release        && run release bash scripts/make_release.sh
stamp "done"