#!/usr/bin/env bash
# reproduce_main.sh — full LipID v1.1 reproduction, from tests to every table / figure.
#
#   bash scripts/reproduce_main.sh                    # DEVICE defaults to cuda if available, else cpu
#   DEVICE=cpu bash scripts/reproduce_main.sh         # CPU-only reproduction (same inputs; slower)
#   STEPS="tests explore confirm build" bash scripts/reproduce_main.sh   # run a subset of steps
#
# Steps (in order): env tests explore confirm fig2 metric scaling real build checksums
# Optional:         legacy_explore  (re-runs the archived nested_anchors_v2.py on seeds 1-4 for a second crosswalk)
# Every step writes logs/<step>.log; a failing step stops the script (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PYTHON:-python}"
if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$($PY -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
fi
STEPS="${STEPS:-env tests explore confirm fig2 metric scaling real build checksums}"
mkdir -p logs runs results figures legacy
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

has() { [[ " $STEPS " == *" $1 "* ]]; }
stamp() { echo "[$(date -Iseconds)] $*" | tee -a logs/timeline.log; }

stamp "start DEVICE=$DEVICE STEPS='$STEPS' commit=$(git rev-parse HEAD 2>/dev/null || echo none) dirty=$(git status --porcelain 2>/dev/null | wc -l)"

if has env; then
  stamp "env"
  { $PY -c 'import sys,platform,numpy,scipy,torch,pandas,matplotlib;print(sys.version);print(platform.platform());
print("numpy",numpy.__version__,"scipy",scipy.__version__,"torch",torch.__version__,"cuda",torch.version.cuda,
      "cuda_available",torch.cuda.is_available(),"pandas",pandas.__version__,"matplotlib",matplotlib.__version__)
try:
    import pyarrow; print("pyarrow", pyarrow.__version__)
except Exception as e: print("pyarrow MISSING -> csv.gz fallback", e)
import sklearn; print("sklearn", sklearn.__version__)'
    $PY -m pip freeze 2>/dev/null || true
    (nvidia-smi 2>/dev/null || echo "no nvidia-smi")
    (lscpu 2>/dev/null || sysctl -a 2>/dev/null | grep machdep.cpu || true)
  } > logs/env.log 2>&1
fi

if has tests; then
  stamp "tests"
  $PY -m pytest -q -rfEs tests.py test_t6.py tests/ --junitxml=logs/pytest_junit.xml 2>&1 | tee logs/tests.log
fi

if has explore; then
  stamp "explore (seeds 1-4)"
  $PY scripts/run_nested.py --seeds 1 2 3 4 --tag explore --out runs/explore --device "$DEVICE" 2>&1 | tee logs/explore.log
fi

if has confirm; then
  stamp "confirm (seeds 8-11)"
  $PY scripts/run_nested.py --seeds 8 9 10 11 --tag confirm --out runs/confirm --device "$DEVICE" 2>&1 | tee logs/confirm.log
fi

if has legacy_explore; then
  stamp "legacy nested_anchors_v2 on seeds 1-4 (optional crosswalk)"
  $PY nested_anchors_v2.py --seeds 1 2 3 4 --out legacy/nested2_explore --device "$DEVICE" 2>&1 | tee logs/legacy_explore.log
  cp legacy/nested2_explore/nested2_summary.json legacy/nested2_summary_explore.json
fi

if has fig2; then
  stamp "figure 2"
  $PY scripts/fig2_counterexample.py 2>&1 | tee logs/fig2.log
fi

if has metric; then
  stamp "MET-01 metric sensitivity"
  $PY scripts/metric_sensitivity.py 2>&1 | tee logs/metric.log
fi

if has scaling; then
  stamp "SCALE-01 scaling benchmark"
  $PY scripts/benchmark_scaling.py --device "$DEVICE" 2>&1 | tee logs/scaling.log
fi

if has real; then
  stamp "REAL-01 real audit"
  $PY scripts/real_audit.py 2>&1 | tee logs/real.log
fi

if has build; then
  stamp "build results"
  $PY scripts/build_results.py --explore runs/explore --confirm runs/confirm \
      --legacy-confirm legacy/nested2_summary_confirm.json 2>&1 | tee logs/build.log
fi

if has checksums; then
  stamp "checksums"
  (cd "$ROOT" && find results figures -type f ! -path 'results/matched/*' ! -path 'results/scaling_cache/*' -print0 \
     | sort -z | xargs -0 sha256sum) > logs/checksums_results.txt
  (cd "$ROOT" && find runs -maxdepth 2 -type f -print0 | sort -z | xargs -0 sha256sum) > logs/checksums_runs.txt
fi
stamp "done"
