#!/usr/bin/env bash
set -euo pipefail
study_container="${RESEARCH_CONTAINER:-dlinear20261029-research-1}"
study_stage="${1:-all}"
docker --context default start "$study_container" >/dev/null
docker --context default exec -w /workspace/ampliacion_20261110 "$study_container" python scripts/check_running.py
exec docker --context default exec -w /workspace/ampliacion_20261110 "$study_container" python scripts/run_experiments.py --stage "$study_stage"
