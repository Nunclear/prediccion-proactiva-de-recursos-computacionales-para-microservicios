#!/usr/bin/env bash
set -euo pipefail
study_container="${RESEARCH_CONTAINER:-dlinear20261029-research-1}"
docker --context default exec -w /workspace/ampliacion_20261110 "$study_container" python scripts/check_running.py
for study_script in analyze_results.py benchmark.py summarize_resources.py; do
  docker --context default exec -w /workspace/ampliacion_20261110 "$study_container" python "scripts/$study_script"
done
