#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python tests/test_compact.py
if [ ! -f results/smoke/PASSED.json ]; then python tests/smoke_v2.py; fi
if [ ! -f results/nuevos_experimentos/COMPLETED.json ]; then python scripts/run_v2.py; fi
python scripts/analyze.py
python scripts/benchmark.py
python scripts/figures.py
python scripts/render_article.py
python scripts/document_revision.py
python tests/test_delivery.py
