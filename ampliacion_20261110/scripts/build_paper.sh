#!/usr/bin/env bash
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$study_root/articulo_final"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error revision_espanol.tex
