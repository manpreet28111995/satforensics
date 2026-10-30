#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${1:-$ROOT/data}"
OUT="${2:-$ROOT/results}"
EPOCHS="${EPOCHS:-40}"
RUNS="${RUNS:-5}"
export PYTHONPATH="$ROOT"
mkdir -p "$DATA" "$OUT"

if [ ! -f "$DATA/meta.jsonl" ]; then
  python3 "$ROOT/scripts/collect_metadata.py" --out "$DATA/meta.jsonl" --windows 600 --pages 2 --workers 12
fi
if [ ! -f "$DATA/dataset.npz" ] && [ ! -d "$DATA/wf" ]; then
  python3 "$ROOT/scripts/fetch_waterfalls.py" --meta "$DATA/meta.jsonl" --out "$DATA/wf" --workers 8
fi
if [ ! -f "$DATA/dataset.npz" ]; then
  python3 "$ROOT/scripts/build_dataset.py" --meta "$DATA/meta.jsonl" --wf "$DATA/wf" --out "$DATA/dataset.npz"
fi
python3 "$ROOT/scripts/geometry_check.py" "$DATA/dataset.npz" "$DATA/meta.jsonl" "$OUT/geo_check.json"
python3 "$ROOT/scripts/data_summary.py" "$DATA/meta.jsonl" "$DATA/dataset.npz" "$DATA/dataset_stats.json" "$OUT/geo_check.json" "$OUT/data_macros.tex"

for stage in search main jnr cross temporal station calibration ledger latency; do
  python3 -W ignore "$ROOT/scripts/run_experiments.py" --data "$DATA/dataset.npz" --out "$OUT" --stage "$stage" --runs "$RUNS" --epochs "$EPOCHS"
done

python3 -W ignore "$ROOT/scripts/triage_rules.py" "$DATA/dataset.npz" "$OUT/triage_rules.json"
python3 -W ignore "$ROOT/scripts/make_tables.py" "$OUT" "$DATA/dataset_stats.json" "$OUT/tex"
mkdir -p "$OUT/figures"
python3 "$ROOT/figures/fig1_architecture.py" "$OUT/figures/Fig1_Architecture_Pipeline.pdf"
python3 "$ROOT/figures/make_figures.py" "$OUT" "$OUT/figures"
