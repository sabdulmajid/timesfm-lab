#!/usr/bin/env bash
set -euo pipefail

# Event-driven seed-42 production evaluation. This waits on the exact matched-training
# scheduler PID, rather than polling GPU/process state, and then keeps both GPUs busy.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MATCHED_TRAINING_SCHEDULER_PID" >&2
  exit 2
fi

training_pid="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export PYTHONPATH=src
config=configs/distillation/production_1m.yaml
plan=results/reproduction/distillation/production-1m-corpus-plan.json
pretrain_data=data/gift-pretrain-production
teacher_cache=teacher_cache/production-1m
gift_data=data/gift-eval-full
raw=results/raw
distillation=results/reproduction/distillation
systems=results/reproduction/systems
mkdir -p "$raw" "$distillation" "$systems" results/reproduction/gift_short_full

if kill -0 "$training_pid" 2>/dev/null; then
  echo "waiting for matched training scheduler PID $training_pid"
  tail --pid="$training_pid" -f /dev/null
fi

for variant in gt kd dual_view cvrd; do
  result="$distillation/production-1m-student-$variant-seed42.json"
  jq -e '.status == "succeeded" and .extra.training.steps == 200000' "$result" >/dev/null
  test -s "checkpoints/production-1m/$variant/student-$variant-best.pt"
done
echo "matched 200k training verified; starting response diagnostics"

run_diagnostics() {
  local gpu="$1"
  shift
  local variant
  for variant in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/diagnose_production_responses.py \
      "$config" "$plan" \
      --data-root "$pretrain_data" \
      --cache-root "$teacher_cache" \
      --student-checkpoint "checkpoints/production-1m/$variant/student-$variant-best.pt" \
      --student-label "$variant" \
      --student-batch-size 32 \
      --output "$distillation/production-1m-student-$variant-response-diagnostics.json" \
      >"$raw/production-1m-student-$variant-response-diagnostics.log" 2>&1
  done
}

run_diagnostics 0 gt dual_view &
diagnostics_gpu0=$!
run_diagnostics 1 kd cvrd &
diagnostics_gpu1=$!
wait "$diagnostics_gpu0"
wait "$diagnostics_gpu1"
echo "response diagnostics complete; starting 19-configuration evaluations"

run_student_scope() {
  local gpu="$1"
  local scope_config="$2"
  local scope_label="$3"
  shift 3
  local variant mode
  for variant in "$@"; do
    for mode in multivariate univariate; do
      CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/evaluate_student.py \
        "$config" "$scope_config" \
        --checkpoint "checkpoints/production-1m/$variant/student-$variant-best.pt" \
        --variant "$variant" \
        --mode "$mode" \
        --data-root "$gift_data" \
        --output "$distillation/production-1m-student-$variant-$mode-$scope_label-seed42.json" \
        >"$raw/production-1m-student-$variant-$mode-$scope_label-seed42.log" 2>&1
    done
  done
}

run_student_scope 0 configs/reproduction/multivariate_short_full.yaml gift-mv19 gt dual_view &
gift19_gpu0=$!
run_student_scope 1 configs/reproduction/multivariate_short_full.yaml gift-mv19 kd cvrd &
gift19_gpu1=$!
wait "$gift19_gpu0"
wait "$gift19_gpu1"
echo "19-configuration evaluations complete; starting full short-horizon teacher reference"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_gift_quickscan.py \
  configs/reproduction/gift_short_full.yaml \
  --data-root "$gift_data" \
  --mode multivariate \
  --output results/reproduction/gift_short_full/timesfm3-gift-short-full-multivariate-seed42.json \
  >"$raw/timesfm3-gift-short-full-multivariate-seed42.log" 2>&1 &
teacher_mv=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_gift_quickscan.py \
  configs/reproduction/gift_short_full.yaml \
  --data-root "$gift_data" \
  --mode univariate \
  --output results/reproduction/gift_short_full/timesfm3-gift-short-full-univariate-seed42.json \
  >"$raw/timesfm3-gift-short-full-univariate-seed42.log" 2>&1 &
teacher_uv=$!
wait "$teacher_mv"
wait "$teacher_uv"
echo "teacher full short-horizon reference complete; starting student full scope"

run_student_scope 0 configs/reproduction/gift_short_full.yaml gift-short55 gt dual_view &
gift55_gpu0=$!
run_student_scope 1 configs/reproduction/gift_short_full.yaml gift-short55 kd cvrd &
gift55_gpu1=$!
wait "$gift55_gpu0"
wait "$gift55_gpu1"
echo "full short-horizon student evaluations complete; starting inference benchmarks"

run_benchmarks() {
  local gpu="$1"
  shift
  local variant
  for variant in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/benchmark_student.py \
      "$config" configs/systems/teacher_reference_blackwell.yaml \
      --checkpoint "checkpoints/production-1m/$variant/student-$variant-best.pt" \
      --variant "$variant" \
      --output "$systems/production-1m-student-$variant-blackwell.json" \
      >"$raw/production-1m-student-$variant-blackwell.log" 2>&1
  done
}

run_benchmarks 0 gt dual_view &
benchmark_gpu0=$!
run_benchmarks 1 kd cvrd &
benchmark_gpu1=$!
wait "$benchmark_gpu0"
wait "$benchmark_gpu1"

comparison_results=(
  results/reproduction/mv_short/timesfm3-multivariate-short-full-multivariate-seed42.json
  results/reproduction/mv_short/timesfm3-multivariate-short-full-univariate-seed42.json
  results/reproduction/gift_short_full/timesfm3-gift-short-full-multivariate-seed42.json
  results/reproduction/gift_short_full/timesfm3-gift-short-full-univariate-seed42.json
)
for scope_label in gift-mv19 gift-short55; do
  for variant in gt kd dual_view cvrd; do
    for mode in multivariate univariate; do
      comparison_results+=(
        "$distillation/production-1m-student-$variant-$mode-$scope_label-seed42.json"
      )
    done
  done
done
comparison_args=()
for result in "${comparison_results[@]}"; do
  comparison_args+=(--result "$result")
done
.venv/bin/python scripts/summarize_gift_comparison.py \
  "${comparison_args[@]}" \
  --output "$distillation/production-1m-gift-comparison-seed42.json" \
  >"$raw/production-1m-gift-comparison-seed42.log" 2>&1

echo "seed-42 production post-training pipeline complete"
