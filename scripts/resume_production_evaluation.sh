#!/usr/bin/env bash
set -euo pipefail

# Resume only the production evaluation stages that were not completed after the
# seed-42 post-training pipeline reached the 55-configuration teacher reference.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIRMATORY_SCHEDULER_PID_OR_0" >&2
  exit 2
fi

confirmatory_pid="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PYTHONPATH=src

config=configs/distillation/production_1m.yaml
scope55=configs/reproduction/gift_short_full.yaml
gift_data=data/gift-eval-full
distillation=results/reproduction/distillation
teacher55=results/reproduction/gift_short_full
systems=results/reproduction/systems
raw=results/raw
mkdir -p "$distillation" "$teacher55" "$systems" "$raw"

if [[ "$confirmatory_pid" != 0 ]] && kill -0 "$confirmatory_pid" 2>/dev/null; then
  echo "waiting for confirmatory scheduler PID $confirmatory_pid"
  tail --pid="$confirmatory_pid" -f /dev/null
fi

for seed in 43 44; do
  for variant in dual_view cvrd; do
    jq -e ".status == \"succeeded\" and .seed == $seed and .extra.training.steps == 200000 and .extra.validation_split_seed == 42" \
      "$distillation/production-1m-student-$variant-seed$seed.json" >/dev/null
    jq -e '.status == "succeeded" and .summary.student != null' \
      "$distillation/production-1m-student-$variant-response-diagnostics-seed$seed.json" >/dev/null
    for scope in gift-mv19 gift-short55; do
      for mode in multivariate univariate; do
        jq -e '.status == "succeeded" and (.extra.failures | length) == 0' \
          "$distillation/production-1m-student-$variant-$mode-$scope-seed$seed.json" >/dev/null
      done
    done
  done
done
echo "confirmatory training/evaluation artifacts verified; starting missing seed-42 stages"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_gift_quickscan.py \
  "$scope55" --data-root "$gift_data" --mode multivariate \
  --output "$teacher55/timesfm3-gift-short-full-multivariate-seed42.json" \
  >"$raw/timesfm3-gift-short-full-multivariate-seed42.log" 2>&1 &
teacher_mv=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_gift_quickscan.py \
  "$scope55" --data-root "$gift_data" --mode univariate \
  --output "$teacher55/timesfm3-gift-short-full-univariate-seed42.json" \
  >"$raw/timesfm3-gift-short-full-univariate-seed42.log" 2>&1 &
teacher_uv=$!
wait "$teacher_mv"
wait "$teacher_uv"

evaluate_seed42() {
  local gpu="$1"
  shift
  local variant mode
  for variant in "$@"; do
    for mode in multivariate univariate; do
      CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/evaluate_student.py \
        "$config" "$scope55" \
        --checkpoint "checkpoints/production-1m/$variant/student-$variant-best.pt" \
        --variant "$variant" --seed 42 --mode "$mode" --data-root "$gift_data" \
        --output "$distillation/production-1m-student-$variant-$mode-gift-short55-seed42.json" \
        >"$raw/production-1m-student-$variant-$mode-gift-short55-seed42.log" 2>&1
    done
  done
}
evaluate_seed42 0 gt dual_view &
eval_gpu0=$!
evaluate_seed42 1 kd cvrd &
eval_gpu1=$!
wait "$eval_gpu0"
wait "$eval_gpu1"

benchmark_students() {
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

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_teacher.py \
  configs/systems/teacher_reference_blackwell.yaml --physical-gpu-index 0 \
  --output "$systems/timesfm3-teacher-reference-blackwell-model-and-e2e.json" \
  >"$raw/timesfm3-teacher-reference-blackwell-model-and-e2e.log" 2>&1 &
teacher_benchmark=$!
benchmark_students 1 kd cvrd &
student_benchmark_gpu1=$!
wait "$teacher_benchmark"
benchmark_students 0 gt dual_view
wait "$student_benchmark_gpu1"

seed42_results=(
  results/reproduction/mv_short/timesfm3-multivariate-short-full-multivariate-seed42.json
  results/reproduction/mv_short/timesfm3-multivariate-short-full-univariate-seed42.json
  "$teacher55/timesfm3-gift-short-full-multivariate-seed42.json"
  "$teacher55/timesfm3-gift-short-full-univariate-seed42.json"
)
confirmatory_results=("${seed42_results[@]}")
for scope in gift-mv19 gift-short55; do
  for variant in gt kd dual_view cvrd; do
    for mode in multivariate univariate; do
      seed42_results+=("$distillation/production-1m-student-$variant-$mode-$scope-seed42.json")
    done
  done
  for variant in dual_view cvrd; do
    for seed in 42 43 44; do
      for mode in multivariate univariate; do
        confirmatory_results+=("$distillation/production-1m-student-$variant-$mode-$scope-seed$seed.json")
      done
    done
  done
done

seed42_args=()
for result in "${seed42_results[@]}"; do seed42_args+=(--result "$result"); done
.venv/bin/python scripts/summarize_gift_comparison.py "${seed42_args[@]}" \
  --output "$distillation/production-1m-gift-comparison-seed42.json" \
  >"$raw/production-1m-gift-comparison-seed42.log" 2>&1

confirmatory_args=()
for result in "${confirmatory_results[@]}"; do confirmatory_args+=(--result "$result"); done
.venv/bin/python scripts/summarize_gift_comparison.py "${confirmatory_args[@]}" \
  --output "$distillation/production-1m-gift-comparison-confirmatory.json" \
  >"$raw/production-1m-gift-comparison-confirmatory.log" 2>&1

response_args=()
for variant in dual_view cvrd; do
  response_args+=(--result "$distillation/production-1m-student-$variant-response-diagnostics.json")
  for seed in 43 44; do
    response_args+=(--result "$distillation/production-1m-student-$variant-response-diagnostics-seed$seed.json")
  done
done
.venv/bin/python scripts/summarize_response_uncertainty.py "${response_args[@]}" \
  --output "$distillation/production-1m-response-transfer-confirmatory.json" \
  >"$raw/production-1m-response-transfer-confirmatory.log" 2>&1

frontier_args=()
for variant in gt kd dual_view cvrd; do
  frontier_args+=(--student "$systems/production-1m-student-$variant-blackwell.json")
done
.venv/bin/python scripts/summarize_inference_frontier.py \
  --teacher "$systems/timesfm3-teacher-reference-blackwell-model-and-e2e.json" \
  "${frontier_args[@]}" --output "$systems/production-1m-inference-frontier.json" \
  >"$raw/production-1m-inference-frontier.log" 2>&1

for result in \
  "$distillation/production-1m-gift-comparison-seed42.json" \
  "$distillation/production-1m-gift-comparison-confirmatory.json" \
  "$distillation/production-1m-response-transfer-confirmatory.json" \
  "$systems/production-1m-inference-frontier.json"; do
  jq -e '.status == "succeeded"' "$result" >/dev/null
done
echo "production evaluation and confirmatory aggregation complete"
