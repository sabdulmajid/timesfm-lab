#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate two preselected methods at seeds 43 and 44 while keeping
# the leakage-resistant validation split fixed at the seed-42 production split.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 METHOD_GPU0 METHOD_GPU1" >&2
  exit 2
fi

method_gpu0="$1"
method_gpu1="$2"
case "$method_gpu0" in gt|kd|dual_view|cvrd) ;; *) exit 2 ;; esac
case "$method_gpu1" in gt|kd|dual_view|cvrd) ;; *) exit 2 ;; esac
if [[ "$method_gpu0" == "$method_gpu1" ]]; then
  echo "confirmatory methods must differ" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PYTHONPATH=src

config=configs/distillation/production_1m.yaml
plan=results/reproduction/distillation/production-1m-corpus-plan.json
pretrain_data=data/gift-pretrain-production
teacher_cache=teacher_cache/production-1m
gift_data=data/gift-eval-full
distillation=results/reproduction/distillation
raw=results/raw
mkdir -p "$distillation" "$raw"

train_method() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local checkpoint_dir="checkpoints/production-1m/seed$seed/$variant"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/train_production_student.py \
    "$config" "$plan" \
    --variant "$variant" \
    --training-seed "$seed" \
    --split-seed 42 \
    --data-root "$pretrain_data" \
    --cache-root "$teacher_cache" \
    --checkpoint-dir "$checkpoint_dir" \
    --disable-early-stopping \
    --output "$distillation/production-1m-student-$variant-seed$seed.json" \
    >"$raw/production-1m-student-$variant-seed$seed.log" 2>&1
}

for seed in 43 44; do
  train_method 0 "$method_gpu0" "$seed" &
  train_gpu0=$!
  train_method 1 "$method_gpu1" "$seed" &
  train_gpu1=$!
  wait "$train_gpu0"
  wait "$train_gpu1"
  for variant in "$method_gpu0" "$method_gpu1"; do
    result="$distillation/production-1m-student-$variant-seed$seed.json"
    jq -e \
      ".status == \"succeeded\" and .seed == $seed and .extra.training.steps == 200000 and .extra.validation_split_seed == 42" \
      "$result" >/dev/null
  done
done

evaluate_method() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local checkpoint="checkpoints/production-1m/seed$seed/$variant/student-$variant-best.pt"
  local scope_config scope_label mode
  for scope_config in \
    configs/reproduction/multivariate_short_full.yaml \
    configs/reproduction/gift_short_full.yaml; do
    if [[ "$scope_config" == *multivariate_short_full.yaml ]]; then
      scope_label=gift-mv19
    else
      scope_label=gift-short55
    fi
    for mode in multivariate univariate; do
      CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/evaluate_student.py \
        "$config" "$scope_config" \
        --checkpoint "$checkpoint" \
        --variant "$variant" \
        --seed "$seed" \
        --mode "$mode" \
        --data-root "$gift_data" \
        --output "$distillation/production-1m-student-$variant-$mode-$scope_label-seed$seed.json" \
        >"$raw/production-1m-student-$variant-$mode-$scope_label-seed$seed.log" 2>&1
    done
  done
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/diagnose_production_responses.py \
    "$config" "$plan" \
    --data-root "$pretrain_data" \
    --cache-root "$teacher_cache" \
    --student-checkpoint "$checkpoint" \
    --student-label "$variant-seed$seed" \
    --student-batch-size 32 \
    --output "$distillation/production-1m-student-$variant-response-diagnostics-seed$seed.json" \
    >"$raw/production-1m-student-$variant-response-diagnostics-seed$seed.log" 2>&1
}

for seed in 43 44; do
  evaluate_method 0 "$method_gpu0" "$seed" &
  eval_gpu0=$!
  evaluate_method 1 "$method_gpu1" "$seed" &
  eval_gpu1=$!
  wait "$eval_gpu0"
  wait "$eval_gpu1"
done

comparison=(
  results/reproduction/mv_short/timesfm3-multivariate-short-full-multivariate-seed42.json
  results/reproduction/mv_short/timesfm3-multivariate-short-full-univariate-seed42.json
  results/reproduction/gift_short_full/timesfm3-gift-short-full-multivariate-seed42.json
  results/reproduction/gift_short_full/timesfm3-gift-short-full-univariate-seed42.json
)
for scope_label in gift-mv19 gift-short55; do
  for variant in "$method_gpu0" "$method_gpu1"; do
    for seed in 42 43 44; do
      for mode in multivariate univariate; do
        comparison+=(
          "$distillation/production-1m-student-$variant-$mode-$scope_label-seed$seed.json"
        )
      done
    done
  done
done
comparison_args=()
for result in "${comparison[@]}"; do
  comparison_args+=(--result "$result")
done
.venv/bin/python scripts/summarize_gift_comparison.py \
  "${comparison_args[@]}" \
  --output "$distillation/production-1m-gift-comparison-confirmatory.json" \
  >"$raw/production-1m-gift-comparison-confirmatory.log" 2>&1

echo "confirmatory seeds complete for $method_gpu0 and $method_gpu1"
