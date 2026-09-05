#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted seed-44 pair from exact optimizer checkpoints, evaluate
# both confirmatory seeds, then complete the remaining production evaluation.

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

for variant in dual_view cvrd; do
  jq -e '.status == "succeeded" and .seed == 43 and .extra.training.steps == 200000 and .extra.validation_split_seed == 42' \
    "$distillation/production-1m-student-$variant-seed43.json" >/dev/null
done

resume_method() {
  local gpu="$1"
  local variant="$2"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/train_production_student.py \
    "$config" "$plan" --variant "$variant" --training-seed 44 --split-seed 42 \
    --data-root "$pretrain_data" --cache-root "$teacher_cache" \
    --checkpoint-dir "checkpoints/production-1m/seed44/$variant" \
    --resume "checkpoints/production-1m/seed44/$variant/student-$variant-resume.pt" \
    --disable-early-stopping \
    --output "$distillation/production-1m-student-$variant-seed44.json" \
    >>"$raw/production-1m-student-$variant-seed44.log" 2>&1
}

resume_method 0 dual_view &
dual_pid=$!
resume_method 1 cvrd &
cvrd_pid=$!
wait "$dual_pid"
wait "$cvrd_pid"
for variant in dual_view cvrd; do
  jq -e '.status == "succeeded" and .seed == 44 and .extra.training.steps == 200000 and .extra.validation_split_seed == 42' \
    "$distillation/production-1m-student-$variant-seed44.json" >/dev/null
done
echo "seed-44 matched training complete; starting confirmatory evaluations"

evaluate_method() {
  local gpu="$1"
  local variant="$2"
  local seed checkpoint scope_config scope_label mode
  for seed in 43 44; do
    checkpoint="checkpoints/production-1m/seed$seed/$variant/student-$variant-best.pt"
    for scope_config in configs/reproduction/multivariate_short_full.yaml configs/reproduction/gift_short_full.yaml; do
      if [[ "$scope_config" == *multivariate_short_full.yaml ]]; then
        scope_label=gift-mv19
      else
        scope_label=gift-short55
      fi
      for mode in multivariate univariate; do
        CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/evaluate_student.py \
          "$config" "$scope_config" --checkpoint "$checkpoint" --variant "$variant" \
          --seed "$seed" --mode "$mode" --data-root "$gift_data" \
          --output "$distillation/production-1m-student-$variant-$mode-$scope_label-seed$seed.json" \
          >"$raw/production-1m-student-$variant-$mode-$scope_label-seed$seed.log" 2>&1
      done
    done
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/diagnose_production_responses.py \
      "$config" "$plan" --data-root "$pretrain_data" --cache-root "$teacher_cache" \
      --student-checkpoint "$checkpoint" --student-label "$variant-seed$seed" \
      --student-batch-size 32 \
      --output "$distillation/production-1m-student-$variant-response-diagnostics-seed$seed.json" \
      >"$raw/production-1m-student-$variant-response-diagnostics-seed$seed.log" 2>&1
  done
}

evaluate_method 0 dual_view &
dual_eval_pid=$!
evaluate_method 1 cvrd &
cvrd_eval_pid=$!
wait "$dual_eval_pid"
wait "$cvrd_eval_pid"
echo "confirmatory evaluations complete; resuming missing production stages"

scripts/resume_production_evaluation.sh 0
