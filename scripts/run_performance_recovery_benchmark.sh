#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 SYSTEMS_CONFIG DATA_ROOT STUDENT_CONFIG CHECKPOINT LABEL OUTPUT_DIR GPU [STUDENT_FACTORY] [COMPILE_MODE]" >&2
  exit 2
}

[[ $# -ge 7 && $# -le 9 ]] || usage

systems_config="$1"
data_root="$2"
student_config="$3"
checkpoint="$4"
label="$5"
output_dir="$6"
physical_gpu="$7"
student_factory="${8:-}"
compile_mode="${9:-}"

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PYTHONPATH=src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1

if ! [[ "$label" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "LABEL may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if ! [[ "$physical_gpu" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a physical integer index" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "refusing a claim-grade benchmark with uncommitted tracked changes" >&2
  exit 1
fi
for path in "$systems_config" "$student_config" "$checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "required file does not exist: $path" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
manifest="$output_dir/workload-manifest.json"
.venv/bin/python scripts/benchmark_performance_recovery.py \
  "$systems_config" --data-root "$data_root" --model-kind validate --output "$manifest"

current_commit="$(git rev-parse HEAD)"
config_sha="$(sha256sum "$systems_config" | awk '{print $1}')"
student_config_sha="$(sha256sum "$student_config" | awk '{print $1}')"
checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
suite_sha="$(jq -r '.suite_input_sha256' "$manifest")"
authority_sha="$(jq -r '.target_authority_sha256' "$manifest")"
gpu_uuid="$(nvidia-smi --id="$physical_gpu" --query-gpu=uuid --format=csv,noheader,nounits | xargs)"

assert_gpu_free() {
  local active
  active="$(nvidia-smi --id="$physical_gpu" --query-compute-apps=pid --format=csv,noheader,nounits | xargs)"
  if [[ -n "$active" ]]; then
    echo "physical GPU $physical_gpu is not free; active compute PID(s): $active" >&2
    exit 1
  fi
}

valid_existing() {
  local output="$1"
  local kind="$2"
  local trial="$3"
  [[ -f "$output" ]] || return 1
  jq -e \
    --arg commit "$current_commit" \
    --arg config_sha "$config_sha" \
    --arg suite_sha "$suite_sha" \
    --arg authority_sha "$authority_sha" \
    --arg gpu_uuid "$gpu_uuid" \
    --arg kind "$kind" \
    --arg label "$label" \
    --arg checkpoint_sha "$checkpoint_sha" \
    --arg student_config_sha "$student_config_sha" \
    --argjson trial "$trial" \
    '.status == "succeeded"
      and .git_commit == $commit
      and .extra.config_sha256 == $config_sha
      and .extra.suite_input_sha256 == $suite_sha
      and .extra.target_authority_sha256 == $authority_sha
      and .extra.gpu.uuid == $gpu_uuid
      and .extra.model_kind == $kind
      and .extra.label == $label
      and .extra.trial == $trial
      and (if ($kind | startswith("teacher_")) then .extra.checkpoint_sha256 == null
           else .extra.checkpoint_sha256 == $checkpoint_sha
             and .extra.student_config_sha256 == $student_config_sha end)' \
    "$output" >/dev/null
}

run_trial() {
  local kind="$1"
  local trial="$2"
  local output="$output_dir/$kind-trial$trial.json"
  if valid_existing "$output" "$kind" "$trial"; then
    echo "verified existing $kind trial $trial: $output"
    return
  fi
  assert_gpu_free
  local command=(
    .venv/bin/python scripts/benchmark_performance_recovery.py
    "$systems_config" --data-root "$data_root" --model-kind "$kind"
    --label "$label" --trial "$trial" --physical-gpu-index "$physical_gpu"
    --output "$output"
  )
  if [[ "$kind" == student ]]; then
    command+=(--student-config "$student_config" --checkpoint "$checkpoint")
    [[ -z "$student_factory" ]] || command+=(--student-factory "$student_factory")
    [[ -z "$compile_mode" ]] || command+=(--compile --compile-mode "$compile_mode")
  fi
  echo "running isolated $kind trial $trial on physical GPU $physical_gpu"
  CUDA_VISIBLE_DEVICES="$physical_gpu" "${command[@]}"
}

# Symmetric serial order balances thermal/order effects and records both the
# unmodified official teacher path and the evaluator-equivalent optimized path.
run_trial teacher_stock 1
run_trial teacher_optimized 1
run_trial student 1
run_trial student 2
run_trial teacher_optimized 2
run_trial teacher_stock 2

.venv/bin/python scripts/summarize_performance_recovery_benchmark.py \
  "$systems_config" \
  --teacher "$output_dir/teacher_optimized-trial1.json" \
  --teacher "$output_dir/teacher_optimized-trial2.json" \
  --student "$output_dir/student-trial1.json" \
  --student "$output_dir/student-trial2.json" \
  --output "$output_dir/comparison-optimized-teacher.json"

.venv/bin/python scripts/summarize_performance_recovery_benchmark.py \
  "$systems_config" \
  --teacher "$output_dir/teacher_stock-trial1.json" \
  --teacher "$output_dir/teacher_stock-trial2.json" \
  --student "$output_dir/student-trial1.json" \
  --student "$output_dir/student-trial2.json" \
  --output "$output_dir/comparison-stock-teacher.json"

jq '.speed_target, .parameter_comparison' "$output_dir/comparison-optimized-teacher.json"
jq '.speed_target, .parameter_comparison' "$output_dir/comparison-stock-teacher.json"
