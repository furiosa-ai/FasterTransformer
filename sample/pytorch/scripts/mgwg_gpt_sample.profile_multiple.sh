#! /bin/bash

dry_run=${1:-false}

function run_experiment() {
  local model=$1
  local s1=$2
  local s2=$3
  local bs=$4
  local tp=$5
  local lp=$6
  local pp=$7
  if (( ${bs}%${pp} > 0 )); then
    return
  fi
  local lbs=$((${bs}/${pp}))
  local gpus=$((${tp}*${lp}))
  if (( ${gpus} > 8 )); then
    return
  fi
  local suffix="_${model}_${s1}_${s2}_${bs}_${lbs}_${tp}_${lp}_${gpus}"
  echo "<< model:${model} / s1:${s1} / s2:${s2} / bs:${bs} / lbs:${lbs} / tp:${tp} / lp:${lp} / pp:${pp} / gpus:${gpus} >>"
  if [[ "${dry_run}" == "true" ]]; then
    return
  fi
  case ${gpus} in
    1)
      devices=0
      ;;
    2)
      devices=0,1
      ;;
    4)
      devices=0,1,2,3
      ;;
    5)
      devices=0,1,2,3,4
      ;;
    *)
      devices=all
      ;;
  esac

  echo "[$(date +'%H:%M:%S')] Preparing... ${suffix}"
  ./pytorch/scripts/mgwg_gpt_sample.prepare.sh \
      "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
      > prepare${suffix}.txt 2> prepare${suffix}_err.txt
  sleep 1
  echo "[$(date +'%H:%M:%S')] Profiling... ${suffix}"
  nsys profile -o "gpt${suffix}" --force-overwrite true \
      --gpu-metrics-device="${devices}" \
      ./pytorch/scripts/mgwg_gpt_sample.run.sh \
      "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
      > run${suffix}.txt 2> run${suffix}_err.txt
  sleep 1
  echo "[$(date +'%H:%M:%S')] Measuring time... ${suffix}"
  ./pytorch/scripts/mgwg_gpt_sample.run_time.sh \
      "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
      > time${suffix}.txt 2> time${suffix}_err.txt
  sleep 1
}

function batches_and_seq_lens() {
  local model=$1
  local tp=$2
  local lp=$3
  local pp=$4
  # fix bs to 8 for now
  for bs in 8; do
    run_experiment "${model}" 1 32 8 "${tp}" "${lp}" "${pp}"
    run_experiment "${model}" 512 1 8 "${tp}" "${lp}" "${pp}"
  done
}

function tp_tests() {
  local model=$1
  # tp list = $2, $3, $4, ...
  while [[ $# -gt 1 ]]; do
    shift
    batches_and_seq_lens "${model}" "$1" 1 1
  done
}

function lp_pp_tests() {
  local model=$1
  # lp list = $2, $3, $4, ...
  while [[ $# -gt 1 ]]; do
    shift
    local lp=$1
    for pp in 1 2 4 8; do
      batches_and_seq_lens "${model}" 1 "${lp}" "${pp}"
    done
  done
}

function tp_lp_tests() {
  local model=$1
  # 1x8, 8x1 cases are included in tp_tests or lp_pp_tests
  for tp in 2 4; do
    local lp=$((8/${tp}))
    # Don't know optimal pp value.. just run all
    for pp in 1 2 4 8; do
      batches_and_seq_lens "${model}" "${tp}" "${lp}" "${pp}"
    done
  done
}

echo "<< Tests for tensor parallelism>>"
tp_tests 124M 1 2 4 # head = 12
tp_tests 1558M 1 5 # head = 25
tp_tests 89B 8 # head = 96

echo "<< Tests for layer parallelism>>"
lp_pp_tests 124M 2 4 8
lp_pp_tests 1558M 2 4 8
lp_pp_tests 89B 8

echo "<< Tests for mixed parallelism>>"
tp_lp_tests 89B
