#! /bin/bash

outdir=${1:-output-$(date +'%Y%m%d%H%M%S')}
if [[ "${outdir}" == "none" ]]; then
  dry_run=true
  echo "Dry run for listing configurations..."
else
  mkdir -p "${outdir}"
  echo "Output files will be saved to ${outdir}."

  summary_file="${outdir}/gpt_summary.csv"
  echo "suffix,model,seq1,seq2,batch_size,local_bs,tensor_p,layer_p,pipeline_p,gpus," \
      "prepare_ret,timerun_ret,profile_ret,nsys_abort,prepare_stime,timerun_stime,profile_stime," \
      "prepare_out,prepare_err,timerun_out,timerun_err,profile_out,profile_err,nsys_report" \
      | tr -d " " > "${summary_file}"
fi

sleep_time=5

function run_experiment() {
  local model=$1
  local s1=$2
  local s2=$3
  local bs=$4
  local tp=$5
  local lp=$6
  local pp=$7
  if (( ${bs}%${pp} > 0 )); then
    echo "bs(${bs})%pp(${pp}) is not 0."
    return
  fi
  local lbs=$((${bs}/${pp}))
  local gpus=$((${tp}*${lp}))
  if (( ${gpus} > 8 )); then
    echo "number of gpus (${gpus}) is larger than 8."
    return
  fi
  local suffix="_${model}_${s1}_${s2}_${bs}_${lbs}_${tp}_${lp}_${gpus}"
  #echo "<< model:${model} / s1:${s1} / s2:${s2} / bs:${bs} / lbs:${lbs} / tp:${tp} / lp:${lp} / pp:${pp} / gpus:${gpus} >>"
  printf "<< model:%s / s1:%-3d / s2:%-3d / bs:%-2d / lbs:%-2d / tp:%-d / lp:%-d / pp:%-d / gpus:%-d >>\n" \
      ${model} ${s1} ${s2} ${bs} ${lbs} ${tp} ${lp} ${pp} ${gpus}

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

  local prepare_stime=$(date +'%Y-%m-%dT%H:%M:%S')
  echo "[${prepare_stime}] Preparing... ${suffix}"
  bash ./pytorch/scripts/mgwg_gpt_sample.prepare.sh \
      "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
      > "${outdir}/prepare${suffix}.txt" 2> "${outdir}/prepare${suffix}_err.txt"
  prepare_ret=$? # No local

  sleep ${sleep_time}

  local timerun_stime=$(date +'%Y-%m-%dT%H:%M:%S')
  echo "[${timerun_stime}] Measuring time... ${suffix}"
  bash ./pytorch/scripts/mgwg_gpt_sample.run_time.sh \
      "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
      > "${outdir}/timerun${suffix}.txt" 2> "${outdir}/timerun${suffix}_err.txt"
  timerun_ret=$? # No local

  sleep ${sleep_time}

  local profile_stime=$(date +'%Y-%m-%dT%H:%M:%S')
  echo "[${profile_stime}] Profiling... ${suffix}"
  nsys profile -o "${outdir}/profile${suffix}" --force-overwrite true \
      --gpu-metrics-device="${devices}" \
      bash ./pytorch/scripts/mgwg_gpt_sample.run.sh \
      "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
      > "${outdir}/profile${suffix}.txt" 2> "${outdir}/profile${suffix}_err.txt"
  profile_ret=$? # No local
  local nsys_abort=false
  if [[ "${profile_ret}" == "134" ]]; then
    nsys_abort=true
  fi

  echo "${suffix},${model},${s1},${s2},${bs},${lbs},${tp},${lp},${pp},${gpus}," \
      "${prepare_ret},${timerun_ret},${profile_ret},${nsys_abort}," \
      "${prepare_stime},${timerun_stime},${profile_stime}," \
      "prepare${suffix}.txt,prepare${suffix}_err.txt," \
      "timerun${suffix}.txt,timerun${suffix}_err.txt," \
      "profile${suffix}.txt,profile${suffix}_err.txt,profile${suffix}.nsys-rep" \
      | tr -d " " >> "${summary_file}"

  sleep ${sleep_time}
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

function np_tests() {
  batches_and_seq_lens $1 1 1 1
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

echo "<<< gpt 124M model - without parallelism >>>"
np_tests 124M
echo "<<< gpt 124M model - tensor parallelism >>>"
tp_tests 124M 2 4 # heads = 12
echo "<<< gpt 124M model - layer(/pipeline) parallelism >>>"
lp_pp_tests 124M 2 4 # layers = 12

echo "<<< gpt 1588M model - without parallelism >>>"
np_tests 1588M
echo "<<< gpt 1588M model - tensor parallelism >>>"
tp_tests 1558M 5 # heads = 25
echo "<<< gpt 1588M model - layer(/pipeline) parallelism >>>"
lp_pp_tests 1558M 2 4 8 # layers = 48

echo "<<< gpt 89B model - tensor parallelism >>>"
tp_tests 89B 8 # heads = 96
echo "<<< gpt 89B model - layer parallelism >>>"
lp_pp_tests 89B 8 # layers = 48
echo "<<< gpt 89B model - tensor+layer parallelism >>>"
tp_lp_tests 89B
