#! /bin/bash
# Compatible with nsys CLI tool 2021.3.xx or later.
# In nvidia container image, remove pre-installed nsight-systems CLI tool.
# apt remove -y nsight-systems-cli-2020.3.4 && rm -f /usr/local/cuda/bin/nsys
# Usage: bash mgwg_gpt_sample.profile_multiple.sh [test_name]
#        test_name - default: "YYYYmmDDHHMMSS"
# Output will be saved to output_${test_name} directory.
# Set environment variables
#   SLEEP_TIME=xx (time between each runs. default: 5)
#   BATCH_SIZE=xx (default: 8)
#   DRY_RUN=true (default: false)
#   SKIP_PREPARE=true (default: false, for debugging)
#   SKIP_TIMERUN=true (default: false)
#   SKIP_PROFILE=true (default: false)
#   PROFILE_WITH_TIMERUN=true (default: false)
#   CAPTURE_GPU_METRICS=true (default: false)

sleep_time=${SLEEP_TIME:-5}
batch_size=${BATCH_SIZE:-8}

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run for listing configurations..."
else
  test_name=${1:-$(date +'%Y%m%d%H%M%S')}
  outdir="output_${test_name}"
  mkdir -p "${outdir}"
  echo "Output files will be saved to ${outdir}."

  summary_file="${outdir}/gpt_summary_${test_name}.csv"
  echo "suffix,model,seq1,seq2,batch_size,local_bs,tensor_p,layer_p,pipeline_p,gpus," \
      "gpt_time_costs_n,gpt_time_costs_mean," \
      "profiled_gpt_time_costs_n,profiled_gpt_time_costs_mean," \
      "prepare_ret,timerun_ret,profile_ret,nsys_abort," \
      "prepare_time,timerun_time,profile_time," \
      "prepare_out,prepare_err,timerun_out,timerun_err,profile_out,profile_err,nsys_report" \
      | tr -d " " > "${summary_file}"
fi

profile_target="./pytorch/scripts/mgwg_gpt_sample.run.sh"
if [[ "${PROFILE_WITH_TIMERUN}" == "true" ]]; then
  profile_target="./pytorch/scripts/mgwg_gpt_sample.run_time.sh"
fi

function start_timer() {
  SECONDS=0
}

function get_time_elapsed() {
  local duration=${SECONDS}
  printf "%02d:%02d:%02d" $((${duration}/3600)) $(((${duration}/60)%60)) $((${duration}%60))
}

function get_gpu_metrics_device() {
  local devices=none
  if [[ "${CAPTURE_GPU_METRICS}" == "true" ]]; then
    local gpus=$1
    if (( ${gpus} > 0 )); then
      devices=0
      local id=1
      while (( ${id} < ${gpus} )); do
        devices="${devices},${id}"
        (( id++ ))
      done
    fi
  fi
  echo "${devices}"
}

function get_gpt_time_costs() {
  local file=$1
  if [[ ! -f ${file} ]]; then
    echo "no_file N/A"
    return
  fi
  local nums=$(egrep "^\[INFO\] GPT time costs: ([0-9]+(\.[0-9]*)?) ms$" ${file} \
      | sed -E "s/^\[INFO\] GPT time costs: ([0-9]+(\.[0-9]*)?) ms$/\1/g")
  local cnt=$(echo ${nums} | wc -w | tr -d " ")
  if (( ${cnt} == 0 )); then
    echo "0 N/A"
    return
  fi
  local expression="(0.0"
  for n in ${nums}; do
    expression="${expression}+${n}"
  done
  expression="${expression})/${cnt}"
  mean=$(python -c "print(${expression})")
  echo "${cnt} ${mean}"
}

function chk_file() {
  local file=$1
  if [[ -f "${outdir}/${file}" ]]; then
    echo "${file}"
  else
    echo "no_file(${file})"
  fi
}

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

  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi

  local prepare_ret="skip"
  local prepare_time="skip"
  if [[ "${SKIP_PREPARE}" != "true" ]]; then
    echo "[$(date +'%Y-%m-%dT%H:%M:%S')] Preparing... ${suffix}"
    start_timer
    bash ./pytorch/scripts/mgwg_gpt_sample.prepare.sh \
        "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
        > "${outdir}/prepare${suffix}.txt" 2> "${outdir}/prepare${suffix}_err.txt"
    prepare_ret=$?
    prepare_time=$(get_time_elapsed)
    echo "Returned code: ${prepare_ret} / Elapsed time: ${prepare_time}"
    sleep ${sleep_time}
  fi

  local timerun_ret="skip"
  local timerun_time="skip"
  local gpt_time_costs="skip skip"
  if [[ "${SKIP_TIMERUN}" != "true" ]]; then
    echo "[$(date +'%Y-%m-%dT%H:%M:%S')] Measuring time... ${suffix}"
    start_timer
    bash ./pytorch/scripts/mgwg_gpt_sample.run_time.sh \
        "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
        > "${outdir}/timerun${suffix}.txt" 2> "${outdir}/timerun${suffix}_err.txt"
    timerun_ret=$?
    timerun_time=$(get_time_elapsed)
    echo "Returned code: ${timerun_ret} / Elapsed time: ${timerun_time}"
    sleep ${sleep_time}
    gpt_time_costs=$(get_gpt_time_costs "${outdir}/timerun${suffix}.txt")
  fi

  local profile_ret="skip"
  local profile_time="skip"
  local p_gpt_time_costs="skip skip"
  local nsys_abort="skip"
  if [[ "${SKIP_PROFILE}" != "true" ]]; then
    local gpu_metrics_device="$(get_gpu_metrics_device ${gpus})"
    local count=10
    while (( ${count} > 0 )); do
      (( count-- ))
      echo "[$(date +'%Y-%m-%dT%H:%M:%S')] Profiling... ${suffix}"
      start_timer
      nsys profile -o "${outdir}/profile${suffix}" --force-overwrite true \
          --gpu-metrics-device="${gpu_metrics_device}" \
          bash "${profile_target}" \
          "${model}" "${s1}" "${s2}" "${bs}" "${lbs}" "${tp}" "${lp}" "${gpus}" \
          > "${outdir}/profile${suffix}.txt" 2> "${outdir}/profile${suffix}_err.txt"
      profile_ret=$?
      profile_time=$(get_time_elapsed)
      echo "Returned code: ${profile_ret} / Elapsed time: ${profile_time}"
      if [[ "${profile_ret}" == "134" ]]; then
        nsys_abort="maybe"
      else
        nsys_abort="no"
        break
      fi
    done
    sleep ${sleep_time}
    if [[ "${PROFILE_WITH_TIMERUN}" == "true" ]]; then
      p_gpt_time_costs=$(get_gpt_time_costs "${outdir}/profile${suffix}.txt")
    fi
  fi

  echo "${suffix},${model},${s1},${s2},${bs},${lbs},${tp},${lp},${pp},${gpus}," \
      "$(echo ${gpt_time_costs} | awk '{print $1}'),$(echo ${gpt_time_costs} | awk '{print $2}')," \
      "$(echo ${p_gpt_time_costs} | awk '{print $1}'),$(echo ${p_gpt_time_costs} | awk '{print $2}')," \
      "${prepare_ret},${timerun_ret},${profile_ret},${nsys_abort}," \
      "${prepare_time},${timerun_time},${profile_time}," \
      "$(chk_file prepare${suffix}.txt),$(chk_file prepare${suffix}_err.txt)," \
      "$(chk_file timerun${suffix}.txt),$(chk_file timerun${suffix}_err.txt)," \
      "$(chk_file profile${suffix}.txt),$(chk_file profile${suffix}_err.txt)," \
      "$(chk_file profile${suffix}.nsys-rep)" \
      | tr -d " " >> "${summary_file}"
}

function batches_and_seq_lens() {
  local model=$1
  local tp=$2
  local lp=$3
  local pp=$4
  # fix bs to 8 for now
  for bs in ${batch_size}; do
    run_experiment "${model}" 1 32 ${bs} "${tp}" "${lp}" "${pp}"
    run_experiment "${model}" 512 1 ${bs} "${tp}" "${lp}" "${pp}"
  done
}

function np_tests() {
  batches_and_seq_lens $1 1 1 1
}

function tp_tests() {
  local model=$1
  # tp list = $2, $3, $4, ...
  while (( $# > 1 )); do
    shift
    batches_and_seq_lens "${model}" "$1" 1 1
  done
}

function lp_pp_tests() {
  local model=$1
  # lp list = $2, $3, $4, ...
  while (( $# > 1 )); do
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

echo "<<< gpt 1558M model - without parallelism >>>"
np_tests 1558M
echo "<<< gpt 1558M model - tensor parallelism >>>"
tp_tests 1558M 5 # heads = 25
echo "<<< gpt 1558M model - layer(/pipeline) parallelism >>>"
lp_pp_tests 1558M 2 4 8 # layers = 48

echo "<<< gpt 2525M model - without parallelism >>>"
np_tests 2525M
echo "<<< gpt 2525M model - tensor parallelism >>>"
tp_tests 2525M 2 4 8 # heads = 32
echo "<<< gpt 2525M model - layer(/pipeline) parallelism >>>"
lp_pp_tests 2525M 2 4 8 # layers = 48
echo "<<< gpt 2525M model - tensor+layer parallelism >>>"
tp_lp_tests 2525M

echo "<<< gpt 89B model - tensor parallelism >>>"
tp_tests 89B 8 # heads = 96
#echo "<<< gpt 89B model - layer parallelism >>>"
#lp_pp_tests 89B 8 # layers = 48 # fails
#echo "<<< gpt 89B model - tensor+layer parallelism >>>"
#tp_lp_tests 89B # fails
