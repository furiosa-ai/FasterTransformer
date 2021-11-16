#! /bin/bash

for model in 124M 1558M 89B
do
  if [[ "${model}" == "124M" ]]; then
    n_head=12
  elif [[ "${model}" == "1558M" ]]; then
    n_head=25
  else
    n_head=96
  fi
  for tp in 1 2 4 8
  do
    if [[ $((${n_head} % ${tp})) -ne 0 ]]; then
      echo ">> Skipping ${n_head} / ${tp}"
      continue
    fi
    if [[ ${tp} -eq 1 ]]; then
      devices="0"
    elif [[ ${tp} -eq 2 ]]; then
      devices="0,1"
    elif [[ ${tp} -eq 4 ]]; then
      devices="0,1,2,3"
    else
      devices="all"
    fi
    for s1 in 32 128 384
    do
      for s2 in 32 128
      do
        for b in 1 8 32
        do
          suffix="_${model}_${s1}_${s2}_${b}_${b}_${tp}tp_1lp"
          echo ">> Preparing... ${suffix}"
          ./pytorch/scripts/mgwg_gpt_sample.prepare.sh \
              "${model}" "${s1}" "${s2}" "${b}" "${b}" "${tp}" 1 "${tp}" \
              > prepare${suffix}.txt 2> prepare${suffix}_err.txt
          echo ">> Profiling... ${suffix}"
          nsys profile -o "gpt${suffix}" --force-overwrite true \
              --gpu-metrics-device="${devices}" \
              ./pytorch/scripts/mgwg_gpt_sample.run.sh \
              "${model}" "${s1}" "${s2}" "${b}" "${b}" "${tp}" 1 "${tp}" \
              > run${suffix}.txt 2> run${suffix}_err.txt
          echo ">> Measuring time... ${suffix}"
          ./pytorch/scripts/mgwg_gpt_sample.run_time.sh \
              "${model}" "${s1}" "${s2}" "${b}" "${b}" "${tp}" 1 "${tp}" \
              > time${suffix}.txt 2> time${suffix}_err.txt
        done
      done
    done
  done
done
