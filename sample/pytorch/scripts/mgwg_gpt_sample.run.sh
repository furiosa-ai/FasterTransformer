#! /bin/bash
# ex) ./pytorch/scripts/mgwg_gpt_sample.run.sh 124M 256 32 8 8 1 1 1

# $1 = <model_name>
# $2 = max input sequence length ( S1 )
# $3 = max output sequence length ( S2 )
# $4 = batch_size
# $5 = <local_batch_size>
# $6 = tensor_para_size
# $7 = layer_para_size
# $8 = n_gpu

if [ $# -lt 7 ]; then
    echo "More arguments are needed."
    exit 1
fi

model_name=$1
s1=$2
s2=$3
batch_size=$4
local_batch_size=$5
context_local_batch_size=$local_batch_size
tensor_para_size=$6
layer_para_size=$7
n_gpu=$8

max_seq_len=$(($s1+$s2))
_n_gpu=$(($tensor_para_size*$layer_para_size))

if [ $n_gpu != $_n_gpu  ]; then
    echo "[Error]n_gpu should be equal to tensor_para_size * layer_para_size"
    exit 1
fi

if [ $model_name == "124M" ]; then
    layer_num=12
    head_num=12
    size_per_head=64
    vocab_size=50257
    vara="../models/openai-gpt-models/c-model/124m/"
    _vara="-gpu"
    ckpt_path="$vara$tensor_para_size$_vara"
elif [ $model_name == "1558M" ]; then
    layer_num=48
    head_num=25
    size_per_head=64
    vocab_size=50257
    vara="../models/openai-gpt-models/c-model/1558m/"
    _vara="-gpu"
    ckpt_path="$vara$tensor_para_size$_vara"
elif [ $model_name == "89B" ]; then
    layer_num=48
    head_num=96
    size_per_head=128
    vocab_size=51200
    ckpt_path=-1
elif [ $model_name == "175B" ]; then
    layer_num=96
    head_num=96
    size_per_head=128
    vocab_size=51200
    ckpt_path=-1
elif [ $model_name == "test" ]; then
    layer_num=48
    head_num=25
    size_per_head=64
    vocab_size=50257
    ckpt_path=-1
else
    echo "wrong model name"
    exit 1
fi

echo $ckpt_path

is_fp16=1

# echo $ckpt_path
# echo $max_seq_len

# ./bin/gpt_gemm $local_batch_size $context_local_batch_size $head_num $size_per_head $vocab_size $s1 $tensor_para_size $is_fp16
# python ./pytorch/gen_sample_input_file.py --batch_size $batch_size --input_seq_len $s1 --o_file_name ./sample_input.txt

mpirun -n $n_gpu --allow-run-as-root python ./pytorch/gpt_sample.py \
    --ckpt_path $ckpt_path --fp16 --top_k 0 --top_p 0 --sample_input_file ./sample_input.txt \
    --layer_num $layer_num --head_num $head_num --size_per_head $size_per_head --vocab_size $vocab_size \
    --output_len $s2 --max_seq_len $max_seq_len \
    --tensor_para_size $tensor_para_size --layer_para_size $layer_para_size \
    --max_batch_size $batch_size --layer_para_batch_size $local_batch_size

