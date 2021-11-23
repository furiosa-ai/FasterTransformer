# ex) ./pytorch/scripts/mgwg_opennmt_run_translation.prepare.sh beamsearch 8 4 32

# $1 = decoding_mode
# $2 = batch_size
# $3 = beam_size
# $4 = output_seq_len

decoding_mode=$1
batch_size=$2
beam_size=$3
s2=$4

model_name='base'

if [ $model_name == "base" ]; then
    layer_num=6
    head_num=8
    size_per_head=64
    vocab_size=31538
    model_path="./pytorch/translation/models/averaged-10-epoch.pt"
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

if [ $decoding_mode == "beamsearch" ]; then
    test_time=2
elif [ $decoding_mode == "sampling" ]; then
    # test_time=5
    echo "sampling decoding is not supported in PyTorch."
    exit 1
else
    echo "wrong decoding mode name"
    exit 1
fi

memory_hidden_dim=$(($head_num*$size_per_head))
data_type='fp16'
model_type='decoding_ext'
is_fp16=1
s1=1    # s1 is not used in OpenNMT


./bin/decoding_gemm $batch_size $beam_size $head_num $size_per_head $vocab_size $s1 $memory_hidden_dim $is_fp16

# python pytorch/run_translation.py --batch_size $batch_size --beam_size $beam_size --model_type $model_type --data_type $data_type --max_seq_len $s2

