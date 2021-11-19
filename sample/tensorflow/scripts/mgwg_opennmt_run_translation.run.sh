# ex) ./tensorflow/scripts/mgwg_opennmt_run_translation.run.sh beamsearch 8 4 32

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
    test_time=5
else
    echo "wrong decoding mode name"
    exit 1
fi

memory_hidden_dim=$(($head_num*$size_per_head))
data_type='fp16'
is_fp16=1
s1=1    # s1 is not used in OpenNMT


# ./bin/decoding_gemm $batch_size $beam_size $head_num $size_per_head $vocab_size $s1 $memory_hidden_dim $is_fp16

python tensorflow/translate_sample.py \
      --batch_size $batch_size \
      --beam_width $beam_size \
      --encoder_head_number $head_num \
      --encoder_size_per_head $size_per_head \
      --decoder_head_number $head_num \
      --decoder_size_per_head $size_per_head \
      --max_seq_len $s2 \
      --encoder_num_layer $layer_num \
      --decoder_num_layer $layer_num \
      --data_type $data_type \
      --beam_search_diversity_rate -1.3 \
      --sampling_topk 4 \
      --sampling_topp 0.00 \
      --test_time $test_time


