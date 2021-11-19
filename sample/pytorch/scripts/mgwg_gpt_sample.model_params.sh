# ex.) source ./pytorch/scripts/mgwg_gpt_sample.model_params.sh

if [ $model_name == "124M" ]; then
    layer_num=12
    head_num=12
    size_per_head=64
    vocab_size=50257
    ckpt_path="../models/openai-gpt-models/c-model/124m/${tensor_para_size}-gpu"
elif [ $model_name == "1558M" ]; then
    layer_num=48
    head_num=25
    size_per_head=64
    vocab_size=50257
    ckpt_path="../models/openai-gpt-models/c-model/1558m/${tensor_para_size}-gpu"
elif [ $model_name == "2525M" ]; then
    # Variation of 1558M model with head_num 32.
    layer_num=48
    head_num=32
    size_per_head=64
    vocab_size=50257
    ckpt_path=-1
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
