#!/usr/bin/env bash
set -eo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
if [[ $# -lt 2 || ! $1 =~ ^[0-7](,[0-7])*$ || ! $2 =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/train.sh GPU_IDS SEQ_LENGTH [DATA_PREFIX]" >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${PAI_ROOT:?Set PAI_ROOT to Pai-Megatron-Patch}"
: "${QWEN_HF:?Set QWEN_HF to the model/tokenizer directory}"
: "${QWEN_CHECKPOINT:?Set QWEN_CHECKPOINT to the converted Megatron checkpoint}"
patch="$PAI_ROOT"
hf="$QWEN_HF"
checkpoint="$QWEN_CHECKPOINT"
python=${PYTHON:-python3}
IFS=, read -r -a devices <<< "$1"
tp=${#devices[@]}
if [[ $tp != 1 && $tp != 2 && $tp != 4 && $tp != 8 ]]; then
    echo "Expected 1, 2, 4 or 8 tensor-parallel ranks" >&2
    exit 2
fi
status=$(nvidia-smi --id="$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if ! awk -F, -v count="$tp" '($1 + 0 > 256 || $2 + 0 > 0) { busy=1 } END { exit (busy || NR != count) }' <<< "$status"; then
    echo "Requested GPUs are in use" >&2
    exit 3
fi
[[ $tp == 2 && $2 == 4096 ]] || { echo "This release requires TP2 and 4096 tokens" >&2; exit 2; }
test -f "$checkpoint/latest_checkpointed_iteration.txt"
model_dims=$("$python" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
if config["model_type"] != "qwen3" or config["hidden_act"] != "silu" or config.get("attention_bias"):
    raise ValueError("Expected dense Qwen3 with SwiGLU and bias-free attention")
keys = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps",
        "rope_theta", "max_position_embeddings")
print(*(config[key] for key in keys), int(config["tie_word_embeddings"]))
' "$hf/config.json")
read -r layers hidden ffn heads kv_heads head_dim vocab eps rope max_seq tied <<< "$model_dims"
embedding_args=()
if [[ $tied == 0 ]]; then
    embedding_args+=(--untie-embeddings-and-output-weights)
fi
if [[ -n ${MEGATRON_RUN_DIR:-} ]]; then
    profile_dir=$MEGATRON_RUN_DIR
    mkdir -- "$profile_dir"
else
    mkdir -p "$repo/runs"
    profile_dir=$(mktemp -d "$repo/runs/train.XXXXXX")
fi
echo "Run directory: $profile_dir"
export CUDA_VISIBLE_DEVICES="$1" CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=4
parallel_args=(--tensor-model-parallel-size "$tp" --pipeline-model-parallel-size 1)
if (( tp > 1 )); then
    parallel_args+=(--sequence-parallel)
fi
data_args=(--mock-data --split 100,0,0)
if [[ $# == 3 ]]; then
    data_args=(--data-path "$3" --split 100,0,0)
fi
impl_args=(--transformer-impl transformer_engine --no-gradient-accumulation-fusion)
export PYTHONPATH="$repo/src:$patch:$patch/backends/megatron/Megatron-LM-250624:$patch/examples/qwen3"
timeout_seconds=${MEGATRON_TIMEOUT_SECONDS:-300}
if [[ ! $timeout_seconds =~ ^[1-9][0-9]*$ ]]; then
    echo "MEGATRON_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
"$python" -m tilescale_threeops.launch --timeout "$timeout_seconds" -- "$python" -m torch.distributed.run --standalone --nproc-per-node="$tp" \
    --module tilescale_threeops.train \
    --load "$checkpoint" --ckpt-format torch_dist --finetune --no-load-optim --no-load-rng \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$hf" --padded-vocab-size "$vocab" \
    --num-layers "$layers" --hidden-size "$hidden" --ffn-hidden-size "$ffn" --num-attention-heads "$heads" \
    --group-query-attention --num-query-groups "$kv_heads" --kv-channels "$head_dim" --qk-layernorm \
    --normalization RMSNorm --norm-epsilon "$eps" --swiglu --disable-bias-linear \
    --position-embedding-type rope --rotary-base "$rope" --hidden-dropout 0 --attention-dropout 0 "${impl_args[@]}" \
    --seq-length "$2" --max-position-embeddings "$max_seq" --bf16 "${parallel_args[@]}" "${embedding_args[@]}" \
    --micro-batch-size 1 --global-batch-size 1 --train-iters "${MEGATRON_TRAIN_ITERS:-201}" --lr 1e-5 \
    --dataset MMAP --train-mode pretrain "${data_args[@]}" --data-cache-path "$profile_dir/data_cache" \
    --eval-iters 0 --num-workers 0 \
    --tensorboard-dir "$profile_dir" --log-interval 1 > "$profile_dir/train.log" 2>&1
tail -15 "$profile_dir/train.log"
