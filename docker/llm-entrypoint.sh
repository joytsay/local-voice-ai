#!/bin/sh
set -eu

if [ -n "${LLM_MODEL_PATH:-}" ]; then
    model_name="${LLM_MODEL_PATH##*/}"
else
    model_name="${LLM_MODEL:-${LLM_HF_REPO:-local-model}}"
fi

set -- \
    --host 0.0.0.0 \
    --port 8080 \
    --alias "${model_name}"

if [ -n "${GPU_LLAMA_N_GPU_LAYERS:-}" ]; then
    set -- "$@" --n-gpu-layers "${GPU_LLAMA_N_GPU_LAYERS}"
fi

if [ "${LLM_EMBEDDING:-0}" = "1" ]; then
    set -- "$@" --embedding
fi

if [ -n "${LLM_POOLING:-}" ]; then
    set -- "$@" --pooling "${LLM_POOLING}"
fi

if [ -n "${LLM_MODEL_PATH:-}" ]; then
    if [ ! -f "${LLM_MODEL_PATH}" ]; then
        echo "LLM_MODEL_PATH does not exist inside the container: ${LLM_MODEL_PATH}" >&2
        exit 1
    fi
    exec /app/llama-server "$@" --model "${LLM_MODEL_PATH}"
fi

exec /app/llama-server "$@" \
    --hf-repo "${LLM_HF_REPO:-bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M}"
