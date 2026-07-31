ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
FROM ${LLAMA_IMAGE}

COPY docker/llm-entrypoint.sh /app/local-llm-entrypoint.sh
ENTRYPOINT ["/bin/sh", "/app/local-llm-entrypoint.sh"]
