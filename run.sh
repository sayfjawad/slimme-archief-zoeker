#!/usr/bin/env bash
# Start the politici smart-search web app (all politicians in config/, picked
# from a dropdown). Default http://localhost:8902; override with PORT=.
cd "$(dirname "$0")"
export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# AI answers use a local OpenAI-compatible LLM. Default: auto-discovers the
# scrib-r llama.cpp container. Override: export LLM_BASE_URL / LLM_MODEL_ID.
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8902}"
