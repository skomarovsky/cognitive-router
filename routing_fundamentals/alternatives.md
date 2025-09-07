Using Alternate Qwen Configs (3B + 7B AWQ) with the Router
What these are

Two drop-in config.yaml variants optimized for smaller GPUs (≈11 GB):

Qwen2.5-3B-Instruct (easy fit)

config.qwen3b.vllm.yaml

Qwen2.5-7B-Instruct (AWQ quantized) (fits with tighter limits)

config.qwen7b_awq.vllm.yaml

Where to put them

Place either file beside your current config.yaml at your project root, e.g.

/home/stan/wss/cognitive-router/

How to use them (CLI)

Point the router to the config you want with --config:

# 3B config
poetry run router --config config.qwen3b.vllm.yaml rank "write a python function that validates emails"
poetry run router --config config.qwen3b.vllm.yaml --deterministic chat "write a python function that validates emails"

# 7B AWQ config
poetry run router --config config.qwen7b_awq.vllm.yaml rank "optimize this SQL query"

Start vLLM to match the config
3B server

Make sure the served model name matches generator.openai_model in the config.

poetry run vllm serve Qwen/Qwen2.5-3B-Instruct \
  --dtype bfloat16 \
  --port 8000 \
  --served-model-name "Qwen2.5-3B-Instruct" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code

7B AWQ server (example repo; use your chosen AWQ checkpoint)
poetry run vllm serve TheBloke/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --calculate-kv-scales \
  --port 8000 \
  --served-model-name "Qwen2.5-7B-Instruct-AWQ" \
  --max-model-len 3072 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code

Notebook users

The playground notebook opens config.yaml by default. Either:

Rename your chosen file to config.yaml, or

Edit the notebook’s open("config.yaml") to open("config.qwen3b.vllm.yaml") (or the AWQ one).

Keeping configs in a subfolder (optional)

You can store them anywhere—just pass the path:

poetry run router --config configs/config.qwen3b.vllm.yaml rank "..."