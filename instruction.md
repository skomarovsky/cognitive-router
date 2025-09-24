# Model Configuration and Usage Instructions

This document explains how to run different models described in the configuration files using both vLLM and Transformers backends.

## Available Model Configurations

### 1. Qwen2.5-7B-Instruct (vLLM) - High Performance
**File:** `config.qwen3_7b.vllm.yaml`
- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Backend:** vLLM with OpenAI-compatible API
- **Precision:** Auto (typically bfloat16/float16)
- **Use case:** High-performance inference with vLLM optimization

### 2. Qwen3-4B-Instruct-2507 (vLLM) - Latest Generation
**File:** `config.qwen3_3b.vllm.yaml`
- **Model:** Qwen/Qwen3-4B-Instruct-2507
- **Backend:** vLLM with OpenAI-compatible API
- **Precision:** half (float16) - Memory efficient
- **Use case:** Latest Qwen3 generation with improved capabilities

### 3. Qwen2.5-7B-Instruct (vLLM) - Default Configuration
**File:** `config.yaml`
- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Backend:** vLLM with OpenAI-compatible API
- **Precision:** Auto (typically bfloat16/float16)
- **Use case:** High-performance inference with vLLM optimization

### 4. Qwen2.5-3B-Instruct (vLLM)
**File:** `config.qwen3b.vllm.yaml`
- **Model:** Qwen/Qwen2.5-3B-Instruct
- **Backend:** vLLM with OpenAI-compatible API
- **Precision:** bfloat16
- **Use case:** Lightweight model with vLLM optimization

### 5. Qwen2.5-3B-Instruct (Transformers - bfloat16)
**File:** `config.qwen3b.transformers.yaml`
- **Model:** Qwen/Qwen2.5-3B-Instruct
- **Backend:** Transformers
- **Precision:** bfloat16
- **Use case:** CPU/GPU inference without vLLM server requirement

### 6. Qwen2.5-3B-Instruct (Transformers - float16)
**File:** `config.qwen3b.transformers.fp16.yaml`
- **Model:** Qwen/Qwen2.5-3B-Instruct
- **Backend:** Transformers
- **Precision:** float16
- **Use case:** Memory-efficient CPU/GPU inference

### 7. Qwen2.5-7B-Instruct-AWQ (vLLM)
**File:** `config.qwen7b_awq.vllm.yaml`
- **Model:** TheBloke/Qwen2.5-7B-Instruct-AWQ
- **Backend:** vLLM with OpenAI-compatible API
- **Precision:** 4-bit quantized (AWQ)
- **Use case:** Memory-efficient high-performance inference

## Prerequisites

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **GPU Requirements:**
   - vLLM models: Requires CUDA-compatible GPU
   - Transformers models: Can run on CPU or GPU

## Starting Model Servers

### vLLM Models (Server-based)

For configurations using vLLM backend, you need to start a vLLM server first:

#### Qwen2.5-7B-Instruct (High Performance)
```bash
uv run vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype auto \
  --port 8000 \
  --served-model-name "Qwen2.5-7B-Instruct" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
```

#### Qwen3-4B-Instruct-2507 (Latest Generation)
```bash
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype half \
  --port 8000 \
  --served-model-name "Qwen3-4B-Instruct-2507" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
```

#### Qwen2.5-7B-Instruct (Default)
```bash
uv run vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype auto \
  --port 8000 \
  --served-model-name "qwen2.5-7b-instruct" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
```

#### Qwen2.5-3B-Instruct (vLLM)
```bash
uv run vllm serve Qwen/Qwen2.5-3B-Instruct \
  --dtype bfloat16 \
  --port 8000 \
  --served-model-name "Qwen2.5-3B-Instruct" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
```

#### Qwen2.5-7B-Instruct-AWQ (Quantized)
```bash
uv run vllm serve TheBloke/Qwen2.5-7B-Instruct-AWQ \
  --dtype auto \
  --port 8000 \
  --served-model-name "Qwen2.5-7B-Instruct-AWQ" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
```

### Transformers Models (No Server Required)

For Transformers backend configurations, no separate server is needed as the model runs directly in the routing application.

## Using the Routing CLI

The routing system provides three main commands:

### 1. Rank Routes
Show ranked routes for a query with scores:

```bash
# Using Qwen2.5 7B model
uv run router --config config.qwen3_7b.vllm.yaml rank "What is machine learning?"

# Using Qwen3 4B model
uv run router --config config.qwen3_3b.vllm.yaml rank "Explain neural networks"

# Using default config
uv run router --config config.yaml rank "What is AI?"

# With explanation of score components
uv run router --config config.qwen3_7b.vllm.yaml rank "What is AI?" --explain
```

### 2. Pick Route
Select a single winning route for a query:

```bash
# Using Qwen2.5 7B model
uv run router --config config.qwen3_7b.vllm.yaml pick "How does photosynthesis work?"

# Using Qwen3 4B model
uv run router --config config.qwen3_3b.vllm.yaml pick "Explain quantum computing"

# Deterministic selection with default config
uv run router --config config.yaml --deterministic pick "What is the capital of France?"
```

### 3. Chat with Routing
Route query and generate response using the selected model:

```bash
# Chat using Qwen2.5 7B model
uv run router --config config.qwen3_7b.vllm.yaml chat "Write a Python function to calculate fibonacci numbers"

# Chat using Qwen3 4B model
uv run router --config config.qwen3_3b.vllm.yaml chat "Explain the concept of recursion"

# Chat using default configuration
uv run router --config config.yaml chat "What are the latest features in Qwen3?"

# Chat using quantized model
uv run router --config config.qwen7b_awq.vllm.yaml chat "What are the benefits of model quantization?"
```

## Changing Embedding Dimensions

### Key Points
- The `build_embedder` helper simply wraps a `SentenceTransformer` model, so the
  output dimensionality is entirely determined by the pretrained checkpoint you
  load—MiniLM produces 384-dimensional vectors, for example.
- Configuration files such as `config.yaml` select that model; swapping the
  `embedding.model` field to another SentenceTransformer checkpoint (e.g., a
  768-dim `all-mpnet-base-v2` or a 256-dim distilled variant) is the supported
  way to change the embedding size.

### How to Request a Different Embedding Size
1. **Pick a model with the desired dimensionality.** Consult the
   [SentenceTransformers model zoo](https://www.sbert.net/docs/pretrained_models.html)
   for checkpoints that emit the dimension you need. There is no runtime flag to
   “ask” a model to output a different width; you must choose a checkpoint
   trained with that width.
2. **Update your configuration.** Edit the `embedding.model` entry in your
   configuration (for example, `config.yaml`) to point at the new checkpoint
   name. When `build_embedder` runs, it will load that model and all downstream
   code will automatically see embeddings with the new dimensionality.
3. **Optional: add your own projection.** If you truly need a custom dimension
   that no pretrained checkpoint offers, implement a post-processing step (e.g.,
   an additional linear projection) after `build_embedder` returns. The current
   codebase does not include such a feature, so you would extend the embedder
   yourself if necessary.

Because dimensionality is tied to the model weights, switching models (or
adding a custom projection layer) is the supported approach.

## Configuration Details

### Router Weights
All configurations use the same routing weights:
- **sim:** 1.0 (semantic similarity)
- **kw:** 0.45 (keyword matching)
- **prior:** 0.05 (prior probability)
- **cost:** 0.2 (cost factor)
- **lat:** 0.1 (latency factor)
- **health:** 0.2 (health factor)

### Generation Parameters
All configurations use consistent generation settings:
- **max_new_tokens:** 512
- **temperature:** 0.35
- **top_p:** 0.9
- **top_k:** 40
- **repetition_penalty:** 1.05

## Performance Comparison

| Model | Backend | Size | Memory Usage | Speed | Use Case |
|-------|---------|------|--------------|-------|----------|
| Qwen2.5-7B | vLLM | ~14GB | High | Very Fast | High performance, best for complex tasks |
| Qwen3-4B | vLLM | ~4GB | Low | Very Fast | Latest generation, improved capabilities, memory efficient |
| Qwen2.5-7B | vLLM | ~14GB | High | Fast | Stable performance (default) |
| Qwen2.5-3B | vLLM | ~6GB | Medium | Fast | Lightweight, balanced |
| Qwen2.5-3B | Transformers | ~6GB | Medium | Medium | No server needed |
| Qwen2.5-7B-AWQ | vLLM | ~4GB | Low | Fast | Memory efficient |

## Troubleshooting

### vLLM Server Issues
- Ensure CUDA is properly installed and accessible
- Check GPU memory availability
- Verify port 8000 is not already in use
- **Memory Management:** Adjust `--gpu-memory-utilization` (default 0.85) to limit GPU memory usage. For example, use 0.5 for 50% utilization or 0.3 for 30% if experiencing OOM errors

### Transformers Backend Issues
- Ensure sufficient RAM for model loading
- Consider using CPU inference for smaller models
- Check that all required dependencies are installed

### Routing Issues
- Verify routes.yaml file exists and is properly formatted
- Check that embedding model is accessible
- Ensure configuration file syntax is correct

## Examples by Use Case

### High-Performance Setup
```bash
# Start 7B model server
uv run vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto

# Use for complex queries
uv run router --config config.yaml chat "Explain the theory of relativity"
```

### Memory-Efficient Setup
```bash
# Start quantized model server
uv run vllm serve TheBloke/Qwen2.5-7B-Instruct-AWQ --port 8000

# Use for resource-constrained environments
uv run router --config config.qwen7b_awq.vllm.yaml chat "What is the capital of France?"
```

### Development/Testing Setup
```bash
# Use 3B model without server setup
uv run router --config config.qwen3b.transformers.yaml rank "Test query"

## Last

This section captures any final reminders or updates that should be reviewed before running the router.
