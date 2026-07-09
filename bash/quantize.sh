#!/usr/bin/env bash
# Post-training ternarization with PT2-LLM. Edit MODEL to your local HF checkpoint.
set -e

MODEL=/path/to/Llama-2-7b-hf   # e.g. a local LLaMA / LLaMA-2 / LLaMA-3 / Qwen3 checkpoint
GPU=0

# ---------------------------------------------------------------------------
# Full PT2-LLM: ATQ (ITF + AGA) + SSR. The quantized model is saved to ./output/.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$GPU python quantize.py "$MODEL" wikitext2 atq \
    --blocksize 128 --ssr --save

# ---------------------------------------------------------------------------
# Ablations (paper Table 2a) -- ATQ components:
#   ternary-init : asymmetric init only        (no ITF, no AGA)
#   atq-itf      : Iterative Ternary Fitting    (ITF only)
#   atq-aga      : Activation-aware Grid Align. (AGA only)
#   atq          : full ATQ                     (ITF + AGA)
# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=$GPU python quantize.py "$MODEL" wikitext2 ternary-init --blocksize 128 --ssr
# CUDA_VISIBLE_DEVICES=$GPU python quantize.py "$MODEL" wikitext2 atq-itf      --blocksize 128 --ssr
# CUDA_VISIBLE_DEVICES=$GPU python quantize.py "$MODEL" wikitext2 atq-aga      --blocksize 128 --ssr

# Ablation on SSR (paper Table 2b): run the same command without --ssr.
# CUDA_VISIBLE_DEVICES=$GPU python quantize.py "$MODEL" wikitext2 atq --blocksize 128
