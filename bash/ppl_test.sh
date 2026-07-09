#!/usr/bin/env bash
# Evaluate WikiText-2 / C4 perplexity of a saved (ternarized) model.
# quantize.py already reports PPL at the end; use this to re-evaluate a saved .pt.
set -e

CKPT=./output/Llama-2-7b-hf_wikitext2_atq_groupsize_128_ssr_True_nsamples_128.pt

CUDA_VISIBLE_DEVICES=0 python run_ppl_eval.py \
    --model_name_or_path "$CKPT" \
    --datasets wikitext2,c4 \
    --seqlen 2048
