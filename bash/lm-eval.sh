#!/usr/bin/env bash
# Zero-shot accuracy on 7 QA benchmarks for a saved (ternarized) model.
set -e

CKPT=./output/Llama-2-7b-hf_wikitext2_atq_groupsize_128_ssr_True_nsamples_128.pt

CUDA_VISIBLE_DEVICES=0 python run_lm_eval.py \
    --model_name_or_path "$CKPT" \
    --tasks "piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa,boolq"
