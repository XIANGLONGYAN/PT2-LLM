import argparse
import importlib
import numpy as np
import random, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, LlamaForCausalLM, modeling_utils

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name_or_path', type=str, help='model to load')
    parser.add_argument('--lt_bits', type=int, help='Bits for low_rank latents. \
                        When lt_bits is less than 16, we quantize the low_rank latents in low_bits', default=16)
    parser.add_argument('--lt_group_size', type=int, help='Group size for low_rank latents', default=0)
    parser.add_argument('--lt_sym', action='store_true', help='Symmetric quantization for low_rank latents')
    parser.add_argument('--lt_clip_ratio', type=float, help='Clip ratio for low_rank latents', default=1.0)
    parser.add_argument('--lt_hadamard', action='store_true', help='Apply Hadamard transform to low_rank latents')
    parser.add_argument('--flash2', action='store_true', help='whether to use flash-attention2')
    return parser


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child,
                        layers=layers,
                        name=name + '.' + name1 if name != '' else name1))
    return res

# slow inference, just to get perplexity/accuracy
def load_quant(model, checkpoint, eval=True):
    config = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=False,
        trust_remote_code=True,
    )
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False) # weird version issue with rotary_emb

    model.seqlen = 2048
    # model.seqlen = 4096
    print('Done.')

    return model, tokenizer, config

def load_model_and_tokenizer(model_name_or_path, use_flash_attn2=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        # attn_implementation="flash_attention_2" if use_flash_attn2 else "sdpa",
    )
    model.eval()
    # Fix the bug in generation configs
    #TODO: Add reference to the issue that also faced this bug
    if "vicuna" in model.config._name_or_path.lower() or "longchat" in model.config._name_or_path.lower():
        model.generation_config.do_sample = True
        
    return model, tokenizer