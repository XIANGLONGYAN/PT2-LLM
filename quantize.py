import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from pt2_llm.quantizer import TernaryQuantizer
from pt2_llm.model_utils import find_layers, cleanup_memory
from pt2_llm.data import get_tokenizer
from pt2_llm.gptq_ssr import GPTQ_SSR
from pt2_llm.gptq import GPTQ
from pt2_llm import model_utils
import logging


def setup_logger(log_file):
    """Console shows concise INFO logs; the log file keeps full DEBUG detail."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_model(model, calib_seqlen=2048):
    import torch

    def skip(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    if "opt" in model.lower():
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model.lower():
        from transformers import LlamaForCausalLM
        
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = calib_seqlen
    elif "qwen" in model.lower():
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = calib_seqlen

    return model

'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    logging.debug("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model.lower():
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif "qwen" in args.model.lower():
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module, attention_type=None):
            super().__init__()
            self.module = module
            self.attention_type = attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            raise ValueError
    if "qwen" in args.model.lower():
        attention_type = layers[0].attention_type
    else:
        attention_type = None
    layers[0] = Catcher(layers[0], attention_type)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "qwen" in args.model.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    
    if "opt" in args.model.lower():
        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.out_proj'],
            ['fc1'],
            ['fc2']
        ]
    else:
        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]

    fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    logging.info(f"Ternarizing {len(layers)} layers ...")

    for i in tqdm(range(len(layers)), desc="Ternarizing layers", unit="layer"):
        layer = layers[i].to(dev)

        full = find_layers(layer)
        
        fp_inputs_cache.add_hook(full)
        for j in range(args.nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        fp_inputs_cache.clear_hook()
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            gptq = {}
            for name in subset:
                braq_quantizer = TernaryQuantizer(
                    subset[name].weight,
                    method=args.low_quant_method,
                    groupsize=groupsize,
                )
                if args.ssr:
                    # SSR reorders columns; skip it on the down-projection layers.
                    if 'fc2' in name or 'mlp.down_proj' in name:
                        gptq[name] = GPTQ(
                            subset[name],
                            braq_quantizer,
                            salient_metric=args.salient_metric,
                            disable_gptq=args.disable_gptq,
                            method=args.low_quant_method,
                            gptaq=True,
                            reorder=False,
                        )
                    else:
                        gptq[name] = GPTQ_SSR(
                            subset[name],
                            braq_quantizer,
                            salient_metric=args.salient_metric,
                            disable_gptq=args.disable_gptq,
                            method=args.low_quant_method,
                            gptaq=True,
                            reorder=True,
                        )
                else:
                    gptq[name] = GPTQ(
                        subset[name],
                        braq_quantizer,
                        salient_metric=args.salient_metric,
                        disable_gptq=args.disable_gptq,
                        method=args.low_quant_method,
                        gptaq=True,
                        reorder=False,
                    )
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp
        
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            for name in subset:
                logging.debug(f'[layer {i}] quantizing {name}')

                info = gptq[name].fasterquant(
                    percdamp=args.percdamp, 
                    blocksize=args.blocksize,
                    num_p=args.num_p,
                    disable_mask=True,
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        fp_inputs_cache.clear_cache()


        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)

if __name__ == "__main__":
    import argparse
    from pt2_llm.data import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=['atq', 'atq-itf', 'atq-aga', 'ternary-init', 'fp16'],
        help="Ternarization method. atq=full ATQ (ITF+AGA); atq-itf=ITF only; "
             "atq-aga=AGA only; ternary-init=asymmetric init only; fp16=no quantization.",
    )
    parser.add_argument(
        "--order2_group",
        action='store_true',
        help="division for salient weights",
    )
    parser.set_defaults(order2_group=False)
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--num_p",
        type=int,
        default=1,
        help="Number of division for non-salient weights",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="hessian",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--calib_seqlen", type=int, default=2048, help="Calibration seqlen."
    )
    parser.add_argument(
        "--ppl_seqlen", type=int, default=2048, help="PPL seqlen."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the fake-quantized model.",
    )
    parser.add_argument(
        "--ssr",
        action="store_true",
        help="Enable Structural Similarity-based Reordering (SSR).",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device

    if args.load_quantized:
        model = get_model(args.model, args.calib_seqlen)
        model.eval()
        save_title = f"{args.model.split('/')[-1]}_{args.dataset}_{args.low_quant_method}_groupsize_{groupsize}_ssr_{args.ssr}"
        save_file = "./output/" + save_title.replace("/", "_") + ".pt"
        
    else: 
        model = get_model(args.model, args.calib_seqlen)
        model.eval()
            
        save_title = f"{args.model.split('/')[-1]}_{args.dataset}_{args.low_quant_method}_groupsize_{groupsize}_ssr_{args.ssr}_nsamples_{args.nsamples}"
        save_file = "./output/" + save_title.replace("/", "_") + ".pt"
        log_file = "./log/" + save_title.replace("/", "_") + f"_{args.experiment}" + ".log"
        log_path = os.path.dirname(log_file)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = setup_logger(log_file)
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        quant_sequential(model, dataloader, device)
        logging.info(f"Quantization time: {time.time() - tick:.1f}s")

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)

    from pt2_llm.eval_ppl import opt_eval, llama_eval, qwen_eval
    ppl_results = {}
    for dataset in ["wikitext2", "c4"]:
        _, testloader = get_loaders(dataset, seed=args.seed, seqlen=2048, model=args.model)
        if "opt" in args.model.lower():
            ppl_results[dataset] = opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "qwen" in args.model.lower():
            ppl_results[dataset] = qwen_eval(model, testloader, device, dataset, args.log_wandb, args.ppl_seqlen)
        else:
            ppl_results[dataset] = llama_eval(model, testloader, device, dataset, args.log_wandb, args.ppl_seqlen)

    logging.info("=" * 44)
    logging.info(f"  Perplexity  |  method={args.low_quant_method}  ssr={args.ssr}")
    logging.info("-" * 44)
    for name, ppl in ppl_results.items():
        logging.info(f"  {name:<10s} : {ppl:8.4f}")
    logging.info("=" * 44)
