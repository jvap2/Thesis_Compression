import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from FP_Quantization_Experiments import brecq_quantize_exp_fp, brecq_quantize_exp_fp_scale, quantize_model_fp, QuantConv2dFP, QuantLinearFP, apply_smoothquant
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import torch.nn.functional as F
from torch import nn
import math
bitwidth = 4
e_bits = 2
m_bits = 1
e_scale_bits = 8
m_scale_bits = 0
blocksize = 64
batch_size = 8
from huggingface_hub import login

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OPT_MODELS = {
    "125m": "facebook/opt-125m",
    "1.3b": "facebook/opt-1.3b",
    "2.7b": "facebook/opt-2.7b",
    "6.7b": "facebook/opt-6.7b",
    "13b": "facebook/opt-13b",
    "llama-1b":"meta-llama/Llama-3.2-1B",
        # GPT-2 family — very lightweight, good sanity-check baseline
    # Papers like GPTQ and LLM-FP4 use these as small-scale references
    "gpt2":      "gpt2",           # 117M
    "gpt2-med":  "gpt2-medium",    # 345M
 
    # BLOOM small variants — multilingual, different architecture
    "bloom-560m": "bigscience/bloom-560m",
     "bloom-1b1": "bigscience/bloom-1b1",  # 1.1B, still manageable on a single GPU
    # Pythia family — EleutherAI, great for reproducibility
    # Trained on known data (The Pile), many sizes available
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b":   "EleutherAI/pythia-1b",
 
    # Uncomment if your GPU can handle these:
    # "opt-2.7b":       "facebook/opt-2.7b",
    # "opt-6.7b":       "facebook/opt-6.7b",
    "pythia-1.4b":    "EleutherAI/pythia-1.4b",
    # "llama2-7b":      "meta-llama/Llama-2-7b-hf",   # requires HF access token
}


DATASETS = {
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
    # "ptb":        ("ptb_text_only", "penn_treebank"),  # uncomment to add PTB
    # "c4":         ("c4", "en"),                        # large — slow to download
}
# def compute_perplexity(model, tokenizer, dataset, device="cuda"):
#     model.eval()
#     total_loss = 0
#     total_tokens = 0

#     for text in dataset["text"]:
#         if len(text.strip()) == 0:
#             continue

#         enc = tokenizer(text, return_tensors="pt").to(device)
#         input_ids = enc.input_ids

#         with torch.no_grad():
#             outputs = model(input_ids, labels=input_ids)
#             loss = outputs.loss

#         total_loss += loss.item() * input_ids.size(1)
#         total_tokens += input_ids.size(1)

#     ppl = math.exp(total_loss / total_tokens)
#     return ppl


# def compute_standard_ppl(model, tokenizer, dataset, stride=512, device="cuda"):
#     model.eval()
    
#     # 1. Join all text segments into one massive string
#     # Assuming 'dataset' is a Hugging Face dataset with a 'text' column
#     full_text = "\n\n".join(dataset["text"])
    
#     # 2. Tokenize the whole thing
#     encodings = tokenizer(full_text, return_tensors="pt").to(device)
    
#     # max_length is usually 2048 for OPT
#     max_length = model.config.max_position_embeddings
#     seq_len = encodings.input_ids.size(1)

#     nll_sum = 0.0
#     n_tokens = 0
    
#     # 3. Sliding Window
#     for begin_loc in range(0, seq_len, stride):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - begin_loc  # Length of the current window
        
#         input_ids = encodings.input_ids[:, begin_loc:end_loc]
#         target_ids = input_ids.clone()
        
#         # In standard sliding window PPL, we often mask the context 
#         # that was seen in the previous window, but for a simple 
#         # benchmark, calculating loss on the whole window is fine.
        
#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#             # outputs.loss is the average NLL over the window
#             nll_sum += outputs.loss * trg_len
#             n_tokens += trg_len

#     ppl = torch.exp(nll_sum / n_tokens)
#     return ppl.item()


def compute_standard_ppl(model, tokenizer, dataset, stride=512, device="cuda"):
    model.eval()
    full_text  = "\n\n".join(dataset["text"])
    encodings  = tokenizer(full_text, return_tensors="pt").to(device)

    # Use model's native max length — cap generously for memory safety
    if hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        max_length = model.config.seq_length
    else:
        max_length = 2048

    # Model-specific caps to prevent OOM on large context models
    model_type = getattr(model.config, "model_type", "")
    if model_type == "gpt2":
        max_length = min(max_length, 1024)
    elif model_type == "llama":
        max_length = min(max_length, 2048)  # LLaMA-3.2 supports 131k but 2048 is standard for benchmarking
    else:
        max_length = min(max_length, 2048)  # safe default for any other model

    # Make sure stride is less than max_length
    stride = min(stride, max_length // 2)

    seq_len = encodings.input_ids.size(1)
    nll_sum, n_tokens = 0.0, 0

    for begin_loc in range(0, seq_len, stride):
        end_loc   = min(begin_loc + max_length, seq_len)
        trg_len   = end_loc - begin_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids.clone())
            loss    = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        nll_sum  += loss.item() * trg_len
        n_tokens += trg_len

    if n_tokens == 0:
        return float('inf')
    return math.exp(nll_sum / n_tokens)

def collate_fn(batch, tokenizer, seq_len=512, device="cuda"):
    texts = [item["text"] for item in batch if len(item["text"].strip()) > 0]
    
    if not texts:
        return None, None

    # Ensure pad token exists before calling with padding=True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len
    )

    input_ids = enc.input_ids.to(device)
    return input_ids, input_ids

def get_llm_dataloader(dataset, tokenizer, batch_size=8):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
res_file = "quant_res.csv"
model_name = OPT_MODELS["1.3b"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)
model = model.float()  # cast to float32 just before quantize_model_fp
model.eval()
# Run this on the original unquantized model
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name, module.weight.shape, module.in_features, module.out_features)
        break
print(f"After loading model:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

data ="wikitext-2-raw-v1"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

ppl_fp32 = compute_standard_ppl(model, tokenizer, dataset)
print("FP32 PPL:", ppl_fp32)
calib_loader = get_llm_dataloader(dataset, tokenizer, batch_size=batch_size)
hadamard = True
import gc
torch.cuda.empty_cache()
gc.collect()
if "bloom" in model_name:
    print("Applying SmoothQuant...")
    model = apply_smoothquant(model)
    # Run this BEFORE quantization, AFTER smoothing
    # to confirm smoothing actually changed the weight distribution
    for name, module in model.named_modules():
        if name == "transformer.h.0.self_attention.query_key_value":
            w = module.weight.data
            print(f"=== After smoothing, before quantization ===")
            print(f"weight shape: {w.shape}")
            print(f"weight max:   {w.abs().max():.6f}")
            print(f"weight mean:  {w.abs().mean():.6f}")
            print(f"weight std:   {w.std():.6f}")
            print(f"values > 0.1: {(w.abs() > 0.1).float().mean():.4f}")
            print(f"values > 0.5: {(w.abs() > 0.5).float().mean():.4f}")
            print(f"values > 1.0: {(w.abs() > 1.0).float().mean():.4f}")
            print(f"values > 2.0: {(w.abs() > 2.0).float().mean():.4f}")
            break

    # Also check the LayerNorm to confirm scale was absorbed
    for name, module in model.named_modules():
        if name == "transformer.h.0.input_layernorm":
            print(f"\n=== input_layernorm after smoothing ===")
            print(f"weight max:  {module.weight.data.abs().max():.4f}")
            print(f"weight mean: {module.weight.data.abs().mean():.4f}")
            break
    # Before quantization, after smoothing
    for name, module in model.named_modules():
        if name == "transformer.h.0.self_attention.query_key_value":
            print(type(module))           # should be nn.Linear, NOT QuantLinearFP
            print(module.weight.shape)    # should exist directly on module
            break
    test_input = torch.randint(0, 1000, (1, 32)).to(device)
    # Print BLOOM block structure to see exact attribute names
    # Print full BLOOM block 0 structure
    for name, module in model.named_modules():
        if name.startswith("transformer.h.0") and not any(
            name.startswith(f"transformer.h.0.{x}.")
            for x in ["self_attention", "mlp"]
        ):
            print(f"{name}: {type(module).__name__}")
    with torch.no_grad():
        out_orig  = model(test_input, labels=test_input)
quant_model = quantize_model_fp(model,calib_loader, block_size=blocksize,e_bits=e_bits,m_bits=m_bits,e_bits_scale=e_scale_bits,m_bits_scale=m_scale_bits, device = device, use_HG=False, use_Hessian=False, use_adap= (not hadamard), use_forward=False, Hadamard=hadamard)

if "bloom" in model_name:
    # After quantization
    for name, module in quant_model.named_modules():
        if name == "transformer.h.0.self_attention.query_key_value":
            print(f"\n=== Quantized weight stats ===")
            print(f"weight_q max:  {module.weight_q.abs().max():.6f}")
            print(f"weight_q mean: {module.weight_q.abs().mean():.6f}")
            
            # Direct comparison
            w_orig  = module.linear.weight.data
            w_quant = module.weight_q
            
            abs_err = (w_orig - w_quant).abs()
            rel_err = abs_err / w_orig.abs().clamp(min=1e-8)
            
            print(f"abs error max:  {abs_err.max():.6f}")
            print(f"abs error mean: {abs_err.mean():.6f}")
            print(f"rel error mean: {rel_err.mean():.6f}")
            
            # Check if outliers remain after smoothing
            print(f"weight_q values > 0.5: {(w_quant.abs() > 0.5).float().mean():.4f}")
            print(f"weight_q values > 1.0: {(w_quant.abs() > 1.0).float().mean():.4f}")
            break
    for name, module in quant_model.named_modules():
        if name == "transformer.h.0.self_attention.query_key_value":
            print(f"module.linear.weight max: {module.linear.weight.abs().max():.4f}")
            print(f"module.weight_q max:      {module.weight_q.abs().max():.4f}")
            
            x_test = torch.randn(1, 1024).to(device)
            
            # What the original linear would produce
            out_linear = F.linear(x_test, module.linear.weight, module.linear.bias)
            # What weight_q produces  
            out_quant  = F.linear(x_test, module.weight_q,      module.linear.bias)
            # What QuantLinearFP.forward actually produces
            out_module = module(x_test)
            
            print(f"out_linear max: {out_linear.abs().max():.4f}")
            print(f"out_quant max:  {out_quant.abs().max():.4f}")
            print(f"out_module max: {out_module.abs().max():.4f}")
            break
    for name, module in quant_model.named_modules():
        if name == "transformer.h.0.self_attention.query_key_value":
            print(f"weight_q is None: {module.weight_q is None}")
            print(f"weight_q shape:   {module.weight_q.shape if module.weight_q is not None else 'N/A'}")
    with torch.no_grad():
        out_quant = quant_model(test_input, labels=test_input)
        
    print(f"Original loss:  {out_orig.loss:.4f}  (PPL={math.exp(out_orig.loss):.2f})")
    print(f"Quantized loss: {out_quant.loss:.4f} (PPL={math.exp(out_quant.loss):.2f})")

    del model
    torch.cuda.empty_cache()
ppl_fp4  = compute_standard_ppl(quant_model, tokenizer, dataset)


print("FP4  PPL:", ppl_fp4)

import pandas as pd
## The file is already created, so we will append to it
# columns in include net, dataset, acc_perplex_original, acc_perplex_quantized, batch_size, e_bits, m_bits, e_bits_scale, m_bits_scale, block_size
df = pd.DataFrame({
    "net": [model_name],
    "dataset": [data],
    "acc_perplex_original": [ppl_fp32],
    "acc_perplex_quantized": [ppl_fp4],
    "batch_size": [batch_size],
    "e_bits": [e_bits],
    "m_bits": [m_bits],
    "e_bits_scale": [e_scale_bits],
    "m_bits_scale": [m_scale_bits],
    "block_size": [blocksize],
    "mode": ["hadamard" if hadamard else "adap"]
})
if not os.path.exists(res_file):
    df.to_csv(res_file, index=False)
else:
    df.to_csv(res_file, mode='a', header=False, index=False)