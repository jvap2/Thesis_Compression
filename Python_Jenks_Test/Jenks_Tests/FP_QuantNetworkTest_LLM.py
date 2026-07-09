import os
# Must be set before any CUDA extension is JIT-compiled (cuda_helpers loads at import time).
# Without this, PyTorch's _get_cuda_arch_flags fails with IndexError when NVML can't init.
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from FP_Quantization_Experiments import (
    quantize_model_fp,
    QuantConv2dFP, QuantLinearFP, HadamardQuantLinearFP,
    apply_smoothquant, act_quant_mode,
    quantize_activations, quantize_activations_gf4,
    quantize_activations_gf4_adaptive, quantize_activations_gf4_residual,
    calibrate_gf4_learned_levels, calibrate_model_gf4_hsmooth, apply_gf4_hsmooth,
    apply_block_smoothquant_opt, preshifted_beta_only_mode,
    enable_fast_kernels,
    save_quantized_model, load_quantized_model,
    LLAMA_SKIP_PATTERNS,
)
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import torch.nn.functional as F
from torch import nn
import math
bitwidth = 4
e_bits = 2
m_bits = 1
e_scale_bits = 4
m_scale_bits = 3
blocksize = 16
batch_size = 8
try:
    import os as _os
    from huggingface_hub import login
    _hf_token = _os.environ.get("HF_TOKEN")
    if _hf_token:
        login(token=_hf_token)
except Exception:
    pass  # Use cached model if token is missing or invalid
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_first_valid_batch(calib_loader, device, max_attempts=50):
    """
    Pull batches from the loader until we get a non-None one.
    Returns input_ids tensor ready to use.
    """
    for i, batch in enumerate(calib_loader):
        if batch is None:
            continue
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        if x is not None:
            print(f"  Got valid batch at index {i}, shape={x.shape}")
            return x.to(device)
        if i >= max_attempts:
            break
    raise RuntimeError("Could not find a valid batch in the dataloader")


# def diagnose_activations(model, calib_loader, device, n_batches=2):
#     stats = {}
#     hooks = []

#     def make_hook(name):
#         def hook(module, inp, out):
#             x      = inp[0].detach().float()
#             x_flat = x.reshape(-1, x.shape[-1])
#             stats[name] = {
#                 "max":  x_flat.abs().max().item(),
#                 "mean": x_flat.abs().mean().item(),
#                 "p99":  torch.quantile(x_flat.abs().reshape(-1), 0.99).item(),
#                 "p999": torch.quantile(x_flat.abs().reshape(-1), 0.999).item(),
#             }
#         return hook

#     for name, module in model.named_modules():
#         # Check by class name to avoid import path mismatches
#         if type(module).__name__ in ("QuantLinearFP", "QuantConv2dFP", "QuantConv1dFP", "HadamardQuantLinearFP"):
#             hooks.append(module.register_forward_hook(make_hook(name)))

#     print(f"Registered {len(hooks)} hooks")

#     model.eval()
#     batches_run = 0
#     with torch.no_grad():
#         for i, batch in enumerate(calib_loader):
#             if batch is None:
#                 continue
#             if isinstance(batch, (list, tuple)):
#                 x = batch[0]
#             else:
#                 x = batch
#             if x is None:
#                 continue
#             model(x.to(device))
#             batches_run += 1
#             if batches_run >= n_batches:
#                 break

#     for h in hooks:
#         h.remove()

#     print(f"Ran {batches_run} batches, collected stats for {len(stats)} layers")

#     if not stats:
#         print("WARNING: No stats collected — hooks did not fire")
#         return stats

#     print(f"\n{'Layer':<60} {'max':>8} {'p99':>8} {'p999':>8} {'mean':>8}")
#     print("-" * 92)
#     for name, s in stats.items():
#         print(f"{name:<60} {s['max']:>8.3f} {s['p99']:>8.3f} "
#               f"{s['p999']:>8.3f} {s['mean']:>8.3f}")
#     return stats


# def check_activation_quantization_error(model, calib_loader, device,
#                                          block_size=16, e_bits=2, m_bits=1,
#                                          e_bits_scale=4, m_bits_scale=3,
#                                          n_batches=1):
#     errors = {}
#     hooks  = []

#     def make_hook(name):
#         def hook(module, inp, out):
#             x      = inp[0].detach().float()
#             x_q    = quantize_activations(
#                 x, block_size, e_bits, m_bits,
#                 e_bits_scale, m_bits_scale
#             ).float()
#             x_flat  = x.reshape(-1, x.shape[-1])
#             xq_flat = x_q.reshape(-1, x.shape[-1])
#             abs_err = (x_flat - xq_flat).abs()
#             rel_err = abs_err / x_flat.abs().clamp(min=1e-8)
#             snr     = 10 * torch.log10(
#                 x_flat.pow(2).mean() /
#                 (x_flat - xq_flat).pow(2).mean().clamp(min=1e-10)
#             )
#             errors[name] = {
#                 "act_max":  x_flat.abs().max().item(),
#                 "act_mean": x_flat.abs().mean().item(),
#                 "abs_err":  abs_err.mean().item(),
#                 "rel_err":  rel_err.mean().item(),
#                 "snr_db":   snr.item(),
#             }
#         return hook

#     for name, module in model.named_modules():
#         if type(module).__name__ in ("QuantLinearFP", "QuantConv2dFP", "QuantConv1dFP", "HadamardQuantLinearFP"):
#             hooks.append(module.register_forward_hook(make_hook(name)))

#     print(f"Registered {len(hooks)} hooks for error checking")

#     model.eval()
#     batches_run = 0
#     with torch.no_grad():
#         for i, batch in enumerate(calib_loader):
#             if batch is None:
#                 continue
#             if isinstance(batch, (list, tuple)):
#                 x = batch[0]
#             else:
#                 x = batch
#             if x is None:
#                 continue
#             model(x.to(device))
#             batches_run += 1
#             if batches_run >= n_batches:
#                 break

#     for h in hooks:
#         h.remove()

#     print(f"Ran {batches_run} batches, collected error stats for {len(errors)} layers")

#     if not errors:
#         print("WARNING: No activation error stats collected — hooks did not fire")
#         return errors

#     print(f"\n{'Layer':<55} {'act_max':>8} {'act_mean':>8} "
#           f"{'rel_err':>8} {'SNR_dB':>8}")
#     print("-" * 90)
#     for name, s in errors.items():
#         flag = " <-- BAD" if s["snr_db"] < 10 else ""
#         print(f"{name:<55} {s['act_max']:>8.2f} {s['act_mean']:>8.2f} "
#               f"{s['rel_err']:>8.3f} {s['snr_db']:>8.1f}{flag}")
#     return errors
# Run before PPL evaluation

def diagnose_activations(model, calib_loader, device, n_batches=2):
    stats = {}
    hooks = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])

            # For Hadamard wrappers, also show post-Hadamard stats
            if type(mod).__name__ == "HadamardQuantLinearFP":
                if mod.had_block_size is not None and mod.D is not None:
                    from FP_Quantization_Experiments import fwht_blockwise
                    import torch.nn.functional as _F
                    _D = mod.D.to(device=x_flat.device, dtype=x_flat.dtype)
                    if x_flat.shape[-1] < _D.shape[-1]:   # pad M -> P (padded Hadamard)
                        x_flat = _F.pad(x_flat, (0, _D.shape[-1] - x_flat.shape[-1]))
                    x_had = fwht_blockwise(x_flat * _D, mod.had_block_size)
                    # If mean subtraction is active, apply it too
                    if mod.mu is not None:
                        x_had = x_had - mod.mu.to(device=x_flat.device,
                                                    dtype=x_flat.dtype)
                    x_flat = x_had   # report post-Hadamard stats

            stats[name] = {
                "max":  x_flat.abs().max().item(),
                "mean": x_flat.abs().mean().item(),
                "p99":  torch.quantile(x_flat.abs().reshape(-1),
                                       0.99).item(),
                "p999": torch.quantile(x_flat.abs().reshape(-1),
                                       0.999).item(),
            }
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ in ("QuantLinearFP", "QuantConv2dFP",
                                     "QuantConv1dFP", "HadamardQuantLinearFP"):
            # Skip inner QuantLinearFP that lives inside a wrapper
            if ".inner" in name:
                continue
            hooks.append(module.register_forward_hook(
                make_hook(name, module)))

    print(f"Registered {len(hooks)} hooks")

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= n_batches:
                break

    for h in hooks:
        h.remove()

    print(f"Ran {batches_run} batches, collected stats for {len(stats)} layers")

    if not stats:
        print("WARNING: No stats collected — hooks did not fire")
        return stats

    print(f"\n{'Layer':<60} {'max':>8} {'p99':>8} {'p999':>8} {'mean':>8}")
    print("-" * 92)
    for name, s in stats.items():
        print(f"{name:<60} {s['max']:>8.3f} {s['p99']:>8.3f} "
              f"{s['p999']:>8.3f} {s['mean']:>8.3f}")
    return stats


def check_activation_quantization_error(model, calib_loader, device,
                                         block_size=16, e_bits=2, m_bits=1,
                                         e_bits_scale=4, m_bits_scale=3,
                                         n_batches=1):
    errors = {}
    hooks  = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            x      = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])

            # For Hadamard wrappers, apply the full pre-quantization
            # pipeline so we measure error on the actual quantized signal
            if type(mod).__name__ == "HadamardQuantLinearFP":
                if mod.had_block_size is not None and mod.D is not None:
                    from FP_Quantization_Experiments import fwht_blockwise
                    import torch.nn.functional as _F
                    _D = mod.D.to(device=x_flat.device, dtype=x_flat.dtype)
                    if x_flat.shape[-1] < _D.shape[-1]:   # pad M -> P (padded Hadamard)
                        x_flat = _F.pad(x_flat, (0, _D.shape[-1] - x_flat.shape[-1]))
                    x_flat = fwht_blockwise(x_flat * _D, mod.had_block_size)
                if mod.mu is not None:
                    x_flat = x_flat - mod.mu.to(device=x_flat.device,
                                                  dtype=x_flat.dtype)

            x_q = quantize_activations(
                x_flat, block_size, e_bits, m_bits,
                e_bits_scale, m_bits_scale
            ).float()

            abs_err = (x_flat - x_q).abs()
            rel_err = abs_err / x_flat.abs().clamp(min=1e-8)
            snr     = 10 * torch.log10(
                x_flat.pow(2).mean() /
                (x_flat - x_q).pow(2).mean().clamp(min=1e-10)
            )
            errors[name] = {
                "act_max":  x_flat.abs().max().item(),
                "act_mean": x_flat.abs().mean().item(),
                "abs_err":  abs_err.mean().item(),
                "rel_err":  rel_err.mean().item(),
                "snr_db":   snr.item(),
            }
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ in ("QuantLinearFP", "QuantConv2dFP",
                                     "QuantConv1dFP", "HadamardQuantLinearFP"):
            if ".inner" in name:
                continue
            hooks.append(module.register_forward_hook(
                make_hook(name, module)))

    print(f"Registered {len(hooks)} hooks for error checking")

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= n_batches:
                break

    for h in hooks:
        h.remove()

    print(f"Ran {batches_run} batches, collected error stats "
          f"for {len(errors)} layers")

    if not errors:
        print("WARNING: No activation error stats collected — "
              "hooks did not fire")
        return errors

    print(f"\n{'Layer':<55} {'act_max':>8} {'act_mean':>8} "
          f"{'rel_err':>8} {'SNR_dB':>8}")
    print("-" * 90)
    for name, s in errors.items():
        flag = " <-- BAD" if s["snr_db"] < 10 else ""
        print(f"{name:<55} {s['act_max']:>8.2f} {s['act_mean']:>8.2f} "
              f"{s['rel_err']:>8.3f} {s['snr_db']:>8.1f}{flag}")
    return errors

OPT_MODELS = {
    "125m": "facebook/opt-125m",
    "1.3b": "facebook/opt-1.3b",
    "2.7b": "facebook/opt-2.7b",
    "6.7b": "facebook/opt-6.7b",
    "13b": "facebook/opt-13b",
    "llama-1b":"meta-llama/Llama-3.2-1B",
    "llama-3b":"meta-llama/Llama-3.2-3B",
    "llama-7b":"meta-llama/Llama-2-7b-hf",
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


def _tokenize_corpus(tokenizer, dataset, device="cpu", add_special_tokens=True):
    """Tokenize once and keep on CPU; callers slice and move to GPU as needed."""
    full_text = "\n\n".join(dataset["text"])
    return tokenizer(full_text, return_tensors="pt",
                     add_special_tokens=add_special_tokens).input_ids.squeeze(0)  # [total_tokens]


def _get_max_length(model):
    model_type = getattr(model.config, "model_type", "")
    if hasattr(model.config, "max_position_embeddings"):
        raw = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        raw = model.config.seq_length
    else:
        raw = 2048
    # GPT-2 native context is 1024; LLaMA supports huge context but 2048 matches benchmarks
    cap = 1024 if model_type == "gpt2" else 2048
    return min(raw, cap)


def _chunk_nll(model, chunk: torch.Tensor) -> torch.Tensor | None:
    """
    Forward one [B, T] chunk, return mean NLL scalar (CUDA tensor).
    Computes cross-entropy from logits directly so we can del the logit
    tensor immediately — avoids the 200 MB/pass VRAM accumulation that
    happens when HuggingFace retains logits + past_key_values in outputs.
    """
    with torch.inference_mode():
        logits = model(chunk, use_cache=False).logits          # [B, T, V]
        shift_logits = logits[:, :-1, :].contiguous()          # [B, T-1, V]
        shift_labels = chunk[:, 1:].contiguous()               # [B, T-1]
        del logits                                             # free 200 MB now
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        del shift_logits, shift_labels
    if torch.isnan(loss) or torch.isinf(loss):
        return None
    return loss


def compute_standard_ppl(model, tokenizer, dataset, stride=None, device="cuda",
                          _input_ids=None):
    """
    Sliding-window perplexity.  stride defaults to max_length (non-overlapping),
    which is 4× faster than stride=512 with negligible PPL difference for
    benchmarking.  Pass stride < max_length for higher-accuracy context overlap.

    Pass _input_ids (CPU LongTensor, 1-D) to skip re-tokenization.
    """
    model.eval()
    max_length = _get_max_length(model)
    if stride is None:
        stride = max_length
    stride = min(stride, max_length)

    if _input_ids is None:
        _input_ids = _tokenize_corpus(tokenizer, dataset)

    seq_len  = _input_ids.size(0)
    nll_sum  = 0.0
    n_tokens = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        chunk   = _input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        loss    = _chunk_nll(model, chunk)
        del chunk
        if loss is None:
            continue
        trg_len   = end_loc - begin_loc - 1      # T-1 predicted tokens
        nll_sum  += loss.item() * trg_len
        n_tokens += trg_len

    if n_tokens == 0:
        return float('inf')
    return math.exp(nll_sum / n_tokens)


def compute_ppl_gptq_style(model, tokenizer, dataset,
                            seq_len=2048, device="cuda", batch_size=4,
                            _input_ids=None):
    """
    GPTQ paper evaluation protocol: non-overlapping seq_len chunks, mean NLL
    exponentiated.  batch_size=4 balances throughput vs VRAM for a 1-2 B model;
    reduce to 2 or 1 if you hit OOM on a smaller GPU.

    Pass _input_ids (CPU LongTensor, 1-D) to skip re-tokenization.
    """
    model.eval()

    if _input_ids is None:
        _input_ids = _tokenize_corpus(tokenizer, dataset, add_special_tokens=False)

    total_tokens = _input_ids.size(0)
    n_chunks     = total_tokens // seq_len
    chunks       = _input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)  # CPU

    nll_sum  = 0.0
    n_tokens = 0

    for i in range(0, n_chunks, batch_size):
        batch = chunks[i:i + batch_size].to(device)   # [B, seq_len]
        loss  = _chunk_nll(model, batch)
        del batch
        if loss is None:
            continue
        b = min(batch_size, n_chunks - i)
        nll_sum  += loss.item() * b * (seq_len - 1)
        n_tokens += b * (seq_len - 1)

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
# Model selection: set the MODEL env var (e.g. MODEL=llama-1b) to override
# without editing this file — convenient on Colab. Falls back to the default.
_model_key = os.environ.get("MODEL", "2.7b")
model_name = OPT_MODELS.get(_model_key, OPT_MODELS["2.7b"])
print(f"Selected model: {_model_key} -> {model_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)
# model = model.float()  # cast to float32 just before quantize_model_fp
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
# Newer datasets/huggingface_hub (e.g. on Colab) reject the legacy bare name
# "wikitext" and require the canonical namespace/name. Fall back across the
# known hosts so this runs on both old and new library versions.
def _load_wikitext():
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            return load_dataset(repo, "wikitext-2-raw-v1", split="test")
        except Exception as _e:
            print(f"  load_dataset({repo!r}) failed: {type(_e).__name__}; trying next")
    raise RuntimeError("Could not load wikitext-2-raw-v1 from any known repo")

dataset = _load_wikitext()

# Tokenize once; reuse across all PPL calls to avoid re-tokenizing per mode
_ppl_ids      = _tokenize_corpus(tokenizer, dataset)
_ppl_ids_gptq = _tokenize_corpus(tokenizer, dataset, add_special_tokens=False)

ppl_fp32 = compute_standard_ppl(model, tokenizer, dataset, _input_ids=_ppl_ids)
print("FP32 PPL (sliding):", ppl_fp32)
torch.cuda.empty_cache()

ppl_fp32_gptq = compute_ppl_gptq_style(model, tokenizer, dataset,
                                        seq_len=2048, _input_ids=_ppl_ids_gptq)
print("FP32 GPTQ PPL:", ppl_fp32_gptq)
torch.cuda.empty_cache()
ppl_fp4_a16           = float('inf')
ppl_fp4_a16_gptq      = float('inf')
ppl_fp4_nv            = float('inf')
ppl_fp4_nv_gptq       = float('inf')
ppl_fp4_gf4           = float('inf')
ppl_fp4_gf4_gptq      = float('inf')
ppl_fp4_adaptive      = float('inf')
ppl_fp4_adaptive_gptq = float('inf')
ppl_fp4_residual      = float('inf')
ppl_fp4_residual_gptq = float('inf')
ppl_fp4_hsmooth       = float('inf')
ppl_fp4_hsmooth_gptq  = float('inf')
ppl_fp4_learned       = float('inf')
ppl_fp4_learned_gptq  = float('inf')
hadamard = False  # safe default in case try block OOMs before setting it
try:
    # Large models OOM during the Hadamard-Hessian forward pass (attention is
    # O(seq²) and hooks promote activations to fp32). Shrink the calibration
    # batch for >2B-param models; calibration only needs enough tokens for
    # stable block statistics, so we trade batch for more batches elsewhere.
    _big_model = any(s in model_name.lower()
                     for s in ("2.7b", "6.7b", "7b", "13b", "8b", "3b"))
    _calib_bs  = 2 if _big_model else batch_size
    if _big_model:
        print(f"Large model detected — calibration batch_size={_calib_bs}")
    calib_loader = get_llm_dataloader(dataset, tokenizer, batch_size=_calib_bs)
    hadamard = False
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    # model = apply_block_smoothquant_opt(model, calib_loader, device, block_size=blocksize, alpha=0.5, num_batches=4)
    if "bloom" in model_name:
        print("Applying SmoothQuant...")
        model = apply_smoothquant(model, calib_loader, device=device)
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
    # Keep the MLP-output projection (the massive-activation / down_proj layer)
    # in FP16. It carries per-token outliers a single FP4 term can't span, so
    # quantizing it collapses single-term W4A4 (OPT-6.7b: fc2 quantized -> 1745
    # ppl; Llama-7b: down_proj skipped -> survives 5.86). Per-architecture name:
    #   Llama/Mistral SwiGLU MLP -> down_proj ;  OPT/GPT-2 MLP -> fc2.
    # lm_head is already skipped by the quantizer's default head handling.
    _mn = model_name.lower()
    if "llama" in _mn or "mistral" in _mn:
        _extra_skip = LLAMA_SKIP_PATTERNS          # ("down_proj",)
    elif "opt" in _mn or "gpt" in _mn:
        _extra_skip = ("fc2",)                     # OPT/GPT-2 MLP-output analog
    else:
        _extra_skip = tuple(LLAMA_SKIP_PATTERNS) + ("fc2",)  # default: skip both variants
    if _extra_skip:
        print(f"Keeping MLP-output layer in FP16 (outlier-safe): {_extra_skip}")
    quant_model = quantize_model_fp(model,calib_loader, block_size=blocksize,e_bits=e_bits,m_bits=m_bits,e_bits_scale=e_scale_bits,m_bits_scale=m_scale_bits, device = device, use_HG=False, use_Hessian=False, use_adap= False, use_forward=False, Hadamard=True, joint=False, preshift=False, decompose=False, had_block_size="auto", use_gf4=True, extra_skip_patterns=_extra_skip)
    act_stats = diagnose_activations(quant_model, calib_loader, device)
    err_stats = check_activation_quantization_error(
        quant_model, calib_loader, device,
        block_size=16, e_bits=2, m_bits=1,
        e_bits_scale=4, m_bits_scale=3
    )
    print(f"\nAfter quantization:")
    print(act_stats)
    print(err_stats)
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

    # Switch Hadamard layers to Triton fast kernels (HadaCore FWHT + GF4 quant).
    # This replaces the Python FWHT with the single-kernel L2-cached version
    # (~5× faster) and the GF4 quantize with the Triton kernel (~2.7× faster).
    # Falls back to Python automatically for custom GF4 levels (learned codebook).
    enable_fast_kernels(quant_model, enable=True)

    # ── Free dead fp32 originals so eval fits a 16GB GPU (e.g. Kaggle T4) ────
    # After calibration, quantized layers run entirely from weight_q (the
    # step-5 GEMM); linear.weight is read only by the fp16-skip fallback
    # (weight_q is None).  Dropping the original weight on every quantized
    # layer is a forward no-op and reclaims ~5GB for OPT-2.7b.  Skipped fp16
    # layers (weight_q is None) keep their weight for the fallback path.
    _freed_n, _freed_bytes = 0, 0
    for _m in quant_model.modules():
        if getattr(_m, "weight_q", None) is None:
            continue
        _lin = getattr(getattr(_m, "inner", _m), "linear", None)
        if _lin is not None and getattr(_lin, "weight", None) is not None:
            _freed_bytes += _lin.weight.numel() * _lin.weight.element_size()
            _lin.weight = None     # registered Parameter → None frees the tensor
            _freed_n += 1
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Freed {_freed_n} original weight tensors "
          f"({_freed_bytes / 1e9:.2f} GB) — quantized layers now run from "
          f"fp16 weight_q only")

    # ── Save a reloadable checkpoint (opt-in via SAVE_MODEL=1) ──────────────
    # Full-object save (see save_quantized_model): a plain state_dict() drops
    # the plain-attribute state (D, had_block_size, act_clip_ratio, gf4_levels)
    # so the reload would not run.  Path uses CKPT_DIR or the cwd — NEVER
    # __file__, which is undefined in a Colab/Jupyter cell (that NameError was
    # the old breakage).  Saved AFTER the weight-free so it's the compact fp16
    # model: quantized layers reload runnable from weight_q, skipped layers from
    # their kept weight.  Reload: load_quantized_model(path, device).
    if os.environ.get("SAVE_MODEL", "0") == "1":
        try:
            _ckpt_dir  = os.environ.get("CKPT_DIR", os.getcwd())
            _ckpt_path = os.path.join(_ckpt_dir, f"{_model_key}_fpquant.pt")
            print(f"\nSaving quantized model → {_ckpt_path}")
            save_quantized_model(quant_model, _ckpt_path)
            print(f"  saved ({os.path.getsize(_ckpt_path) / 1e9:.2f} GB) — reload "
                  f"with load_quantized_model('{_ckpt_path}', device)")
        except Exception as _e:
            print(f"  WARNING: checkpoint save failed ({_e}) — continuing to eval.")

    torch.cuda.empty_cache()
    # ── Activation-quant PPL sweep ─────────────────────────────────────────
    # Every eval (sliding AND GPTQ) is guarded independently so an OOM in one
    # stage/mode can never abort the rest of the sweep — critical so the
    # residual-GF4 headline still runs even if an earlier GPTQ pass OOMs.
    # GPTQ uses batch_size=1 on the quantized model: the padded weight_q + mu +
    # bias_correction buffers make it heavier than the fp16 baseline, and bs=4
    # OOMs the 2048-token chunks even on an 80GB H100.

    # Budget control for slow GPUs (e.g. Colab/Kaggle T4): by default run only
    # the headline modes (A16, gf4, adaptive, residual) and the sliding protocol.
    # Export FULL_SWEEP=1 on a fast/large GPU to also run nvfp4, learned levels,
    # H-smooth, and the (slow, bs=1) GPTQ protocol.
    _FULL_SWEEP = os.environ.get("FULL_SWEEP", "0") == "1"
    _RUN_GPTQ   = _FULL_SWEEP

    def _safe_eval(mode_str, label=None, gptq_batch_size=1):
        """Run sliding + GPTQ PPL for a given act-quant mode; returns (sliding, gptq).
        Each stage is guarded independently so an OOM on GPTQ doesn't lose the
        already-computed sliding result, and an OOM here never aborts the sweep."""
        label = label or str(mode_str)
        ppl_s, ppl_g = float('inf'), float('inf')
        # Sliding window eval
        try:
            torch.cuda.empty_cache()
            gc.collect()
            with act_quant_mode(quant_model, mode=mode_str):
                ppl_s = compute_standard_ppl(quant_model, tokenizer, dataset,
                                             _input_ids=_ppl_ids)
            print(f"FP4 PPL ({label}, sliding):", ppl_s)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during {label} sliding eval — skipping.")
            torch.cuda.empty_cache()
            gc.collect()
            return ppl_s, ppl_g
        # GPTQ-style eval (skipped unless FULL_SWEEP — slow at bs=1 on a T4)
        if not _RUN_GPTQ:
            return ppl_s, ppl_g
        try:
            gc.collect()
            with act_quant_mode(quant_model, mode=mode_str):
                ppl_g = compute_ppl_gptq_style(quant_model, tokenizer, dataset,
                                               seq_len=2048, batch_size=gptq_batch_size,
                                               _input_ids=_ppl_ids_gptq)
            print(f"FP4 GPTQ PPL ({label}):", ppl_g)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during {label} GPTQ eval — skipping.")
            torch.cuda.empty_cache()
            gc.collect()
        return ppl_s, ppl_g

    torch.cuda.empty_cache()
    ppl_fp4_a16, ppl_fp4_a16_gptq = _safe_eval(None,    label="A16")
    if _FULL_SWEEP:
        ppl_fp4_nv,  ppl_fp4_nv_gptq  = _safe_eval("nvfp4", label="A4 nvfp4")
    ppl_fp4_gf4, ppl_fp4_gf4_gptq = _safe_eval("gf4",   label="A4 gf4")

    # Per-block adaptive clip selection (online MSE minimization, no calibration)
    ppl_fp4_adaptive, ppl_fp4_adaptive_gptq = _safe_eval("gf4_adaptive")

    # Two-stage residual GF4 (2× effective resolution)
    ppl_fp4_residual, ppl_fp4_residual_gptq = _safe_eval("gf4_residual")

    # Learned GF4 codebook + H-SmoothQuant — FULL_SWEEP only.  The learned-levels
    # 400-step calibration and the h-smooth recalibration are both too slow for
    # a T4 budget and are not the headline result.
    if _FULL_SWEEP:
        # Learned GF4 codebook (gradient-optimized level positions)
        try:
            torch.cuda.empty_cache()
            gc.collect()
            calibrate_gf4_learned_levels(quant_model, calib_loader, device, blocksize,
                                          num_batches=4, n_steps=400)
            ppl_fp4_learned, ppl_fp4_learned_gptq = _safe_eval("gf4")
            # Reset learned levels for any subsequent evals
            for _, m in quant_model.named_modules():
                if type(m).__name__ == "HadamardQuantLinearFP":
                    m.gf4_levels = None
        except torch.cuda.OutOfMemoryError:
            print("  OOM during learned-levels calibration — skipping.")
            torch.cuda.empty_cache()
            gc.collect()

        # H-SmoothQuant: applied post-hoc to the already-calibrated model.
        # (Full production use: quantize_model_fp(..., gf4_variant="hsmooth"))
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print("\nApplying H-domain smooth scaling post-hoc...")
            # Save original W_had_q so we can measure the effect cleanly
            _saved_wq = {}
            for name, m in quant_model.named_modules():
                if type(m).__name__ == "HadamardQuantLinearFP" and m.weight_q is not None:
                    _saved_wq[name] = m.weight_q.data.clone()
            apply_gf4_hsmooth(quant_model, calib_loader, blocksize, device,
                              num_batches=4)
            ppl_fp4_hsmooth, ppl_fp4_hsmooth_gptq = _safe_eval("gf4")
            # Restore original weights
            for name, m in quant_model.named_modules():
                if name in _saved_wq:
                    if 'weight_q' in m.inner._buffers:
                        m.inner._buffers['weight_q'] = _saved_wq[name]
                    else:
                        m.inner.weight_q = _saved_wq[name]
                    m.h_smooth_scale = None
            del _saved_wq
        except torch.cuda.OutOfMemoryError:
            print("  OOM during H-SmoothQuant eval — skipping.")
            torch.cuda.empty_cache()
            gc.collect()

    # with act_quant_mode(quant_model, mode= "preshifted"):
    #     ppl_fp4  = compute_standard_ppl(quant_model, tokenizer, dataset)
    #     ppl_fp4_gptq = compute_ppl_gptq_style(quant_model, tokenizer, dataset,
    #                                         seq_len=2048)
    #     print("FP4  PPL (A4 preshift):", ppl_fp4)
    #     print("FP4 GPTQ PPL (A4 preshift):", ppl_fp4_gptq)
        # Test each layer in isolation — replace weight_q with original weight
# and measure PPL. If one layer causes most of the degradation,
# the weight reconstruction for that layer is broken.

        # for name, module in model.named_modules():
        #     if type(module).__name__ != "QuantLinearFP":
        #         continue
        #     saved = module.weight_q.clone()
        #     module.weight_q = module.linear.weight.data.clone()  # restore original
        #     ppl = compute_standard_ppl(model, tokenizer, dataset)
        #     print(f"Restoring {name}: PPL = {ppl:.2f}")
        #     module.weight_q = saved  # put quantized back
except torch.cuda.OutOfMemoryError:
    print("  OOM during quantization — saving FP32 baseline only.")
    print("  FP4 columns will be inf in the CSV.")
    torch.cuda.empty_cache()
    gc.collect()

import pandas as pd
from datetime import datetime
df = pd.DataFrame({
    "timestamp":                   [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "net":                         [model_name],
    "dataset":                     [data],
    "ppl_fp32_sliding":            [ppl_fp32],
    "ppl_fp32_gptq":               [ppl_fp32_gptq],
    "ppl_fp4_a16_sliding":         [ppl_fp4_a16],
    "ppl_fp4_a16_gptq":            [ppl_fp4_a16_gptq],
    "ppl_fp4_nvfp4_sliding":       [ppl_fp4_nv],
    "ppl_fp4_nvfp4_gptq":          [ppl_fp4_nv_gptq],
    "ppl_fp4_gf4_sliding":         [ppl_fp4_gf4],
    "ppl_fp4_gf4_gptq":            [ppl_fp4_gf4_gptq],
    "ppl_fp4_adaptive_sliding":    [ppl_fp4_adaptive],
    "ppl_fp4_adaptive_gptq":       [ppl_fp4_adaptive_gptq],
    "ppl_fp4_residual_sliding":    [ppl_fp4_residual],
    "ppl_fp4_residual_gptq":       [ppl_fp4_residual_gptq],
    "ppl_fp4_learned_sliding":     [ppl_fp4_learned],
    "ppl_fp4_learned_gptq":        [ppl_fp4_learned_gptq],
    "ppl_fp4_hsmooth_sliding":     [ppl_fp4_hsmooth],
    "ppl_fp4_hsmooth_gptq":        [ppl_fp4_hsmooth_gptq],
    "e_bits":                      [e_bits],
    "m_bits":                      [m_bits],
    "e_bits_scale":                [e_scale_bits],
    "m_bits_scale":                [m_scale_bits],
    "block_size":                  [blocksize],
    "batch_size":                  [batch_size],
    "mode":                        ["hadamard_gf4" if hadamard else "adap"],
})
if not os.path.exists(res_file):
    df.to_csv(res_file, index=False)
else:
    df.to_csv(res_file, mode='a', header=False, index=False)
print("\n=== FINAL RESULTS ===")
print(df.T.to_string())