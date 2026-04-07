import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from FP_Quantization_Experiments import brecq_quantize_exp_fp, brecq_quantize_exp_fp_scale, quantize_model_fp, QuantConv2dFP, QuantLinearFP
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import math
bitwidth = 4
e_bits = 2
m_bits = 1
e_scale_bits = 4
m_scale_bits = 3
blocksize = 32
batch_size = 16

def compute_perplexity(model, tokenizer, dataset, device="cuda"):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in dataset["text"]:
        if len(text.strip()) == 0:
            continue

        enc = tokenizer(text, return_tensors="pt").to(device)
        input_ids = enc.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

    ppl = math.exp(total_loss / total_tokens)
    return ppl



def collate_fn(batch, tokenizer, seq_len=512, device="cuda"):
    texts = [item["text"] for item in batch if len(item["text"].strip()) > 0]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len
    )

    input_ids = enc.input_ids.to(device)

    # 👇 THIS FIXES YOUR ERROR
    return input_ids, input_ids   # (x, y) format

def get_llm_dataloader(dataset, tokenizer, batch_size=8):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
res_file = "quant_res.csv"
model_name = "facebook/opt-125m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()
data ="wikitext-2-raw-v1"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

ppl_fp32 = compute_perplexity(model, tokenizer, dataset)
calib_loader = get_llm_dataloader(dataset, tokenizer, batch_size=batch_size)
quant_model = quantize_model_fp(model,calib_loader, block_size=blocksize,e_bits=e_bits,m_bits=m_bits,e_bits_scale=e_scale_bits,m_bits_scale=m_scale_bits, device = device, use_HG=False, use_Hessian=False, use_adap=True)


ppl_fp4  = compute_perplexity(quant_model, tokenizer, dataset)

print("FP32 PPL:", ppl_fp32)
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
    "block_size": [blocksize]
})
if not os.path.exists(res_file):
    df.to_csv(res_file, index=False)
else:
    df.to_csv(res_file, mode='a', header=False, index=False)