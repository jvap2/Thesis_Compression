from .quant_layers_fp import(
    convert_to_fp_quant
)

# fp_flexround pulls in the CV codebase (Quantization_Experiments.brecq →
# densenet/resnet + the Jenks CUDA build). The LLM quantization path never
# calls these, so the import is optional — keeping it lets CV code work when
# present, while LLM-only deployments (e.g. Colab) need none of those files.
try:
    from .fp_flexround import (
        brecq_quantize_exp_fp,
        brecq_quantize_exp_fp_scale,
    )
except Exception as _e:  # missing CV deps → LLM path still fully functional
    brecq_quantize_exp_fp = None
    brecq_quantize_exp_fp_scale = None

from .bit_split import(
    quantize_model_fp,
    quantize_model_fp_llama,
    LLAMA_SKIP_PATTERNS,
    LLAMA_SKIP_PATTERNS_AGGRESSIVE,
    apply_smoothquant,
    act_quant_mode,
    quantize_activations,
    quantize_activations_gf4,
    quantize_activations_gf4_adaptive,
    quantize_activations_gf4_residual,
    optimize_gf4_levels,
    calibrate_gf4_learned_levels,
    calibrate_model_gf4_hsmooth,
    apply_gf4_hsmooth,
    QuantConv2dFP,
    QuantLinearFP,
    HadamardQuantLinearFP,
    preshifted_beta_only_mode,
    fwht_blockwise,
    calibrate_model_gf4,
    enable_fast_kernels,
    save_quantized_model,
    load_quantized_model,
)

from .block_smoothquant import(
    apply_block_smoothquant_opt,
)


__all__ = [
    'brecq_quantize_exp_fp',
    'brecq_quantize_exp_fp_scale',
    'quantize_model_fp',
    'quantize_model_fp_llama',
    'LLAMA_SKIP_PATTERNS',
    'LLAMA_SKIP_PATTERNS_AGGRESSIVE',
    'apply_smoothquant',
    'act_quant_mode',
    'quantize_activations',
    'quantize_activations_gf4',
    'quantize_activations_gf4_adaptive',
    'quantize_activations_gf4_residual',
    'optimize_gf4_levels',
    'calibrate_gf4_learned_levels',
    'calibrate_model_gf4_hsmooth',
    'apply_gf4_hsmooth',
    'QuantConv2dFP',
    'QuantLinearFP',
    'apply_block_smoothquant_opt',
    'preshifted_beta_only_mode',
    'HadamardQuantLinearFP',
    'fwht_blockwise',
    'calibrate_model_gf4',
    'enable_fast_kernels',
]