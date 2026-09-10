"""
multipass_vs_datapath.py — the efficiency crossover between the two ways of
handling the outlier-retention layers (fc2 / down_proj / lm_head):

  (A) MULTI-PASS      : run the outliers as N residual FP4 passes on the SAME
                        shared GF4 array.  No extra silicon; cost = N x the
                        outlier MACs (+ per-pass LUT/scale add-ons).
  (B) DEDICATED FP16  : add a small FP16 sub-array and route the outliers to it
                        at 1 pass.  Extra silicon; full-precision in one shot.

hw_sim_gf4.py hardcodes N=4 for (A), which is *exactly* the per-MAC energy
break-even (e_mac_fp16 = 4 x e_mac_fp4).  This sweeps N so the trade is explicit:
  - AREA:   (A) is flat at 1.00x (no added hardware); (B) is fixed and larger.
            (A) wins for every N in a sane range.
  - ENERGY: (A) grows ~linearly in N; ties (B) near N=4, is cheaper below, more
            expensive above (plus a small per-pass LUT/scale/Hadamard tax, so the
            true tie is a hair under 4).
  - The accuracy-parity N (smallest N whose PPL ~ FP16-retain) is what decides
    which side of the crossover we actually live on.  validate_multipass.py pins
    it empirically; the recon SNR ladder is the local proxy.

Reuses hw_sim_gf4's cost model verbatim — only outlier_passes varies.
"""
import sys, copy
import hw_sim_gf4 as H


def sweep(model_key, hw, T=2048, n_max=8):
    # Reference dedicated-FP16-unit engine (outliers at fp16, 1 pass, +0.25 sub-array)
    fp16u = H.Engine("GF4 +FP16unit", "fp4", "fp16", 1, 1, lut=True,
                     fp16_unit_frac=0.25, ppl_key="residual-GF4 (ours,2t)")
    c_b = H.engine_cost(model_key, fp16u, hw, T)
    a_b = H.engine_area(fp16u)

    # FP16-only reference for normalizing energy/tok (matches hw_sim_gf4)
    f16 = next(e for e in H.ENGINES if e.bulk_mode == "fp16")
    e_ref = H.engine_cost(model_key, f16, hw, T)["pj_per_token"]

    print(f"\n{'='*78}\n  {model_key}: outlier treatment sweep (T={T}, {H.PE_N}-PE 45nm)\n{'='*78}")
    print(f"  dedicated-FP16-unit ref : area {a_b/H.engine_area(H.ENGINES[0]):.2f}x  "
          f"energy/tok {c_b['pj_per_token']/e_ref:.3f}x FP16\n")
    print(f"  {'N passes':>8} | {'area':>6} | {'energy/tok':>11} | {'vs FP16unit':>11} | "
          f"{'speed/area':>10} | verdict")
    a_min = H.engine_area(H.ENGINES[0])  # fixed-E2M1 smallest
    dens_ref = H.engine_cost(model_key, f16, hw, T)["tokens_per_s"] / H.engine_area(f16)
    for n in range(1, n_max + 1):
        mp = H.Engine(f"GF4 multipass N={n}", "fp4", "fp4", 1, n, lut=True,
                      ppl_key="residual-GF4 (ours,2t)")
        c_a = H.engine_cost(model_key, mp, hw, T)
        a_a = H.engine_area(mp)
        e_ratio = c_a['pj_per_token'] / c_b['pj_per_token']
        dens = c_a["tokens_per_s"] / a_a
        v = "multipass CHEAPER" if e_ratio < 1.0 else ("~tie" if e_ratio < 1.03 else "FP16unit cheaper")
        print(f"  {n:>8} | {a_a/a_min:5.2f}x | {c_a['pj_per_token']/e_ref:9.3f}x | "
              f"{e_ratio:9.3f}x | {dens/dens_ref:8.2f}x | {v}")
    print(f"\n  AREA: multipass is 1.00x for all N (no added hardware) vs "
          f"{a_b/a_min:.2f}x for the FP16 unit -> multipass wins area at every N.")
    print(f"  ENERGY: crossover near N=4 (=e_mac_fp16/e_mac_fp4="
          f"{hw.e_mac_fp16/hw.e_mac_fp4:.0f}); below it multipass is cheaper, above "
          f"it the dedicated unit is.\n  DECIDER: the accuracy-parity N "
          f"(validate_multipass.py 'ppl').")


if __name__ == "__main__":
    hw = H.HWConfig()
    H.load_colab_exports()
    keys = sys.argv[1:] or ["opt-125m", "opt-1.3b", "opt-2.7b"]
    for k in keys:
        if k in H.MODEL_LAYERS:
            sweep(k, hw)
        else:
            print(f"  [skip] {k} not in MODEL_LAYERS ({list(H.MODEL_LAYERS)[:6]}...)")
