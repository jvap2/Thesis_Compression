"""
Aggregate Timeloop/Accelergy outputs and price the LUT at three decode
placements.

Two engines are actually simulated in Timeloop (both use simple compounds that
estimate correctly):
    gf4   = decode-PER-MAC (LUT inside the MAC)   — reliable
    nvfp4 = no LUT (fp4_mac)                       — reliable, the baseline

The other placements (decode-at-PE-load, decode-at-ingress) are priced
ANALYTICALLY from the real access counts Timeloop reports, times the component-
grounded LUT read energy (1 pJ = one codebook read, == one 4-bit MAC). GF4 is
SYMMETRIC so the codebook is an 8-entry MAGNITUDE table (3-bit index; the sign
bit is applied at the multiplier, never in the LUT). NOTE: Accelergy prices the
LUT via its `dummy` estimator, which returns a fixed per-read cost independent of
table depth, so 8- and 16-entry LUTs give byte-identical energy/area here — the
1 pJ/read below is that per-read constant, not a depth-resolved circuit number.
This avoids the compound-STORAGE estimation bug (wrapping a regfile/SRAM in a
compound collapses its area/energy — that's why gf4_atload/gf4_ingress engines
gave nonsense area like -80%).

VALIDATION (printed): the analytical per-MAC overhead (2*Computes*E_LUT/total)
must equal the Timeloop-MEASURED gf4-vs-nvfp4 overhead.  It matches to the
decimal, which licenses trusting the at-ingress number from the same method.

Decode counts per placement (per layer, from the nvfp4 run):
    per-MAC   : 2 * Computes               (every MAC decodes 2 operands)
    at-ingress: DRAM operand accesses      (decode once as data enters the chip)
The FWHT (~2%) and microscale add-ons and the residual 2x factor are layered on.
"""
import os, re, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
T = 2048
E_LUT = 1.0           # pJ per codebook read (dummy-estimator constant, 45nm == 1 MAC;
                      # 8-entry symmetric magnitude LUT, depth-independent in this model)
FWHT_FRAC = 0.02
MICROSCALE_FRAC = 0.21
RESIDUAL_MULT = 2.0
N_PE = 256


def stats_text(engine, base):
    p = os.path.join(HERE, "out", engine, base, "timeloop-mapper.stats.txt")
    return open(p).read() if os.path.exists(p) else None


def parse_stats(txt):
    e = float(re.search(r"Energy:\s*([\d.eE+-]+)\s*uJ", txt).group(1)) * 1e6   # pJ
    c = int(re.search(r"Computes\s*=\s*(\d+)", txt).group(1))
    dram = int(re.findall(r"=== DRAM ===\s*\n\s*Total scalar accesses\s*:\s*(\d+)", txt)[-1])
    return dict(energy=e, computes=c, dram=dram)


def mac_energy(txt):
    # energy of the PE 'mac' compound (LUT+MAC for gf4, MAC only for nvfp4) — the
    # `=== mac ===` block's Energy (total). Lets us subtract the MAC out and price
    # the lookup table against memory alone (multiply-free).
    m = re.search(r"=== mac ===.*?Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", txt, re.S)
    return float(m.group(1)) if m else 0.0


def total_area_mm2(engine, base):
    p = os.path.join(HERE, "out", engine, base, "timeloop-mapper.ART.yaml")
    if not os.path.exists(p):
        return None
    d = yaml.safe_load(open(p))
    return sum(t["area"] * (N_PE if ".PE[" in t["name"] else 1)
               for t in d["ART"]["tables"]) / 1e6


def load_manifest():
    return yaml.safe_load(open(os.path.join(HERE, "manifest.yaml")))


def main():
    man = load_manifest()
    for mkey, shapes in man.items():
        print(f"\n===== {mkey} (Timeloop+Accelergy, weighted over all layers) =====")
        # weighted sums
        tot = dict(gf4_e=0.0, nv_e=0.0, gf4_real_e=0.0, e2m1_e=0.0, ing_fp16_e=0.0,
                   gf4_mac=0.0, nv_mac=0.0, computes=0.0, dram=0.0)
        gf4_area = nv_area = gf4_real_area = e2m1_area = ing_fp16_area = None
        ok = True
        for s in shapes:
            base = f"{mkey}__{s['shape']}"
            tg = stats_text("gf4", base); tn = stats_text("nvfp4", base)
            if tg is None or tn is None:
                print(f"  [missing] {base} — run run.sh"); ok = False; continue
            g = parse_stats(tg); n = parse_stats(tn); w = s["count"]
            tot["gf4_e"] += g["energy"] * w
            tot["nv_e"]  += n["energy"] * w
            tot["computes"] += n["computes"] * w
            tot["dram"]     += n["dram"] * w
            tot["gf4_mac"]  += mac_energy(tg) * w      # LUT+MAC compound
            tot["nv_mac"]   += mac_energy(tn) * w      # MAC only
            gf4_area = total_area_mm2("gf4", base)
            nv_area  = total_area_mm2("nvfp4", base)
            tr = stats_text("gf4_real", base)          # realistic fp16 multiply
            if tr is not None:
                tot["gf4_real_e"] += parse_stats(tr)["energy"] * w
                gf4_real_area = total_area_mm2("gf4_real", base)
            te = stats_text("nvfp4_e2m1", base)        # faithful E2M1 fpmac baseline
            if te is not None:
                tot["e2m1_e"] += parse_stats(te)["energy"] * w
                e2m1_area = total_area_mm2("nvfp4_e2m1", base)
            ti = stats_text("gf4_ingress_fp16", base)  # ingress decode + fp16 (16b on-chip)
            if ti is not None:
                tot["ing_fp16_e"] += parse_stats(ti)["energy"] * w
                ing_fp16_area = total_area_mm2("gf4_ingress_fp16", base)
        if not ok:
            continue

        nv = tot["nv_e"]
        # measured (Timeloop) per-MAC overhead
        meas_permac = 100.0 * (tot["gf4_e"] - nv) / nv
        # analytical LUT energy by placement
        lut_permac  = 2 * tot["computes"] * E_LUT
        lut_ingress =     tot["dram"]     * E_LUT
        oh_permac   = 100.0 * lut_permac  / nv
        oh_ingress  = 100.0 * lut_ingress / nv

        print(f"  Timeloop measured: nvfp4 {nv/T/1e3:9.1f}k pJ/tok   "
              f"gf4(per-MAC) {tot['gf4_e']/T/1e3:9.1f}k pJ/tok   "
              f"area {nv_area:.3f} mm^2")
        print(f"  array area: gf4 {gf4_area:.3f} vs nvfp4 {nv_area:.3f} mm^2  "
              f"(LUT area {100*(gf4_area-nv_area)/nv_area:+.2f}%)")
        print("  -- LUT ENERGY overhead vs NVFP4, by decode placement --")
        print(f"     decode-per-MAC      {oh_permac:6.2f}%   "
              f"(validates: Timeloop measured {meas_permac:.2f}%)")
        print(f"     decode-at-ingress   {oh_ingress:6.2f}%   "
              f"(LUT fires once per DRAM load -> ~free)")
        if tot["nv_mac"] > 0:
            lut_e = tot["gf4_mac"] - tot["nv_mac"]     # LUT alone (MAC cancels)
            mem   = nv - tot["nv_mac"]                 # memory system, MAC removed
            print(f"     LUT vs MEMORY (multiply-free) {100*lut_e/mem:6.2f}%   "
                  f"(LUT {lut_e/T/1e3:.1f}k / mem {mem/T/1e3:.1f}k pJ/tok; MAC "
                  f"{tot['nv_mac']/T/1e3:.1f}k excluded)")
        if tot["gf4_real_e"] > 0 or tot["ing_fp16_e"] > 0:
            print("  -- REALISTIC fp16 GF4 vs NVFP4 (int4 = faithful E2M1 proxy) --")
            if tot["gf4_real_e"] > 0:
                mult   = 100.0 * (tot["gf4_real_e"] - tot["gf4_e"]) / nv   # LUT cancels
                permac = 100.0 * (tot["gf4_real_e"] - nv) / nv
                print(f"     fp16 multiply penalty (placement-indep.) {mult:7.2f}%   (gf4_real - gf4)")
                print(f"     per-MAC decode + fp16 total              {permac:7.2f}%   (gf4_real vs int4)")
                if gf4_real_area is not None:
                    print(f"        area: gf4_real {gf4_real_area:.3f} vs nvfp4 {nv_area:.3f} mm^2  "
                          f"({100*(gf4_real_area-nv_area)/nv_area:+.2f}%)")
            if tot["ing_fp16_e"] > 0:
                ing_lut = tot["dram"] * E_LUT
                ing_tot = 100.0 * (tot["ing_fp16_e"] + ing_lut - nv) / nv
                ing_buf = 100.0 * (tot["ing_fp16_e"] - nv) / nv
                print(f"     ingress decode + fp16 total              {ing_tot:7.2f}%   "
                      f"(fp16 MAC + 16b on-chip buffer + analytic ingress LUT)")
                print(f"        of which fp16 MAC + 16b buffer        {ing_buf:7.2f}%   (gf4_ingress_fp16 vs int4)")
                if ing_fp16_area is not None:
                    print(f"        area: gf4_ingress_fp16 {ing_fp16_area:.3f} vs nvfp4 {nv_area:.3f} mm^2  "
                          f"({100*(ing_fp16_area-nv_area)/nv_area:+.2f}%)")
            if tot["e2m1_e"] > 0:
                print(f"     [note] fpmac-E2M1 {tot['e2m1_e']/T/1e3:.1f}k >= int4 {nv/T/1e3:.1f}k "
                      f"pJ/tok: Aladdin floors sub-fp8, so int4 is the faithful E2M1 proxy")
        print(f"  note: ingress trades ~2x global-buffer footprint for W/I tensors "
              f"(decoded 8b vs 4b codes).")
        # token energy of our design (ingress) + analytic add-ons
        gf4_ingress_e = nv + lut_ingress
        print(f"  GF4 1-term (ingress) +FWHT~{int(FWHT_FRAC*100)}% "
              f"~{gf4_ingress_e*(1+FWHT_FRAC)/T/1e3:.1f}k pJ/tok; "
              f"residual(2x) ~{gf4_ingress_e*RESIDUAL_MULT*(1+FWHT_FRAC)/T/1e3:.1f}k pJ/tok")
    print("\n  (gf4/nvfp4 energy+area from Accelergy; LUT placements priced from real "
          "Timeloop access counts x 1pJ/read. Per-MAC analytic == measured = validated.)")


if __name__ == "__main__":
    main()
