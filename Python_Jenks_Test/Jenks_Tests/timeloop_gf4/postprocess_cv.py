"""Aggregate the dense CV GF4-Engine sweep into a per-model energy/area table.

Reads out/cv_gf4/<layer> (GF4 LUT-FP4, 4-bit) and out/cv_nvfp4/<layer> (plain 4-bit,
no LUT — isolates the GF4 decode overhead). Both are exact Timeloop+Accelergy 45nm
runs. The 4-bit-vs-fp16 win is reported as OPERAND-TRAFFIC (bits moved) — exact and
model-independent — rather than a fabricated fp16 arch: GF4 moves weights+acts at 4-bit
vs 16-bit = 4x less operand traffic, on top of the LUT overhead measured here.
Writes cv_energy_results.csv."""
import os, re, csv, yaml
HERE=os.path.dirname(os.path.abspath(__file__)); N_PE=256

def stats(engine, base):
    p=os.path.join(HERE,"out",engine,base,"timeloop-mapper.stats.txt")
    if not os.path.exists(p): return None
    t=open(p).read()
    e=float(re.search(r"Energy:\s*([\d.eE+-]+)\s*uJ",t).group(1))*1e6   # pJ
    c=int(re.search(r"Computes\s*=\s*(\d+)",t).group(1))
    cy=int(re.search(r"Cycles:\s*(\d+)",t).group(1))
    return dict(energy=e, computes=c, cycles=cy)

def area_mm2(engine, base):
    p=os.path.join(HERE,"out",engine,base,"timeloop-mapper.ART.yaml")
    if not os.path.exists(p): return None
    d=yaml.safe_load(open(p))
    return sum(t["area"]*(N_PE if ".PE[" in t["name"] else 1) for t in d["ART"]["tables"])/1e6

man=yaml.safe_load(open(os.path.join(HERE,"manifest.yaml")))
rows=[]
for mkey in sorted(k for k in man if k.startswith("cv_")):
    g_e=n_e=macs=cyc=0.0; area=None; ok=True
    for s in man[mkey]:
        base=f"{mkey}__{s['shape']}"; w=s["count"]
        g=stats("cv_gf4",base); n=stats("cv_nvfp4",base)
        if g is None or n is None: print(f"  [missing] {base}"); ok=False; continue
        g_e+=g["energy"]*w; n_e+=n["energy"]*w; cyc+=g["cycles"]*w
        macs+=g["computes"]*w
        if area is None: area=area_mm2("cv_gf4",base)
    if not ok: continue
    lut_ovh=100*(g_e-n_e)/n_e if n_e else 0
    rows.append(dict(model=mkey.replace("cv_",""), macs_M=round(macs/1e6,2),
                     gf4_energy_uJ=round(g_e/1e6,2), gf4_pJ_per_MAC=round(g_e/macs,3),
                     nvfp4_energy_uJ=round(n_e/1e6,2), lut_overhead_pct=round(lut_ovh,1),
                     area_mm2=round(area,4) if area else None,
                     cycles_M=round(cyc/1e6,2)))
    print(f"{rows[-1]['model']:12s}  {rows[-1]['macs_M']:8.2f}M MAC  GF4 {rows[-1]['gf4_energy_uJ']:8.2f}uJ "
          f"({rows[-1]['gf4_pJ_per_MAC']:.2f} pJ/MAC)  LUT-ovh {rows[-1]['lut_overhead_pct']:+.1f}%  area {rows[-1]['area_mm2']} mm2")
OUT=os.path.join(HERE,"cv_energy_results.csv")
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"\nsaved {OUT}")
print("NOTE: 4-bit-vs-fp16 = 4x less operand traffic (weights+acts 4b vs 16b) + 8x smaller weights,")
print("      on top of the GF4 LUT overhead measured above (decode-per-MAC placement).")
