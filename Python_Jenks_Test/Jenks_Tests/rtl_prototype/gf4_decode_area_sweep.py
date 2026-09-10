"""
gf4_decode_area_sweep.py -- decode-ROM area vs. codebook regularity.

Companion to CUDA_FP4_Test/gf4_regularity_sweep.py (which measures PPL). For
each codebook variant we template gf4_decode_fixed.v with that codebook's Q1.7
integer constants (round(level*128)), synthesize with Yosys against Nangate45,
and read the top-module cell area. Regularizing the levels (fewer fractional
bits -> more trailing zeros in the constants) shrinks the mux/ROM. Joining this
area with the PPL sweep gives the accuracy-vs-decode-area Pareto.
"""
import os, re, subprocess, math

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = "/home/jvap2/eda/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT = os.path.join(HERE, "synth_out", "regularity")
os.makedirs(OUT, exist_ok=True)

GF4_LEVEL = [0.0, 0.0796082, 0.1737177, 0.2828685,
             0.3952704, 0.5250730, 0.6961928, 1.0]


def snap_q1b(levels, B):
    g = [round(v * (2 ** B)) / (2 ** B) for v in levels]
    g[0], g[-1] = 0.0, 1.0
    return g


def snap_pow2(levels):
    out = [(2.0 ** round(math.log2(v)) if v > 0 else 0.0) for v in levels]
    out[-1] = 1.0
    return out


def codebooks():
    b = {"exact": list(GF4_LEVEL)}
    for B in (5, 4, 3):
        b[f"q1b{B}"] = snap_q1b(GF4_LEVEL, B)
    b["pow2"] = snap_pow2(GF4_LEVEL)
    # exact constrained optima from constrained_codebook.py (1/16 grid)
    b["mseopt_q4"] = [k / 16 for k in (0, 2, 4, 6, 8, 10, 13, 16)]
    b["mseopt_pop2"] = [k / 16 for k in (0, 2, 4, 6, 8, 10, 12, 16)]
    return b


VMOD = """// auto-generated decode ROM for codebook '{name}'
module gf4_decode_fixed (
    input  wire [2:0] idx,
    input  wire       sign_in,
    output wire       sign_out,
    output reg  [7:0] mag_q1_7
);
    assign sign_out = sign_in;
    always @(*) begin
        case (idx)
{cases}
            default: mag_q1_7 = 8'd0;
        endcase
    end
endmodule
"""


def gen_verilog(name, levels):
    q17 = [int(round(v * 128)) for v in sorted(levels)]
    cases = "\n".join(f"            3'd{i}: mag_q1_7 = 8'd{q};" for i, q in enumerate(q17))
    path = os.path.join(OUT, f"decode_{name}.v")
    with open(path, "w") as f:
        f.write(VMOD.format(name=name, cases=cases))
    return path, q17


def synth(name, vpath):
    area_txt = os.path.join(OUT, f"area_{name}.txt")
    ys = os.path.join(OUT, f"{name}.ys")
    with open(ys, "w") as f:
        f.write(f'read_verilog -sv "{vpath}"\n')
        f.write("hierarchy -top gf4_decode_fixed\n")
        f.write("proc; opt; fsm; opt; memory; opt\ntechmap; opt\n")
        f.write(f'dfflibmap -liberty "{LIB}"\nabc -liberty "{LIB}"\nclean\n')
        f.write(f'tee -o {area_txt} stat -liberty "{LIB}"\n')
    subprocess.run(["yosys", "-q", ys], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    txt = open(area_txt).read()
    m = re.search(r"Chip area for (?:top )?module '\\gf4_decode_fixed':\s*([\d.]+)", txt)
    ncells = re.search(r"Number of cells:\s*(\d+)", txt)
    return float(m.group(1)) if m else float("nan"), int(ncells.group(1)) if ncells else -1


def main():
    print(f"{'codebook':8s} {'distinct':>8s} {'cells':>6s} {'area(um2)':>10s} {'vs exact':>9s}")
    rows = []
    area_exact = None
    for name, levels in codebooks().items():
        vpath, q17 = gen_verilog(name, levels)
        area, cells = synth(name, vpath)
        if name == "exact":
            area_exact = area
        distinct = len(set(q17))
        save = 100.0 * (area_exact - area) / area_exact if area_exact else 0.0
        rows.append((name, distinct, cells, area, save, q17))
        print(f"{name:8s} {distinct:8d} {cells:6d} {area:10.3f} {save:+8.2f}%   q1.7={q17}")
    csv = os.path.join(OUT, "decode_area.csv")
    with open(csv, "w") as f:
        f.write("codebook,distinct_levels,cells,area_um2,area_savings_pct\n")
        for name, distinct, cells, area, save, _ in rows:
            f.write(f"{name},{distinct},{cells},{area:.3f},{save:.2f}\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
