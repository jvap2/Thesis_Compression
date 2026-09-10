# Shift-add MAC (third embodiment) — verification & area/power results

Tooling: Icarus Verilog 11.0 (functional), Yosys 0.9 + Nangate45
(`NangateOpenCellLibrary_typical.lib`, 45 nm, typical corner) for area.
Measured 2026-09-10 on the project workstation.

## 1. Functional verification (bit-exact, exhaustive)

`bash scripts/run_sim.sh`:
- `shift_add_mult`: **256/256** (w_code, a_code) pairs bit-exact vs an *independent*
  Python golden model (which decodes both codes to plain floats and multiplies —
  it does NOT re-use the power-of-two decomposition, so agreement genuinely checks
  Eq. `eq:shift-add-mac`).
- `shift_add_mac`: K=64 accumulated dot product exact (`acc_q = -422 = expected`).
- The product is **exact, not approximate**: F=4 makes every lane a pure left shift,
  so the shift-add datapath introduces **zero** additional quantization error over a
  real multiplier. (Accuracy cost of the third embodiment = 0.)

RTL matches the description:
- `w_decode_e2m1`: `1.5→{2⁰,2⁻¹}`, `3→{2¹,2⁰}`, `6→{2²,2¹}`; `{0.5,1,2,4}` single-term.
- `a_decode_hwcb`: codebook `{0,2,4,6,8,10,12,16}/16`, every entry popcount ≤ 2
  (`6→{2⁻²,2⁻³}`, `10→{2⁻¹,2⁻³}`, `12→{2⁻¹,2⁻²}`).
- `shift_add_mult`: 4 lanes `(i,j)`, `exp = e_i+f_j`, `sign = s_w⊕s_a`, all 4 computed
  unconditionally and zeroed via valid bits, summed in a compressor. No `*` operator,
  no branch on term count → data-independent latency/structure.

## 0. HEADLINE (rigorous gate-level; the baseline choice is everything)

Rigorous gate-level flow: real Nangate cell models, full-node VCD (2000 random ops),
OpenSTA @1 GHz. All numbers are the *multiply mechanism* (the accumulator is common to
every embodiment). The conclusion depends entirely on **what embodiment 1 is**:

| MAC multiply mechanism | Area (µm²) | Energy (pJ/MAC) | Exact? |
|---|---:|---:|---|
| **fp16 multiplier** — "decode + fp16 multiply" (emb1 as the paper frames it) | **796.7** | **1.118** | fp16-rounded |
| **shift-add (emb3)** | **~595** | **0.254** | **exact** |
| ↳ shift-add vs fp16 | **1.3× smaller** | **4.4× lower** | exact vs rounded |
| fixed int decode+multiply (codebook-hardwired) | 145.0 | 0.185 | exact |
| fixed product ROM (codebook-hardwired) | 287.8 | 0.155 | exact |

**The defensible claim: against the fp16 multiplier it replaces, the shift-add is 1.3×
smaller, 4.4× lower power, and exact (vs fp16 rounding).** That matches the paper's
framing (GF4 datapath = decode + fp16 multiply) and is a real power win.

**The honest caveat:** if instead you *hardwire the fixed codebook* into a minimal integer
multiplier or a product ROM (145–288 µm²), those are smaller than the shift-add — because
for a 4-bit×4-bit operand space a fully-minimized block always beats a *structured*
datapath (the 4-lane shift-add has structure the minimizer can't collapse). So do NOT
claim the shift-add beats a codebook-optimized fixed multiplier; claim it beats the **fp16
multiplier**, and add that it needs no multiply primitive and is a two-operand APoT
generalization.

## 0b. LUT×LUT scaling vs codebook size (fixed product ROM, Nangate45)

| $|\mathcal{C}|$ codes/op | 8 | 16 (paper) | 32 | 64 |
|---|---:|---:|---:|---:|
| entries | 64 | 256 | 1024 | 4096 |
| LUT×LUT area (µm²) | 99 | 338 | 1242 | 4056 |
| shift-add area (µm²) | 595 | 595 | 595 | 595 |

LUT×LUT grows as ~$O(|\mathcal{C}|^2)$; shift-add compute is fixed (four lanes, set by the
popcount≤2 bound). **Crossover ≈ |C|≈24**: at the paper's |C|=16 the lookup is SMALLER than
the shift-add (338 vs 595), and the shift-add only wins the table for |C|≥32. So the
shift-add's honest advantages are: (1) beats the **fp16 multiplier** (1.3×/4.4×, exact);
(2) **fixed cost** — does not grow with codebook size like the table; (3) reprogrammable
codebook with no stored table. Do NOT claim it beats the LUT×LUT lookup at |C|=16 — it does
not.

Notes:
- `PROD_W=8` is bit-exact (verified 0/256 mismatches vs PROD_W=16) but does not shrink the
  block — yosys already dead-code-eliminates the unused upper bits at PROD_W=16.
- The programmable-PE area numbers in §2 (2179 / 9376 µm²) and the I/O-seeded power in §3
  (4.29 / 1.36 / 14.95 pJ) are **superseded** by this section: §2 compared fixed-vs-
  programmable, and §3's power was ~14× inflated by activity propagation (no cell models
  at the time). Use the table above.

## 2. Area — programmable-PE basis (NOT apples-to-apples; see §0)

| Embodiment | Full multiply block | Multiply mechanism only |
|---|---:|---:|
| 1 — decode + 16-bit multiply (`gf4_pe_decode_then_multiply`) | **2179.87** | 980.74 (local, excl. 2 decoders) |
| 2 — LUT×LUT joint table (`gf4_pe_lutxlut`) | **9376.23** | 8876.69 (joint table) |
| **3 — shift-add MAC (`shift_add_mult`, this RTL)** | **617.65** | 594.78 (lanes + compressor) |

Breakdown of embodiment 3: `w_decode` 8.25 + `a_decode` 14.63 + combine/lanes/compressor
594.78 = **617.65 µm²**, 557 mapped cells.

**Ratios (full block):**
- shift-add is **3.5× smaller** than decode+16-bit-multiply (2179.87 / 617.65 = 3.53).
- shift-add is **15.2× smaller** than LUT×LUT (9376.23 / 617.65 = 15.18).

**Ratios (multiply mechanism only — isolates the datapath claim, same decoder scope removed):**
- vs the 16-bit multiplier + wrapper: **1.65× smaller** (980.74 / 594.78).
- vs the LUT×LUT joint table: **14.9× smaller** (8876.69 / 594.78).

Caveat: embodiments 1–2 use a *programmable* decoder (599.56 µm² each) for a
reconfigurable codebook; embodiment 3 hard-wires the codebook (fixed decoders, 8–15 µm²).
Part of the 3.5× full-block win vs embodiment 1 is that fixed decode; the multiply-mechanism
column removes that to isolate the shift-add-vs-multiplier claim (still 1.65×).

This confirms the paragraph: the third embodiment removes the 16-bit multiplier of the
first **and** the LUT×LUT joint-product table of the second (measured ~15× the area of
the shift-add block), at the cost of the exponent-addition + term-routing logic (the
594.78 µm² combine block).

## 3. Power (OpenSTA 3.1.0, Nangate45, 1 GHz, per-MAC = power/f)

Built OpenSTA 3.1.0 from source (`../OpenSTA/build/sta`, CUDD in `../cudd-install`).
Each MAC lane synthesized to a gate netlist, driven by a VCD of 2000 random-code MACs
(`power/tb_*.v`), activity read into OpenSTA, `report_power` at a 1 GHz clock.

| Embodiment | Energy / MAC | Total @1GHz | vs shift-add |
|---|---:|---:|---|
| 1 — decode + 16-bit multiply | **1.36 pJ** | 1.361 mW | 0.32× (lower) |
| 2 — LUT×LUT joint table      | **14.95 pJ**| 14.954 mW | 3.49× (higher) |
| **3 — shift-add MAC**        | **4.29 pJ** | 4.286 mW  | — (2.67 pJ combinational + 1.62 pJ acc) |

**Read this carefully — the power story is not the area story:**
- Shift-add is **3.5× lower energy than the LUT×LUT** (its area win carries over here).
- Shift-add is **higher** energy than the compact decode+multiply, i.e. removing the
  multiplier did NOT reduce dynamic power. Cause: the datapath computes **all 4
  barrel-shift lanes unconditionally every cycle** (the no-divergence property), and
  barrel shifters are high-toggle; the multiply-free construction trades area for
  switching, it does not free-lunch power.
- **Caveat on rigor:** no Nangate Verilog cell models are installed, so a true
  gate-level VCD sim is not possible. OpenSTA was seeded from the I/O-level VCD
  (44–68 pins annotated) and *propagated* internal switching. The decode+multiply
  number in particular looks under-annotated (switching power 76 µW vs 1.6 mW for
  shift-add), so its 1.36 pJ is probably an underestimate. Treat the power column as
  a first gate-level estimate, not a signed-off number; a rigorous comparison needs
  cell-model gate-level simulation.

**Bottom line:** the third embodiment's clear, rigorous win is **area** (3.5×/15×) and
**exactness**; on **power** it decisively beats the LUT×LUT but is *not* a guaranteed win
over a compact multiplier — the unconditional 4-lane design is the reason, and a
data-gated variant (at the cost of divergence) would be the lever to reclaim it.

## Reproduce

```bash
bash scripts/run_sim.sh                              # functional (needs iverilog)
# area (needs yosys + Nangate45 liberty at ~/eda/nangate45/lib):
bash ../02_run_synthesis.sh shift_add_mult \
     rtl/w_decode_e2m1.v rtl/a_decode_hwcb.v rtl/shift_add_mult.v
```
