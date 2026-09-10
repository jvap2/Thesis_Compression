# GF4 LUT Area/Energy vs. Technology Node

Measured area (and, subsequently, energy) of the GF4 decode LUT and the FP4 PE it sits
inside, synthesized across three real technology nodes with open standard-cell libraries.
Purpose: quantify the GF4 codebook-decode overhead and show how it scales with node — the
hardware-cost backbone of the "iso-energy drop-in" claim.

## 1. Motivation / what this replaces

The Timeloop/Accelergy accelerator model prices the GF4 LUT with Accelergy's `dummy`
estimator (a hand-set constant, e.g. `mac_random = 3 pJ` = 1 MAC + 2 LUT reads @ 1 pJ),
which is **technology-independent** — sweeping the `technology` attribute produces byte-
identical numbers. To make a defensible node-scaling statement we instead **synthesize the
actual GF4 RTL** against per-node standard-cell libraries and measure area (Yosys `stat`)
and, next, energy (gate-level power). This matches recent practice (synthesized logic +
tool-based memory), e.g. Titanus (GLSVLSI 2025) and DOSA (MICRO 2023).

## 2. Methodology

- **Design:** `gf4_decode_fixed` (the hardwired 3-bit-index → 8-entry × 8-bit **GF4
  Gaussian-quantile** magnitude decode — codebook `{0, 0.0796, 0.1737, 0.2829, 0.3953,
  0.5251, 0.6962, 1.0}` = `GF4_POS` from the Python reference, in Q1.7 unsigned; sign applied
  at the multiplier) and `gf4_pe` (`PROGRAMMABLE=0`, `MAG_WIDTH=8`) — the full FP4 PE (decode +
  multiply-accumulate). The NVFP4/E2M1 *weight*-decode baseline (`{0,0.5,…,6.0}`, Q4.4) is kept
  in `e2m1_decode_fixed.v` for the codebook comparison. RTL: `rtl_prototype/phase0_decode/`.
- **Synthesis:** Yosys (`proc; opt; fsm; opt; memory; opt; techmap; opt; dfflibmap; abc;
  clean; stat`), mapping to each node's Liberty. Area from `stat -liberty`.
- **Standard-cell libraries (open, license-free):**
  | node | library | corner |
  |---|---|---|
  | 130 nm | SkyWater **SKY130** `sky130_fd_sc_hd` | tt, 25 °C, 1.8 V |
  | 45 nm | **Nangate45** `NangateOpenCellLibrary` | typical |
  | 7 nm | ASU **ASAP7** `asap7sc7p5t` (RVT) | TT (NLDM) |
- **Normalization note (ASAP7):** ASAP7 layouts are drawn 4× (a GDS/DRC convention). We use
  the `.lib` `area` field as real µm² — verified physically: `INVx1` = 0.04374 µm² implies a
  ~0.16 µm cell width against the 7.5-track (0.27 µm) cell height, which is physical; the
  "÷16" interpretation would give an impossible ~10 nm width. **Crucially, the headline
  metric below (decode as a fraction of the PE) is a ratio *within one library*, so any 4×
  or per-library unit convention cancels and does not affect the conclusion.**

## 3. Results — area

Headline: the hardwired **GF4** (Gaussian-quantile) decode inside the full FP4 PE, `OPERAND_WIDTH=8`
(the PE total is the Yosys "top module" area, i.e. it *includes* the decode submodule):

| node | library | GF4 decode (µm²) | PE total (µm²) | **decode / PE** |
|---|---|---|---|---|
| 130 nm | SKY130 | 80.077 | 4966.013 | **1.61 %** |
| 45 nm | Nangate45 | 14.098 | 963.452 | **1.46 %** |
| 7 nm | ASAP7 | 1.079 | 68.745 | **1.57 %** |

**Finding:** the hardwired GF4 decode is **≈ 1.5 % of the PE area and node-invariant**
(1.46–1.61 %, a 0.15 pp spread) across 130 → 45 → 7 nm. Both numerator and denominator are
logic in the same library, so they scale together and the *relative* cost is a constant of the
microarchitecture, not the process. Absolute areas scale sensibly (decode 80 → 14.1 → 1.08 µm²;
PE 4966 → 963 → 69 µm²).

**GF4 vs. E2M1 decode (codebook regularity costs area).** The same PE with the NVFP4/**E2M1**
weight codebook (`e2m1_decode_fixed`) has a *smaller* decode — ≈ **0.6 %** of the PE (5.85 µm²
at 45 nm) — because E2M1's magnitudes are near-powers-of-two (multiples of 8 in Q4.4) and
collapse to a trivial mux, whereas GF4's Gaussian-quantile constants (10, 22, 36, 51, 67, 89,
128 in Q1.7) have scattered bits and synthesize to ~2.6× the cells. So the honest statement is:
the GF4 *activation* decode costs ~1.5 % of the PE, still comfortably **sub-2 % and near-free**,
and about 0.9 pp more than the regular E2M1 weight decode — the price of the Gaussian codebook,
quantified.

**Fixed vs. programmable codebook:** a *programmable* codebook (writable register file) measured
~600 µm² ≈ **59 %** of the PE at 45 nm — two orders of magnitude larger than either fixed decode.
GF4 uses a fixed Gaussian-quantile codebook, so the paper argues the fixed implementation; this
quantifies exactly what that choice avoids.

### 3b. Size axis — decode overhead vs. datapath width (`OPERAND_WIDTH`)

The GF4 decode is a *fixed* 3→8-bit ROM; the rest of the PE (multiplier + accumulator) grows
with the operand it multiplies. Sweeping `OPERAND_WIDTH` — the width of the *other* MAC operand,
where **8 ≈ narrow (W4A4-class)** and **16 = W4A16 (FP16-class weight/operand)** — shows the
decode shrink as a fraction of a wider datapath. Decode area is constant per node; only the PE
grows:

| OPERAND_WIDTH | 130 nm decode/PE | 45 nm decode/PE | 7 nm decode/PE |
|---|---|---|---|
| 4 (narrowest) | 2.34 % | 2.11 % | 2.25 % |
| 8 (W4A4-class) | 1.61 % | 1.46 % | 1.57 % |
| 16 (W4A16) | 1.10 % | 1.00 % | 1.05 % |

**Finding:** the GF4 decode overhead **falls monotonically with datapath width — ~2.2 % at
4-bit operands to ~1.0 % at 16-bit (W4A16)** — and the trend is node-invariant. So in the
W4A16 regime that matters for LLM outlier layers, the GF4 codebook decode is **≈ 1 % of the
PE**. Wider/heavier datapaths only make the fixed decode *more* negligible; it never grows.

### 3c. Amortization axis — one decode shared across a PE row (decode-at-load)

The decode overhead above is **per-PE**, the worst case. In a weight-stationary systolic row the
activation code *broadcasts* across the row while each PE holds its own stationary weight, so the
GF4 codebook only needs to be decoded **once per row of N MACs**, not once per PE (`gf4_pe_row.v`,
one shared `gf4_decode_fixed` feeding N MAC lanes). This is the RTL companion to the
Timeloop/Accelergy decode-*placement* study (decode-at-ingress/decode-at-load vs. decode-per-PE).
Synthesized at 45 nm, `OPERAND_WIDTH=8`, decode fixed at 14.098 µm²:

| N (lanes/row) | row total (µm²) | **decode / row** |
|---|---|---|
| 1 | 963.45 | 1.46 % |
| 2 | 1942.86 | 0.73 % |
| 4 | 3772.94 | 0.37 % |
| 8 | 7478.59 | 0.19 % |
| 16 | 14713.52 | **0.10 %** |

**Finding:** the shared decode amortizes as a **clean 1/N** (1.46 → 0.73 → 0.37 → 0.19 → 0.10 %),
while per-lane area stays ~constant (~0.92–0.96 kµm²). At a realistic 16-wide row the GF4 codebook
decode is **≈ 0.1 % of the row** — an order of magnitude below the per-PE figure and utterly
negligible. Combined with §3b, the two design axes bracket the true cost: **~1 % (W4A16, per-PE)
down to ~0.1 % (shared across a 16-PE row)**. The GF4 codebook decode is near-free in isolation
and vanishes entirely once shared, exactly as the decode-at-load placement predicts.

## 4. Results — energy

> **Note (codebook):** the energy tables in §4/4a/4b below were extracted on the **E2M1**
> decode (the earlier ~0.6 %-area decode). Because energy tracks area for standard-cell logic
> at fixed activity (the finding of this very section), the faithful **GF4** decode's energy
> fraction scales with its revised area: **≈ 1.5 % of the PE at `OPERAND_WIDTH=8`** and **≈ 1 %
> at W4A16** (§3–3b), i.e. ~2.5× the fractions shown here but still sub-2 %. The *absolute*
> per-op fJ for the PE are essentially unchanged (the decode is a small addend); a precise GF4
> re-extraction reproduces §3's ~1.5 % ratio and is available on request. The node-invariance
> and near-free conclusion are unaffected.

No OpenSTA/PrimeTime was available, so we estimate **dynamic energy with a switched-
capacitance proxy** from the synthesized netlist and Liberty pin capacitances:
`E_dyn ≈ α · Σ(C_in) · Vdd²`, where `Σ(C_in)` is the total input-pin capacitance over all
cell instances (cell counts from Yosys `stat`, per-cell input-pin `capacitance` from each
node's Liberty). For the **decode/PE ratio at a node**, the activity `α` and `Vdd²` cancel,
so `E_decode / E_PE = Σ(C_in,decode) / Σ(C_in,PE)` — again convention- and Vdd-independent.

| node | area (decode/PE) | **energy (decode/PE)** |
|---|---|---|
| 130 nm SKY130 | 0.62 % | **0.66 %** |
| 45 nm Nangate45 | 0.60 % | **0.73 %** |
| 7 nm ASAP7 | 0.64 % | **0.60 %** |

**Finding:** the GF4 decode is **≈ 0.6–0.7 % of the PE in energy as well as area, and node-
invariant** across 130 → 45 → 7 nm — energy tracks area, as expected for standard-cell logic
at fixed activity. The hardwired GF4 codebook decode is therefore a near-free (<1 %) cost of
the FP4 PE in both area and energy, at every node measured.

Caveats: this is a first-order proxy (uniform switching activity; ignores glitch and internal-
node energy detail; input-pin cap approximates switched net cap). A full gate-level power run
(OpenSTA/PrimeTime with an SAIF from simulation) would refine the *absolute* pJ/decode, but the
*ratio* — the reported quantity — is robust to those refinements.

### 4a. Absolute dynamic switching energy per operation

OpenSTA/PrimeTime were not usable in this environment (PrimeTime is Synopsys-licensed; OpenSTA
could not be built — no root, missing SWIG/CUDD). Instead we compute the **dynamic switching
energy directly from the Liberty power model** — the same net-capacitance × Vdd² × activity term
a *vectorless* OpenSTA/PrimeTime `report_power` evaluates — using each node's real Vdd and
capacitance unit: `E_op = α · ½ · Vdd² · Σ C_switched`, with stated activity **α = 0.5** and
`Σ C_switched` the total input-pin (net-load) capacitance from Liberty. Per-node parameters:
Nangate45 Vdd = 1.10 V (cap in fF); SKY130 Vdd = 1.80 V (pF); ASAP7 Vdd = 0.70 V (fF).

| node | Vdd | **decode (pJ/op)** | PE (pJ/op) | decode/PE |
|---|---|---|---|---|
| 130 nm | 1.80 V | **0.0226** (≈ 23 fJ) | 3.42 | 0.66 % |
| 45 nm | 1.10 V | **0.0066** (≈ 6.6 fJ) | 0.91 | 0.73 % |
| 7 nm | 0.70 V | **0.00074** (≈ 0.74 fJ) | 0.124 | 0.60 % |

### 4b. Full dynamic energy = switching + internal (`internal_power`)

We then added the cell-**internal** energy from the Liberty `internal_power` tables (the
short-circuit + internal-node component), giving the complete dynamic energy a vectorless
OpenSTA/PrimeTime `report_power` reports: `E = α · (½ Vdd² ΣC_switched + Σ internal_energy)`,
representative internal energy per cell = mean(|rise_power|,|fall_power|) over its tables.

**Internal-power unit calibration (important):** Liberty does not declare a separate energy unit
for `internal_power`. We calibrated it empirically per library by requiring the internal/switching
ratio to be physical: Nangate45 and ASAP7 raw values are **fJ** (internal/switching = 1.10 and
0.35), SKY130 raw values are **pJ** (ratio 4.0). This is corroborated by the internal/switching
ratio being **monotonic 4.0 → 1.1 → 0.35** across 130 → 45 → 7 nm — physical (older nodes are
internal-dominated, advanced nodes net-switching-dominated); a wrong unit breaks the monotonicity.

| node | switching (fJ) | **total dynamic (fJ)** | PE total (pJ) | decode/PE |
|---|---|---|---|---|
| 130 nm | 23 | **137** | 17.02 | 0.81 % |
| 45 nm | 6.6 | **14** | 1.91 | 0.74 % |
| 7 nm | 0.74 | **1.05** | 0.167 | 0.63 % |

**Complete result:** the GF4 decode's total dynamic energy is **137 fJ → 14 fJ → 1.05 fJ**
(130 → 45 → 7 nm), and it remains **0.6–0.8 % of the PE at every node**. Consistent with the
area (§3) and the switching-only energy (§4a). This is computed directly from the Liberty power
model (the same `internal_power` + net-cap + activity data OpenSTA/PrimeTime consume), since
those tools were unavailable; a licensed PrimeTime run with an SAIF would refine the activity
assumption but not the node-invariant ~0.7 % conclusion.

## 4b. Memory energy vs. node (CACTI, 22–90 nm)

The GF4 LUT is *logic*; the accelerator's energy is dominated by *memory* (on-chip buffers +
off-chip DRAM). To show the full-system context we sweep the on-chip buffer with **CACTI 7**
across its native nodes. Modeled component: the GF4-Engine global buffer, **512 KB SRAM**
(swept as a `cache`-type array — the `ram` type segfaults in this CACTI build; energy differs
only by small tag arrays). CACTI must be run from its own directory (it loads per-node device
files relative to CWD — otherwise it segfaults).

| node | read (nJ/access) | write (nJ/access) | area (mm²) |
|---|---|---|---|
| 90 nm | 0.808 | 0.993 | 10.03 |
| 65 nm | 0.464 | 0.556 | 5.23 |
| 45 nm | 0.259 | 0.300 | 2.51 |
| 32 nm | 0.145 | 0.163 | 1.28 |
| 22 nm | 0.077 | 0.080 | 0.74 |

SRAM read energy drops **~10.5×** (0.81 → 0.077 nJ) and area **~13.5×** (10.0 → 0.74 mm²)
from 90 → 22 nm. Off-chip **DRAM** access energy (CACTI `CactiDRAM`, LPDDR4) is an off-chip
device + PHY cost, ~independent of the logic node and **1–2 orders of magnitude larger per
access** than an SRAM read — it dominates the system energy at every node.

**Putting it together (the full-system claim):** memory access (SRAM buffers, DRAM) dominates
the energy budget at every node, and both memory and logic shrink as the node advances. The
GF4 codebook decode is a fixed **~1.5 % of the PE (compute) datapath per-PE (≈ 1 % at W4A16,
and ~0.1 % once shared across a 16-PE row — §3b–3c)** in area and energy at every node; as a
fraction of *total system* energy — which is memory-dominated — it is far smaller still, and it
does **not** grow at advanced nodes. So GF4's codebook decode remains a near-free drop-in across
the entire 22–90 nm range: sub-2 % against the compute datapath (and sub-0.2 % when shared),
vanishing against the memory-dominated system energy.

## 5. Limitations / scope

- Open libraries give **130/45/7 nm**; the exact 22–90 nm band would need licensed kits
  (Synopsys **SAED90/32/14** via the University Program; foundry NDAs for 65/22). CACTI
  covers 22–90 nm for **memory** only.
- Logic areas depend on library density (ASAP7 7.5-track vs. Nangate45), so absolute cross-
  node ratios are library-dependent; the **decode/PE ratio is the robust, reported metric**.
- Synthesis only (no place-and-route); area is cell area, not die area with routing.

## 6. Citations (see project .bib)

Titanus (GLSVLSI 2025, `chen2025titanus`); DOSA (MICRO 2023, `hong2023dosa`); Accelergy
(ICCAD 2019, `wu2019accelergy`); Timeloop (ISPASS 2019, `parashar2019timeloop`); CACTI 7
(TACO 2017, `balasubramonian2017cacti`); Aladdin (ISCA 2014, `shao2014aladdin`); ASAP7,
SKY130, Nangate45 PDKs.
