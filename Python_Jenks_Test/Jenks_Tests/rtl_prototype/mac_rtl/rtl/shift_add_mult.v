// ============================================================================
// shift_add_mult -- the "no multiplier" weight x activation product.
//
// Decodes both operand codes to <= 2 signed powers of two each, forms the
// (up to) 4 pairwise exponent sums e_i+f_j, and sums the resulting signed
// one-hot terms. This is a direct RTL rendering of
//
//     w * a = s_w * s_a * sum_i sum_j (+/-) 2^(e_i + f_j)
//
// Output is EXACT (no rounding): prod_q = (w * a) * 2^F, represented as a
// plain two's-complement integer. F=4 is chosen so that the smallest
// possible combined exponent (e_i+f_j = -4, from weight exp -1 and
// activation exp -3) maps to shift amount 0 -- i.e. every lane is a pure
// LEFT shift, never a right shift, so this needs no rounding logic at all.
//
// All 4 lanes are computed unconditionally (invalid lanes are zeroed via
// the valid bits, not skipped) so this has no data-dependent control flow --
// same latency/structure regardless of which/how-many terms are actually
// present, matching the no-divergence requirement of the other two MAC
// embodiments.
// ============================================================================
module shift_add_mult #(
    parameter integer F      = 4,   // fractional bits (exact for this codebook pair)
    parameter integer PROD_W = 16   // output width, signed
)(
    input  wire [3:0]              w_code,
    input  wire [3:0]              a_code,
    output wire signed [PROD_W-1:0] prod_q   // = (w*a) * 2^F, exact integer
);

    // ---- decode ------------------------------------------------------------
    wire               sign_w, wv0, wv1;
    wire signed [3:0]  we0, we1;
    w_decode_e2m1 u_wdec (
        .code(w_code), .sign_w(sign_w), .v0(wv0), .e0(we0), .v1(wv1), .e1(we1));

    wire               sign_a, av0, av1;
    wire signed [3:0]  ae0, ae1;
    a_decode_hwcb u_adec (
        .code(a_code), .sign_a(sign_a), .v0(av0), .e0(ae0), .v1(av1), .e1(ae1));

    wire sign_prod = sign_w ^ sign_a;

    // PROD_W-wide constant "1", used as the shift seed for every lane so the
    // result of <<< is unambiguously PROD_W bits wide (avoids relying on
    // Verilog's 32-bit-literal self-determined-width rules).
    wire signed [PROD_W-1:0] ONE = {{(PROD_W-1){1'b0}}, 1'b1};

    // ---- 4 lanes: (weight term i) x (activation term j) --------------------
    // lane 00
    wire               v00   = wv0 & av0;
    wire signed [5:0]  exp00 = we0 + ae0;
    wire        [3:0]  sh00  = exp00 + F[3:0];             // guaranteed >= 0
    wire signed [PROD_W-1:0] mag00 = v00 ? (ONE <<< sh00) : {PROD_W{1'b0}};
    wire signed [PROD_W-1:0] t00   = sign_prod ? -mag00 : mag00;

    // lane 01
    wire               v01   = wv0 & av1;
    wire signed [5:0]  exp01 = we0 + ae1;
    wire        [3:0]  sh01  = exp01 + F[3:0];
    wire signed [PROD_W-1:0] mag01 = v01 ? (ONE <<< sh01) : {PROD_W{1'b0}};
    wire signed [PROD_W-1:0] t01   = sign_prod ? -mag01 : mag01;

    // lane 10
    wire               v10   = wv1 & av0;
    wire signed [5:0]  exp10 = we1 + ae0;
    wire        [3:0]  sh10  = exp10 + F[3:0];
    wire signed [PROD_W-1:0] mag10 = v10 ? (ONE <<< sh10) : {PROD_W{1'b0}};
    wire signed [PROD_W-1:0] t10   = sign_prod ? -mag10 : mag10;

    // lane 11
    wire               v11   = wv1 & av1;
    wire signed [5:0]  exp11 = we1 + ae1;
    wire        [3:0]  sh11  = exp11 + F[3:0];
    wire signed [PROD_W-1:0] mag11 = v11 ? (ONE <<< sh11) : {PROD_W{1'b0}};
    wire signed [PROD_W-1:0] t11   = sign_prod ? -mag11 : mag11;

    // ---- compressor: sum <= 4 signed one-hot terms --------------------------
    assign prod_q = t00 + t01 + t10 + t11;

endmodule
