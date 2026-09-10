// ============================================================================
// w_decode_e2m1 -- decode a 4-bit E2M1 (NVFP4) weight code into
//                  <= 2 signed powers of two.
//
// Code layout: {sign, exp[1:0], mant}
// Magnitudes:  {0, 0.5, 1, 1.5, 2, 3, 4, 6}
//
//   exp==00 (subnormal): value = mant * 0.5        -> single term, e=-1, or 0
//   exp!=00 (normal):    value = 2^(exp-1)*(1+mant*0.5)
//                         = 2^(exp-1)               [term0, always present]
//                         + mant * 2^(exp-2)         [term1, present iff mant=1]
//
// Both output exponents are plain signed integers (no bias) so they can be
// added directly to the activation-side exponents downstream.
// ============================================================================
module w_decode_e2m1 (
    input  wire [3:0]        code,      // {sign, exp[1:0], mant}
    output wire               sign_w,    // 1 = negative
    output wire               v0,        // term0 valid
    output wire signed [3:0]  e0,        // term0 exponent
    output wire               v1,        // term1 valid
    output wire signed [3:0]  e1         // term1 exponent
);
    wire       s    = code[3];
    wire [1:0] exp_ = code[2:1];
    wire       mant = code[0];

    wire subnormal = (exp_ == 2'b00);
    wire signed [3:0] base = $signed({2'b00, exp_}) - 4'sd1; // exp-1, range -1..2

    assign sign_w = s;

    // term0
    assign v0 = subnormal ? mant       : 1'b1;
    assign e0 = subnormal ? -4'sd1     : base;

    // term1 (only exists for normal codes with mant=1)
    assign v1 = subnormal ? 1'b0       : mant;
    assign e1 = base - 4'sd1;

endmodule
