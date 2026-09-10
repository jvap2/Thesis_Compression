// -----------------------------------------------------------------------------
// gf4_decode_fixed.v
//
// Fixed (hard-wired) GF4 decode ROM -- the GF4 activation decode path.
// Maps a 4-bit code {sign, idx[2:0]} to the GF4 Gaussian-quantile basis
// magnitude (positive half, normalized to max 1.0):
//   {0.0, 0.0796082, 0.1737177, 0.2828685, 0.3952704, 0.5250730, 0.6961928, 1.0}
// These are the equal-mass N(0,1) quantile levels used by the GF4 codebook
// (GF4_POS in the Python reference: triton_kernels.GF4_LEVELS / validate_multipass.GF4_POS),
// i.e. NF4 made symmetric. Represented in Q1.7 unsigned fixed point
// (mag_q1_7 = round(value * 128)); the per-block scale is applied downstream,
// so these are the normalized codebook levels and 1.0 maps to 8'd128.
//
// This is purely combinational logic (a case statement synthesizes to a
// small mux/ROM) -- there is no writable storage, matching GF4's fixed,
// non-programmable codebook. Pair with gf4_decode_programmable.v: synthesize
// each standalone and diff area/cell count to reproduce the paper's
// "fixed decode ROM vs. programmable table" comparison at the RTL level
// instead of the Accelergy component model. The NVFP4/E2M1 *weight* decode
// baseline is preserved in e2m1_decode_fixed.v.
// -----------------------------------------------------------------------------
module gf4_decode_fixed (
    input  wire [2:0] idx,       // 3-bit magnitude index
    input  wire       sign_in,   // sign bit of the 4-bit code
    output wire       sign_out,  // passthrough sign
    output reg  [7:0] mag_q1_7   // unsigned Q1.7 fixed-point magnitude, 0..128 (max 1.0)
);

    assign sign_out = sign_in;

    always @(*) begin
        case (idx)
            3'd0: mag_q1_7 = 8'd0;    // 0.0
            3'd1: mag_q1_7 = 8'd10;   // 0.0796082
            3'd2: mag_q1_7 = 8'd22;   // 0.1737177
            3'd3: mag_q1_7 = 8'd36;   // 0.2828685
            3'd4: mag_q1_7 = 8'd51;   // 0.3952704
            3'd5: mag_q1_7 = 8'd67;   // 0.5250730
            3'd6: mag_q1_7 = 8'd89;   // 0.6961928
            3'd7: mag_q1_7 = 8'd128;  // 1.0
            default: mag_q1_7 = 8'd0;
        endcase
    end

endmodule
