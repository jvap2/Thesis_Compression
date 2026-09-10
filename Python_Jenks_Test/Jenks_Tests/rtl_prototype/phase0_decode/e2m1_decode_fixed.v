// -----------------------------------------------------------------------------
// gf4_decode_fixed.v
//
// Fixed (hard-wired) FP4 E2M1 decode ROM -- the NVFP4 baseline decode path.
// Maps a 4-bit code {sign, idx[2:0]} to the standard E2M1 basis magnitude
// {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, represented in Q4.4 unsigned
// fixed point (mag_q4_4 = round(value * 16)).
//
// This is purely combinational logic (a case statement synthesizes to a
// small mux/ROM) -- there is no writable storage, matching NVFP4's fixed,
// non-programmable decode. Pair with gf4_decode_programmable.v: synthesize
// each standalone and diff area/cell count to reproduce the paper's
// Table II "fixed decode ROM vs. programmable table" comparison at the RTL
// level instead of the Accelergy component model.
// -----------------------------------------------------------------------------
module e2m1_decode_fixed (
    input  wire [2:0] idx,       // 3-bit magnitude index
    input  wire       sign_in,   // sign bit of the 4-bit code
    output wire        sign_out,  // passthrough sign
    output reg  [7:0] mag_q4_4   // unsigned Q4.4 fixed-point magnitude, 0..96 (max 6.0)
);

    assign sign_out = sign_in;

    always @(*) begin
        case (idx)
            3'd0: mag_q4_4 = 8'd0;   // 0.0
            3'd1: mag_q4_4 = 8'd8;   // 0.5
            3'd2: mag_q4_4 = 8'd16;  // 1.0
            3'd3: mag_q4_4 = 8'd24;  // 1.5
            3'd4: mag_q4_4 = 8'd32;  // 2.0
            3'd5: mag_q4_4 = 8'd48;  // 3.0
            3'd6: mag_q4_4 = 8'd64;  // 4.0
            3'd7: mag_q4_4 = 8'd96;  // 6.0
            default: mag_q4_4 = 8'd0;
        endcase
    end

endmodule
