// -----------------------------------------------------------------------------
// gf4_pe.v
//
// Minimal processing element: decode (fixed or programmable) + a signed
// multiply-accumulate, matching the paper's own PE definition in the
// Timeloop/Accelergy model ("a 16-entry codebook register file ... and a MAC
// (modeled as int4)"; NVFP4 baseline = identical array with the table
// removed). PROGRAMMABLE selects which decode submodule is instantiated so
// the two variants can be synthesized separately and diffed for area/energy,
// reproducing Table II at the RTL level instead of the Accelergy component
// model.
//
// IMPORTANT: MAG_WIDTH must match the decode module in use --
//   PROGRAMMABLE=1 (gf4_decode_programmable, Q0.8) -> MAG_WIDTH=9
//   PROGRAMMABLE=0 (gf4_decode_fixed,        Q4.4) -> MAG_WIDTH=8
// This isn't enforced by an assertion here; mismatching them will silently
// pad/truncate at the port connection rather than error, so set both
// parameters together at instantiation (see synth_fixed.ys / synth_programmable.ys).
//
// The MAC here is a plain signed multiply-accumulate over the decoded
// magnitude (sign-extended) and a signed operand of width OPERAND_WIDTH --
// it is a simplified stand-in for the real E2M1/E4M3 block-scaled multiply,
// in the same spirit as the paper modeling its MAC "as int4" rather than a
// full floating-point pipeline. Do not treat mac_acc as numerically
// representative of a production GEMM result; it exists so the decode logic
// has a realistic downstream load for synthesis and switching-activity
// purposes.
//
// OPERAND_WIDTH lets this same module stand in for two different regimes:
//   OPERAND_WIDTH=8  -- the original narrow-operand configuration (Phase 0).
//   OPERAND_WIDTH=16 -- a W4A16 configuration: one operand is still a
//     decoded 4-bit code (via gf4_decode_fixed/gf4_decode_programmable), but
//     `operand_b` now stands in for a full-precision (FP16-class) value that
//     was never itself reduced to a 4-bit code -- i.e. the weight is
//     compressed for storage/bandwidth, but the multiply happens at wider
//     precision, matching the paper's own "A16 (weight-only)" row in
//     Table I. As with the rest of this project, "FP16" is represented here
//     as a wide fixed-point value (e.g. Q8.8), not real IEEE-754 hardware.
// -----------------------------------------------------------------------------
module gf4_pe #(
    parameter PROGRAMMABLE  = 1,   // 1 = gf4_decode_programmable, 0 = gf4_decode_fixed
    parameter MAG_WIDTH     = 9,   // 9 for programmable (Q0.8), 8 for fixed (Q4.4)
    parameter OPERAND_WIDTH = 8,   // 8 = original narrow operand; 16 = W4A16 (FP16-class stand-in)
    parameter ACC_WIDTH     = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
 
    // configuration port (only used when PROGRAMMABLE == 1; tie off otherwise)
    input  wire                         cfg_we,
    input  wire [2:0]                   cfg_waddr,
    input  wire [MAG_WIDTH-1:0]         cfg_wdata,
 
    // per-cycle operands
    input  wire [3:0]                        code,        // {sign, idx[2:0]} activation/weight code
    input  wire signed [OPERAND_WIDTH-1:0]   operand_b,   // e.g. decoded weight value, or a wider (W4A16) operand
    input  wire                              valid,
    input  wire                              acc_clear,
 
    output wire signed [ACC_WIDTH-1:0]  mac_acc
);
 
    wire        sign_bit = code[3];
    wire [2:0]  idx      = code[2:0];
    wire        dec_sign;
    wire [MAG_WIDTH-1:0] dec_mag;
 
    generate
        if (PROGRAMMABLE) begin : g_prog
            gf4_decode_programmable #(.ENTRIES(8), .WIDTH(MAG_WIDTH)) u_decode (
                .clk      (clk),
                .rst_n    (rst_n),
                .cfg_we   (cfg_we),
                .cfg_waddr(cfg_waddr),
                .cfg_wdata(cfg_wdata),
                .idx      (idx),
                .sign_in  (sign_bit),
                .sign_out (dec_sign),
                .mag_out  (dec_mag)
            );
        end else begin : g_fixed
            gf4_decode_fixed u_decode (
                .idx      (idx),
                .sign_in  (sign_bit),
                .sign_out (dec_sign),
                .mag_q1_7 (dec_mag)
            );
        end
    endgenerate
 
    // Sign-extend the decoded magnitude into a signed operand.
    wire signed [MAG_WIDTH:0] dec_signed = dec_sign
        ? -$signed({1'b0, dec_mag})
        :  $signed({1'b0, dec_mag});
 
    wire signed [MAG_WIDTH+OPERAND_WIDTH:0] product = dec_signed * operand_b;
 
    reg signed [ACC_WIDTH-1:0] acc_reg;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= {ACC_WIDTH{1'b0}};
        else if (acc_clear)
            acc_reg <= {ACC_WIDTH{1'b0}};
        else if (valid)
            acc_reg <= acc_reg + product;
    end
 
    assign mac_acc = acc_reg;
 
endmodule