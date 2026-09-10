// -----------------------------------------------------------------------------
// gf4_pe_decode_then_multiply.v
//
// The fair "decode-then-multiply" baseline gf4_pe_lutxlut.v (Phase 1) needs
// to be compared against, for the specific claim the paper's conclusion
// makes: LUTxLUT "replaces decode-then-multiply entirely... removing the
// multiplier from the datapath altogether rather than just reprogramming
// what it decodes." Nothing built so far actually decodes BOTH operands and
// then multiplies them -- gf4_pe.v (Phase 0) only decodes one side (`code`)
// against a raw `operand_b`, which is the different, decode-at-ingress
// Table 3 retrofit comparison, not this one. This module decodes w_code and
// a_code independently (two instances of the same decode tables used
// elsewhere in this project) and multiplies the two decoded magnitudes
// together every cycle -- the symmetric-codebook, both-operands-4-bit W4A4
// case gf4_joint_product_table.v itself assumes (see that file's header:
// "this instance assumes weights and activations share the same GF4
// codebook").
//
// PROGRAMMABLE selects the decode table for BOTH sides together (matching
// gf4_joint_product_table.v's own symmetric assumption): 1 = both operands
// decoded via gf4_decode_programmable (the GF4 case), 0 = both via
// gf4_decode_fixed (the NVFP4/E2M1 baseline case) -- same parameter
// convention as gf4_pe.v in Phase 0. MAG_WIDTH must match PROGRAMMABLE the
// same way it does there: 9 for programmable (Q0.8), 8 for fixed (Q4.4).
//
// Because this computes the exact same mathematical quantity as
// gf4_joint_product_table.v (decode two codes, multiply the magnitudes,
// XOR the signs) via a different implementation path, its testbench reuses
// tb_gf4_joint_product.v's own already-verified 64-entry expected-product
// table as its reference, rather than deriving a fresh set of expected
// values for this file alone -- two independently-built datapaths should
// produce bit-identical results for every input combination if both are
// correct.
// -----------------------------------------------------------------------------
module gf4_pe_decode_then_multiply #(
    parameter PROGRAMMABLE = 1,   // 1 = gf4_decode_programmable (GF4) both sides, 0 = gf4_decode_fixed (NVFP4) both sides
    parameter MAG_WIDTH    = 9,   // 9 for programmable (Q0.8), 8 for fixed (Q4.4) -- must match PROGRAMMABLE, same convention as gf4_pe.v
    parameter ACC_WIDTH    = 32
) (
    input  wire                        clk,
    input  wire                        rst_n,

    // configuration port -- only used when PROGRAMMABLE == 1; drives BOTH
    // decode tables identically (they're independent instances, so a real
    // deployment with two different codebooks would need two separate cfg
    // ports -- tied together here to match gf4_joint_product_table.v's own
    // single-shared-codebook assumption)
    input  wire                        cfg_we,
    input  wire [2:0]                  cfg_waddr,
    input  wire [MAG_WIDTH-1:0]        cfg_wdata,

    input  wire [3:0]                  w_code,
    input  wire [3:0]                  a_code,
    input  wire                        valid,
    input  wire                        acc_clear,

    output wire signed [ACC_WIDTH-1:0] mac_acc
);

    wire        w_sign = w_code[3];
    wire [2:0]  w_idx  = w_code[2:0];
    wire        a_sign = a_code[3];
    wire [2:0]  a_idx  = a_code[2:0];

    wire                 w_dec_sign, a_dec_sign;
    wire [MAG_WIDTH-1:0] w_dec_mag, a_dec_mag;

    generate
        if (PROGRAMMABLE) begin : g_prog
            gf4_decode_programmable #(.ENTRIES(8), .WIDTH(MAG_WIDTH)) u_decode_w (
                .clk      (clk),
                .rst_n    (rst_n),
                .cfg_we   (cfg_we),
                .cfg_waddr(cfg_waddr),
                .cfg_wdata(cfg_wdata),
                .idx      (w_idx),
                .sign_in  (w_sign),
                .sign_out (w_dec_sign),
                .mag_out  (w_dec_mag)
            );
            gf4_decode_programmable #(.ENTRIES(8), .WIDTH(MAG_WIDTH)) u_decode_a (
                .clk      (clk),
                .rst_n    (rst_n),
                .cfg_we   (cfg_we),
                .cfg_waddr(cfg_waddr),
                .cfg_wdata(cfg_wdata),
                .idx      (a_idx),
                .sign_in  (a_sign),
                .sign_out (a_dec_sign),
                .mag_out  (a_dec_mag)
            );
        end else begin : g_fixed
            e2m1_decode_fixed u_decode_w (   // weights: NVFP4/E2M1 codebook
                .idx      (w_idx),
                .sign_in  (w_sign),
                .sign_out (w_dec_sign),
                .mag_q4_4 (w_dec_mag)
            );
            gf4_decode_fixed u_decode_a (    // activations: GF4 Gaussian-quantile codebook
                .idx      (a_idx),
                .sign_in  (a_sign),
                .sign_out (a_dec_sign),
                .mag_q1_7 (a_dec_mag)
            );
        end
    endgenerate

    // The actual "decode-then-multiply" datapath: an unsigned multiply on
    // the two decoded magnitudes, one XOR for the combined sign -- this is
    // the multiplier LUTxLUT's single table read + XOR is claimed to
    // remove entirely.
    wire                     prod_sign = w_dec_sign ^ a_dec_sign;
    wire [2*MAG_WIDTH-1:0]   prod_mag  = w_dec_mag * a_dec_mag;

    wire signed [2*MAG_WIDTH:0] prod_signed = prod_sign
        ? -$signed({1'b0, prod_mag})
        :  $signed({1'b0, prod_mag});

    reg signed [ACC_WIDTH-1:0] acc_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= {ACC_WIDTH{1'b0}};
        else if (acc_clear)
            acc_reg <= {ACC_WIDTH{1'b0}};
        else if (valid)
            acc_reg <= acc_reg + prod_signed;
    end

    assign mac_acc = acc_reg;

endmodule