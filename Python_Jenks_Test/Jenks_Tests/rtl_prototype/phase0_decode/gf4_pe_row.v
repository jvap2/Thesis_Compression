// -----------------------------------------------------------------------------
// gf4_pe_row.v
//
// A row of N MAC lanes that SHARE a single GF4 decode ("decode-at-load" /
// broadcast-activation amortization). In a weight-stationary systolic row the
// activation code broadcasts across the row while each lane holds its own
// stationary weight operand, so the GF4 codebook needs to be decoded ONCE per
// row, not once per PE. This module quantifies that amortization: one
// gf4_decode_fixed feeds N signed multiply-accumulate lanes, so the fixed
// decode overhead falls as 1/N of the row area.
//
// This is the RTL companion to the Timeloop/Accelergy "decode placement" study
// (decode-at-ingress / decode-at-load vs. decode-per-PE): synthesize for
// N = 1, 2, 4, 8, 16 and plot decode / row-total to show the LUT overhead
// vanishing as it is shared across more MACs.
// -----------------------------------------------------------------------------
module gf4_pe_row #(
    parameter N             = 8,   // number of MAC lanes sharing one decode
    parameter MAG_WIDTH     = 8,   // fixed GF4 decode is Q1.7 -> 8 bits
    parameter OPERAND_WIDTH = 8,   // per-lane stationary weight operand width
    parameter ACC_WIDTH     = 32
) (
    input  wire                                          clk,
    input  wire                                          rst_n,

    // shared (broadcast) activation code for the whole row
    input  wire [3:0]                                    code,       // {sign, idx[2:0]}
    input  wire                                          valid,
    input  wire                                          acc_clear,

    // per-lane stationary weight operands, packed N x OPERAND_WIDTH
    input  wire signed [N*OPERAND_WIDTH-1:0]             operand_b,

    // per-lane accumulators, packed N x ACC_WIDTH
    output wire signed [N*ACC_WIDTH-1:0]                 mac_acc
);

    wire       sign_bit = code[3];
    wire [2:0] idx      = code[2:0];

    // ---- ONE shared GF4 decode for the entire row ----
    wire            dec_sign;
    wire [MAG_WIDTH-1:0] dec_mag;
    gf4_decode_fixed u_decode (
        .idx      (idx),
        .sign_in  (sign_bit),
        .sign_out (dec_sign),
        .mag_q1_7 (dec_mag)
    );

    wire signed [MAG_WIDTH:0] dec_signed = dec_sign
        ? -$signed({1'b0, dec_mag})
        :  $signed({1'b0, dec_mag});

    // ---- N MAC lanes, all fed by the shared decoded activation ----
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_lane
            wire signed [OPERAND_WIDTH-1:0] w_i =
                operand_b[i*OPERAND_WIDTH +: OPERAND_WIDTH];
            wire signed [MAG_WIDTH+OPERAND_WIDTH:0] product = dec_signed * w_i;

            reg signed [ACC_WIDTH-1:0] acc_reg;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc_reg <= {ACC_WIDTH{1'b0}};
                else if (acc_clear)
                    acc_reg <= {ACC_WIDTH{1'b0}};
                else if (valid)
                    acc_reg <= acc_reg + product;
            end
            assign mac_acc[i*ACC_WIDTH +: ACC_WIDTH] = acc_reg;
        end
    endgenerate

endmodule
