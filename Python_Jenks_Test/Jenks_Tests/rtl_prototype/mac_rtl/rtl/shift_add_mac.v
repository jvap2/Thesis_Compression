// ============================================================================
// shift_add_mac -- registered accumulator wrapper around shift_add_mult, for
// dot-product-style use (K MACs accumulated per output element).
//
// ACC_W must have enough guard bits above PROD_W for the largest K you plan
// to accumulate without overflow: each term is <= 2^(2+F) in magnitude
// (F fractional bits, max real product magnitude 6*1.0=6 -> ~2^2.6), so K
// accumulations need roughly PROD_W + ceil(log2(K)) bits. Default here
// (ACC_W=32, PROD_W=16) comfortably covers K up to ~2^15.
// ============================================================================
module shift_add_mac #(
    parameter integer F      = 4,
    parameter integer PROD_W = 16,
    parameter integer ACC_W  = 32
)(
    input  wire                      clk,
    input  wire                      rst_n,     // async active-low
    input  wire                      clear,     // synchronous: acc <= 0 (start new dot product)
    input  wire                      en,        // accumulate this cycle
    input  wire [3:0]                w_code,
    input  wire [3:0]                a_code,
    output reg  signed [ACC_W-1:0]   acc_q      // = (running dot product) * 2^F, exact
);
    wire signed [PROD_W-1:0] prod_q;
    shift_add_mult #(.F(F), .PROD_W(PROD_W)) u_mult (
        .w_code(w_code), .a_code(a_code), .prod_q(prod_q));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_q <= {ACC_W{1'b0}};
        else if (clear)
            acc_q <= {ACC_W{1'b0}};
        else if (en)
            acc_q <= acc_q + {{(ACC_W-PROD_W){prod_q[PROD_W-1]}}, prod_q}; // sign-extend
    end
endmodule
