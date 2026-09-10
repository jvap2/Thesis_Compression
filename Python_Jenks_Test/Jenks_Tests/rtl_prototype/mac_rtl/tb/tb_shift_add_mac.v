// ============================================================================
// tb_shift_add_mac -- drives a K-length dot product through the registered
// shift_add_mac accumulator and checks the final sum against the Python
// golden model (dotprod_golden.py). Exercises accumulation/overflow-width
// behavior, not just the single-cycle multiply (that's tb_shift_add_mult).
// ============================================================================
`timescale 1ns/1ps

module tb_shift_add_mac;
    localparam integer F      = 4;
    localparam integer PROD_W = 16;
    localparam integer ACC_W  = 32;
    localparam integer K      = 64;

    reg clk = 0;
    reg rst_n;
    reg clear;
    reg en;
    reg [3:0] w_code, a_code;
    wire signed [ACC_W-1:0] acc_q;

    reg [7:0]      codes      [0:K-1];
    reg [ACC_W-1:0] golden_mem [0:0];
    reg [ACC_W-1:0] expected;

    integer k;

    shift_add_mac #(.F(F), .PROD_W(PROD_W), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .en(en),
        .w_code(w_code), .a_code(a_code), .acc_q(acc_q));

    always #5 clk = ~clk;

    initial begin
        $readmemh("dotprod_codes.hex", codes);
        $readmemh("dotprod_expected.hex", golden_mem);

        rst_n = 0; clear = 0; en = 0; w_code = 0; a_code = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        clear = 1; en = 0;
        @(negedge clk);
        clear = 0;

        for (k = 0; k < K; k = k + 1) begin
            w_code = codes[k][7:4];
            a_code = codes[k][3:0];
            en = 1;
            @(negedge clk); // acc registers on the posedge in between; sample after
        end
        en = 0;

        expected = golden_mem[0];

        $display("----------------------------------------------------------");
        $display("K=%0d dot product: acc_q=%0d (0x%h)   expected=%0d (0x%h)",
                  K, $signed(acc_q), acc_q, $signed(expected), expected);

        if (acc_q === expected) begin
            $display("PASS: shift_add_mac accumulator matches golden dot product exactly.");
        end else begin
            $display("FAIL: accumulator mismatch.");
            $fatal(1);
        end
        $finish;
    end
endmodule
