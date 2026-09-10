// ============================================================================
// tb_shift_add_mult -- exhaustive verification of shift_add_mult against the
// Python golden model. Drives all 256 (w_code, a_code) combinations and
// checks bit-exact agreement.
// ============================================================================
`timescale 1ns/1ps

module tb_shift_add_mult;
    localparam integer F      = 4;
    localparam integer PROD_W = 16;

    reg  [3:0] w_code, a_code;
    wire signed [PROD_W-1:0] prod_q;

    shift_add_mult #(.F(F), .PROD_W(PROD_W)) dut (
        .w_code(w_code), .a_code(a_code), .prod_q(prod_q));

    // golden vectors, indexed by {w_code, a_code} (8-bit address)
    reg signed [PROD_W-1:0] golden [0:255];

    integer i, j, idx;
    integer n_checked, n_fail;

    initial begin
        $readmemh("golden_products.hex", golden);

        n_checked = 0;
        n_fail    = 0;

        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                w_code = i[3:0];
                a_code = j[3:0];
                #1; // let combinational logic settle
                idx = i * 16 + j;
                n_checked = n_checked + 1;
                if (prod_q !== golden[idx]) begin
                    n_fail = n_fail + 1;
                    $display("FAIL  w_code=%04b a_code=%04b  dut=%0d (0x%h)  golden=%0d (0x%h)",
                              w_code, a_code, prod_q, prod_q, golden[idx], golden[idx]);
                end
            end
        end

        $display("----------------------------------------------------------");
        $display("shift_add_mult exhaustive check: %0d/%0d vectors matched",
                  n_checked - n_fail, n_checked);
        if (n_fail == 0) begin
            $display("PASS: all %0d (w_code,a_code) combinations bit-exact vs golden model.", n_checked);
        end else begin
            $display("FAIL: %0d mismatches.", n_fail);
            $fatal(1);
        end
        $finish;
    end
endmodule
