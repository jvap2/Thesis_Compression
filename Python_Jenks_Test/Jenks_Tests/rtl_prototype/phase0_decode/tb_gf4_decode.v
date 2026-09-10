// -----------------------------------------------------------------------------
// tb_gf4_decode.v
//
// Behavioral testbench for gf4_decode_programmable.v and gf4_decode_fixed.v.
// Not synthesizable -- run with a Verilog event simulator such as Icarus
// Verilog, ModelSim/Questa, VCS, or Verilator (with --timing), e.g.:
//
//   iverilog -o sim gf4_decode_fixed.v gf4_decode_programmable.v tb_gf4_decode.v
//   vvp sim
//
// Checks:
//   1) Exhaustive readback of the GF4 programmable table against the paper's
//      Eq. 7 levels L = {0.000,0.080,0.174,0.283,0.395,0.525,0.696,1.000},
//      in Q0.8 fixed point (value*256, rounded).
//   2) Exhaustive readback of the fixed E2M1 table against
//      {0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0}, in Q4.4 fixed point (value*16).
//   3) The paper's own worked numerical example (methods section, GF4
//      quantization worked example): xi = -0.7, block RMS = 0.5, alpha = 2.5
//      => sb = 1.25, |xi|/sb = 0.56, nearest level = 0.525 (idx 5),
//      reconstruction = -(1.25)(0.525) = -0.65625 (paper rounds to -0.656).
//      This value was independently cross-checked in Python outside of any
//      Verilog toolchain before being encoded here.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_decode;
 
    reg         clk = 0;
    reg         rst_n = 0;
    reg         cfg_we = 0;
    reg  [2:0]  cfg_waddr = 0;
    reg  [8:0]  cfg_wdata = 0;
    reg  [2:0]  idx = 0;
    reg         sign_in = 0;
 
    wire        sign_out_prog;
    wire [8:0]  mag_out_prog;
 
    wire        sign_out_fixed;
    wire [7:0]  mag_out_fixed;
 
    integer     i;
    integer     errors = 0;
 
    reg [8:0] expected_gf4  [0:7];
    reg [7:0] expected_e2m1 [0:7];
 
    // Fixed-point (not `real`, for portability across lint/synth-adjacent
    // Verilog frontends that don't accept the `real` type) reconstruction
    // check for the worked example. sb = 1.25 is represented in Q8.8
    // (sb_q8_8 = 320); mag_out_prog is Q0.8. Their product is Q8.16, so
    // recon_q8_16 / 65536.0 is the reconstructed real value.
    reg  [15:0] sb_q8_8;
    reg  [31:0] recon_q8_16;
 
    initial begin
        expected_gf4[0] = 9'd0;    // 0.000
        expected_gf4[1] = 9'd20;   // 0.080
        expected_gf4[2] = 9'd45;   // 0.174
        expected_gf4[3] = 9'd72;   // 0.283
        expected_gf4[4] = 9'd101;  // 0.395
        expected_gf4[5] = 9'd134;  // 0.525
        expected_gf4[6] = 9'd178;  // 0.696
        expected_gf4[7] = 9'd256;  // 1.000
 
        expected_e2m1[0] = 8'd0;   // 0.0
        expected_e2m1[1] = 8'd8;   // 0.5
        expected_e2m1[2] = 8'd16;  // 1.0
        expected_e2m1[3] = 8'd24;  // 1.5
        expected_e2m1[4] = 8'd32;  // 2.0
        expected_e2m1[5] = 8'd48;  // 3.0
        expected_e2m1[6] = 8'd64;  // 4.0
        expected_e2m1[7] = 8'd96;  // 6.0
    end
 
    gf4_decode_programmable #(.ENTRIES(8), .WIDTH(9)) dut_prog (
        .clk      (clk),
        .rst_n    (rst_n),
        .cfg_we   (cfg_we),
        .cfg_waddr(cfg_waddr),
        .cfg_wdata(cfg_wdata),
        .idx      (idx),
        .sign_in  (sign_in),
        .sign_out (sign_out_prog),
        .mag_out  (mag_out_prog)
    );
 
    e2m1_decode_fixed dut_fixed (
        .idx      (idx),
        .sign_in  (sign_in),
        .sign_out (sign_out_fixed),
        .mag_q4_4 (mag_out_fixed)
    );
 
    always #5 clk = ~clk;
 
    initial begin
        rst_n = 0;
        #12 rst_n = 1;
 
        // --- Check 1: exhaustive GF4 table readback (default-initialized) ---
        for (i = 0; i < 8; i = i + 1) begin
            idx = i[2:0];
            #1;
            if (mag_out_prog !== expected_gf4[i]) begin
                $display("FAIL: GF4 idx=%0d expected=%0d got=%0d", i, expected_gf4[i], mag_out_prog);
                errors = errors + 1;
            end else begin
                $display("PASS: GF4 idx=%0d mag_q0_8=%0d", i, mag_out_prog);
            end
        end
 
        // --- Check 2: exhaustive E2M1 table readback ---
        for (i = 0; i < 8; i = i + 1) begin
            idx = i[2:0];
            #1;
            if (mag_out_fixed !== expected_e2m1[i]) begin
                $display("FAIL: E2M1 idx=%0d expected=%0d got=%0d", i, expected_e2m1[i], mag_out_fixed);
                errors = errors + 1;
            end else begin
                $display("PASS: E2M1 idx=%0d mag_q4_4=%0d", i, mag_out_fixed);
            end
        end
 
        // --- Check 3: paper's worked example ---
        idx = 3'd5;
        sign_in = 1'b1;
        #1;
        if (mag_out_prog !== expected_gf4[5]) begin
            $display("FAIL: worked-example idx=5 expected=%0d got=%0d", expected_gf4[5], mag_out_prog);
            errors = errors + 1;
        end
 
        sb_q8_8     = 16'd320;              // 1.25 in Q8.8 (1.25 * 256 = 320, exact)
        recon_q8_16 = sb_q8_8 * mag_out_prog; // Q8.8 * Q0.8 = Q8.16
 
        // Expected (paper): -0.65625. Our Q0.8 table stores level 0.525 as
        // round(0.525*256)=134 rather than 134.4, so a small, expected
        // fixed-point rounding gap (~0.0014) versus the paper's real-valued
        // arithmetic is normal here, not a bug -- widen WIDTH in
        // gf4_decode_programmable.v if you need tighter fidelity.
        // (recon_q8_16 / 65536.0, computed off-line, is ~0.654; see comment above)
        $display("Worked example: mag_q0_8=%0d, sb_q8_8=%0d, recon_q8_16=%0d",
                  mag_out_prog, sb_q8_8, recon_q8_16);
        // recon_q8_16 should be close to 0.65625 * 65536 = 43008 (sign applied separately)
        if ((recon_q8_16 > 32'd43100) || (recon_q8_16 < 32'd42700)) begin
            $display("FAIL: worked-example reconstruction out of expected fixed-point range");
            errors = errors + 1;
        end
 
        // --- Check 4: config write port overwrites the default table ---
        cfg_we    = 1'b1;
        cfg_waddr = 3'd3;
        cfg_wdata = 9'd200;
        #10;  // one full clk period (always #5 clk=~clk -> 10ns period); a real
              // simulator would more idiomatically use @(posedge clk) here, but
              // Yosys's built-in frontend (used for parse/elaborate checking in
              // this sandbox, not full simulation) rejects bare event-control
              // statements in this position, so a plain delay is used instead --
              // functionally equivalent for this synchronous single write.
        cfg_we    = 1'b0;
        idx       = 3'd3;
        #1;
        if (mag_out_prog !== 9'd200) begin
            $display("FAIL: cfg write to idx=3 expected=200 got=%0d", mag_out_prog);
            errors = errors + 1;
        end else begin
            $display("PASS: cfg write readback idx=3 -> %0d", mag_out_prog);
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule