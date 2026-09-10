`timescale 1ns/1ps
module tb_mult_gl;
  reg [3:0] w_code, a_code; wire signed [15:0] prod_q;
  shift_add_mult dut(.w_code(w_code),.a_code(a_code),.prod_q(prod_q));
  integer i;
  initial begin
    $dumpfile("synth_out/mac3_shiftadd/power/gl_mult.vcd");
    $dumpvars(0, tb_mult_gl.dut);
    w_code=0; a_code=0; #1;
    for (i=0;i<2000;i=i+1) begin w_code=$random; a_code=$random; #1; end  // change every 1ns
    $finish;
  end
endmodule
