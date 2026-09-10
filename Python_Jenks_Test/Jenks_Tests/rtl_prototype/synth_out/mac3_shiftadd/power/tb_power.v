`timescale 1ns/1ps
module tb_power;
  reg clk=0, rst_n=0, clear=0, en=0;
  reg [3:0] w_code, a_code;
  wire signed [31:0] acc_q;
  shift_add_mac dut(.clk(clk),.rst_n(rst_n),.clear(clear),.en(en),
                    .w_code(w_code),.a_code(a_code),.acc_q(acc_q));
  always #0.5 clk = ~clk;           // 1.0 ns period = 1 GHz
  integer i;
  initial begin
    $dumpfile("synth_out/mac3_shiftadd/power/power.vcd");
    $dumpvars(0, tb_power.dut);
    w_code=0; a_code=0; rst_n=0; clear=1; en=0;
    @(posedge clk); #0.01; rst_n=1; clear=1; @(posedge clk); #0.01; clear=0; en=1;
    for (i=0;i<2000;i=i+1) begin
      w_code = $random; a_code = $random;   // exercise full 4-bit code space
      @(posedge clk);
    end
    $finish;
  end
endmodule
