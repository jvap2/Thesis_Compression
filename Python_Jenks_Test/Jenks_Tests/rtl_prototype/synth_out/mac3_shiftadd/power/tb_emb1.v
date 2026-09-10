`timescale 1ns/1ps
module tb_pe_power;
  reg clk=0, rst_n=0, cfg_we=0, valid=0, acc_clear=0;
  reg [5:0] cfg_waddr; reg [8:0] cfg_wdata;
  reg [3:0] w_code, a_code;
  wire signed [31:0] mac_acc;
  gf4_pe_decode_then_multiply dut(.clk(clk),.rst_n(rst_n),.cfg_we(cfg_we),.cfg_waddr(cfg_waddr[/*w*/0+:$bits(dut.cfg_waddr)]),
         .cfg_wdata(cfg_wdata),.w_code(w_code),.a_code(a_code),.valid(valid),.acc_clear(acc_clear),.mac_acc(mac_acc));
  always #0.5 clk=~clk;
  integer i;
  initial begin
    $dumpfile("synth_out/mac3_shiftadd/power/pe.vcd"); $dumpvars(0, tb_pe_power.dut);
    rst_n=0; @(posedge clk); #0.01; rst_n=1;
    // config load (ramp)
    cfg_we=1;
    for (i=0;i<8;i=i+1) begin cfg_waddr=i; cfg_wdata=(i*37+11); @(posedge clk); end
    cfg_we=0; acc_clear=1; @(posedge clk); #0.01; acc_clear=0; valid=1;
    for (i=0;i<2000;i=i+1) begin w_code=$random; a_code=$random; @(posedge clk); end
    $finish;
  end
endmodule
