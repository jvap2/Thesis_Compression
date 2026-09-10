`timescale 1ns/1ps
module tb; reg [3:0] w,a; wire signed [15:0] p; emb1_core d(.w_code(w),.a_code(a),.prod(p));
integer i; initial begin $dumpfile("synth_out/mac3_shiftadd/power/cores/emb1_core.vcd"); $dumpvars(0,tb.d); w=0;a=0;#1;
for(i=0;i<2000;i=i+1) begin w=$random;a=$random;#1; end $finish; end endmodule
