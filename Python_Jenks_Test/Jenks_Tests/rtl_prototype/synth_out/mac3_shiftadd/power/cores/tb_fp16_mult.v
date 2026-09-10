`timescale 1ns/1ps
module tb; reg [15:0] a,b; wire [15:0] y; fp16_mult d(.a(a),.b(b),.y(y));
 integer i; initial begin $dumpfile("synth_out/mac3_shiftadd/power/cores/fp16_mult.vcd"); $dumpvars(0,tb.d); a=0;b=0;#1;
 for(i=0;i<2000;i=i+1) begin a=$random;b=$random;#1; end $finish; end endmodule
