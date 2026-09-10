`timescale 1ns/1ps
module tb; reg [15:0] x,y; wire [31:0] z; shift_add_mult d(.w_code(x),.a_code(y),.prod_q(z));
 integer i; initial begin $dumpfile("synth_out/mac3_shiftadd/power/cores/shiftadd_w8.vcd"); $dumpvars(0,tb.d); x=0;y=0;#1;
 for(i=0;i<2000;i=i+1) begin x=$random;y=$random;#1; end $finish; end endmodule
