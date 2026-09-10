module tb; reg [3:0] w,a; wire signed [7:0] p8; wire signed [15:0] p16;
 shift_add_mult #(.PROD_W(8))  d8(.w_code(w),.a_code(a),.prod_q(p8));
 shift_add_mult #(.PROD_W(16)) d16(.w_code(w),.a_code(a),.prod_q(p16));
 integer i,bad; initial begin bad=0;
  for(i=0;i<256;i=i+1) begin w=i[7:4]; a=i[3:0]; #1;
    if ($signed(p8) !== $signed(p16)) begin bad=bad+1; end end
  $display("PROD_W=8 vs 16: %0d mismatches / 256", bad); $finish; end endmodule
