module emb1_core(input [3:0] w_code, input [3:0] a_code, output signed [15:0] prod);
  wire swo, sao; wire [7:0] wmag, amag;
  e2m1_decode_fixed uw(.idx(w_code[2:0]), .sign_in(w_code[3]), .sign_out(swo), .mag_q4_4(wmag));
  gf4_decode_fixed  ua(.idx(a_code[2:0]), .sign_in(a_code[3]), .sign_out(sao), .mag_q1_7(amag));
  wire [15:0] mp = wmag * amag;             // 8x8 -> 16b unsigned product
  assign prod = (swo ^ sao) ? -$signed(mp) : $signed(mp);
endmodule
