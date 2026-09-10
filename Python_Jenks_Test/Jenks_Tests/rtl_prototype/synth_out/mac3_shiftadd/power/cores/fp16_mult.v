module fp16_mult(input [15:0] a, input [15:0] b, output reg [15:0] y);
  wire s = a[15]^b[15];
  wire [4:0] ea=a[14:10], eb=b[14:10];
  wire na=|ea, nb=|eb;                       // normal (nonzero exp)
  wire [10:0] ma={na,a[9:0]}, mb={nb,b[9:0]};
  wire [21:0] p = ma*mb;                      // 11x11 mantissa multiply
  wire norm = p[21];
  wire [9:0] frac = norm ? p[20:11] : p[19:10];
  wire signed [7:0] e = $signed({3'b0,ea})+$signed({3'b0,eb})-15+(norm?1:0);
  always @(*) begin
    if(!na||!nb) y=16'h0000;
    else if(e<=0) y={s,15'b0};
    else if(e>=31) y={s,5'h1f,10'b0};
    else y={s,e[4:0],frac};
  end
endmodule
