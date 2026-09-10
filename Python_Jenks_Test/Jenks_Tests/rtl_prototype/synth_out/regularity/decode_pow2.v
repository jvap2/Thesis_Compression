// auto-generated decode ROM for codebook 'pow2'
module gf4_decode_fixed (
    input  wire [2:0] idx,
    input  wire       sign_in,
    output wire       sign_out,
    output reg  [7:0] mag_q1_7
);
    assign sign_out = sign_in;
    always @(*) begin
        case (idx)
            3'd0: mag_q1_7 = 8'd0;
            3'd1: mag_q1_7 = 8'd8;
            3'd2: mag_q1_7 = 8'd16;
            3'd3: mag_q1_7 = 8'd32;
            3'd4: mag_q1_7 = 8'd64;
            3'd5: mag_q1_7 = 8'd64;
            3'd6: mag_q1_7 = 8'd64;
            3'd7: mag_q1_7 = 8'd128;
            default: mag_q1_7 = 8'd0;
        endcase
    end
endmodule
