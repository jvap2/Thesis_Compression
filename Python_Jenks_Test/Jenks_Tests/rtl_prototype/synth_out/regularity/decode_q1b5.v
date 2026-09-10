// auto-generated decode ROM for codebook 'q1b5'
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
            3'd1: mag_q1_7 = 8'd12;
            3'd2: mag_q1_7 = 8'd24;
            3'd3: mag_q1_7 = 8'd36;
            3'd4: mag_q1_7 = 8'd52;
            3'd5: mag_q1_7 = 8'd68;
            3'd6: mag_q1_7 = 8'd88;
            3'd7: mag_q1_7 = 8'd128;
            default: mag_q1_7 = 8'd0;
        endcase
    end
endmodule
