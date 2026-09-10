// ============================================================================
// a_decode_hwcb -- decode a 4-bit hardware-constrained activation code into
//                  <= 2 signed powers of two.
//
// Code layout: {sign, idx[2:0]}
// idx -> magnitude numerator n (value = n/16):
//   idx 0 1 2 3 4  5  6  7
//   n   0 2 4 6 8 10 12 16
//
// Every n here is dyadic and, by construction of this codebook
// (Sec. codebook-design), popcount(n) <= 2, so each decodes to <= 2 signed
// powers of two just like the weight side. Exponents already fold in the
// 1/16 scale (value = n * 2^-4), so table entries are exponent = bitpos-4.
// ============================================================================
module a_decode_hwcb (
    input  wire [3:0]        code,      // {sign, idx[2:0]}
    output reg                sign_a,    // 1 = negative
    output reg                v0,        // term0 valid
    output reg  signed [3:0]  e0,        // term0 exponent
    output reg                v1,        // term1 valid
    output reg  signed [3:0]  e1         // term1 exponent
);
    wire       s   = code[3];
    wire [2:0] idx = code[2:0];

    always @(*) begin
        sign_a = s;
        v0 = 1'b0; e0 = 4'sd0;
        v1 = 1'b0; e1 = 4'sd0;
        case (idx)
            3'd0: begin v0 = 1'b0;                                     end // n=0
            3'd1: begin v0 = 1'b1; e0 = -4'sd3;                        end // n=2
            3'd2: begin v0 = 1'b1; e0 = -4'sd2;                        end // n=4
            3'd3: begin v0 = 1'b1; e0 = -4'sd2; v1 = 1'b1; e1 = -4'sd3; end // n=6
            3'd4: begin v0 = 1'b1; e0 = -4'sd1;                        end // n=8
            3'd5: begin v0 = 1'b1; e0 = -4'sd1; v1 = 1'b1; e1 = -4'sd3; end // n=10
            3'd6: begin v0 = 1'b1; e0 = -4'sd1; v1 = 1'b1; e1 = -4'sd2; end // n=12
            3'd7: begin v0 = 1'b1; e0 =  4'sd0;                        end // n=16
            default: begin v0 = 1'b0; v1 = 1'b0; end
        endcase
    end
endmodule
