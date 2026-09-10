#!/usr/bin/env python3
"""
golden_model.py -- independent reference model for the joint shift-add MAC.

Decodes weight/activation codes to real (float) values using the SAME
mathematical definitions as the paper (E2M1 bit layout; hardware codebook
n/16 table), but via ordinary floating-point arithmetic -- deliberately NOT
by re-deriving the RTL's power-of-two decomposition. This gives an
independent check: RTL says "w*a = sum of shifted terms", this script says
"w*a = decode(w_code) * decode(a_code)" via plain float multiply, and F=4
fixed-point scaling. If both agree exactly (they must, since every value in
both codebooks is dyadic) across all 256 code pairs, the shift-add
decomposition used in the RTL is verified correct, not just self-consistent.

Emits:
  golden_products.hex  -- $readmemh vectors for the Icarus testbench,
                           256 lines, index = {w_code[3:0], a_code[3:0]},
                           value = (w*a) * 2^F as PROD_W-bit two's complement hex.
  golden_products.csv  -- human-readable dump for spot-checking.
"""
import itertools

F = 4          # fractional bits (must match rtl/shift_add_mult.v parameter F)
PROD_W = 16    # output width (must match rtl/shift_add_mult.v parameter PROD_W)


def decode_e2m1(code: int) -> float:
    """4-bit E2M1: {sign, exp[1:0], mant}. Magnitudes {0,.5,1,1.5,2,3,4,6}."""
    s = (code >> 3) & 1
    e = (code >> 1) & 0b11
    m = code & 1
    if e == 0:
        mag = m * 0.5
    else:
        mag = (2.0 ** (e - 1)) * (1.0 + m * 0.5)
    return -mag if s else mag


def decode_hwcb(code: int) -> float:
    """4-bit hw codebook: {sign, idx[2:0]}. idx -> n in {0,2,4,6,8,10,12,16}, value=n/16."""
    s = (code >> 3) & 1
    idx = code & 0b111
    table = [0, 2, 4, 6, 8, 10, 12, 16]
    mag = table[idx] / 16.0
    return -mag if s else mag


def to_twos_complement_hex(value: int, width: int) -> str:
    if value < 0:
        value += (1 << width)
    assert 0 <= value < (1 << width)
    hexdigits = (width + 3) // 4
    return f'{value:0{hexdigits}x}'


def main():
    rows = []
    for w_code, a_code in itertools.product(range(16), range(16)):
        w = decode_e2m1(w_code)
        a = decode_hwcb(a_code)
        product = w * a
        scaled = product * (2 ** F)
        # must be an exact integer -- every value on both sides is dyadic
        assert abs(scaled - round(scaled)) < 1e-9, (w_code, a_code, w, a, scaled)
        prod_q = round(scaled)
        assert -(1 << (PROD_W - 1)) <= prod_q < (1 << (PROD_W - 1)), \
            f'overflow: w_code={w_code:x} a_code={a_code:x} prod_q={prod_q}'
        rows.append((w_code, a_code, w, a, product, prod_q))

    # $readmemh vector file, indexed by {w_code,a_code} (8-bit address, 0..255)
    with open('golden_products.hex', 'w') as f:
        for w_code in range(16):
            for a_code in range(16):
                prod_q = next(r[5] for r in rows if r[0] == w_code and r[1] == a_code)
                f.write(to_twos_complement_hex(prod_q, PROD_W) + '\n')

    with open('golden_products.csv', 'w') as f:
        f.write('w_code,a_code,w,a,w*a,prod_q(=w*a*2^F)\n')
        for w_code, a_code, w, a, product, prod_q in rows:
            f.write(f'{w_code:#06b},{a_code:#06b},{w},{a},{product},{prod_q}\n')

    print(f'Wrote {len(rows)} golden vectors (F={F}, PROD_W={PROD_W}) '
          f'to golden_products.hex / golden_products.csv')
    # quick sanity print of a few interesting cases
    for w_code, a_code in [(0b0000, 0b0000), (0b1111, 0b1111), (0b0111, 0b0111), (0b1001, 0b0101)]:
        w = decode_e2m1(w_code); a = decode_hwcb(a_code)
        print(f'  w_code={w_code:04b} (w={w:+.3f})  a_code={a_code:04b} (a={a:+.4f})  '
              f'w*a={w*a:+.5f}')


if __name__ == '__main__':
    main()
