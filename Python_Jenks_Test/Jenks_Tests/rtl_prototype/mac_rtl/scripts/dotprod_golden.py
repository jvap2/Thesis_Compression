#!/usr/bin/env python3
"""
dotprod_golden.py -- generates a random K-element (weight_code, activation_code)
sequence and its exact expected accumulated dot product, for exercising the
registered shift_add_mac accumulator (not just the single-cycle multiply).
"""
import random
from golden_model import decode_e2m1, decode_hwcb, F, PROD_W

ACC_W = 32
K = 64
SEED = 1234

random.seed(SEED)

codes = [(random.randrange(16), random.randrange(16)) for _ in range(K)]

acc = 0
for w_code, a_code in codes:
    w = decode_e2m1(w_code)
    a = decode_hwcb(a_code)
    scaled = w * a * (2 ** F)
    assert abs(scaled - round(scaled)) < 1e-9
    acc += round(scaled)

assert -(1 << (ACC_W - 1)) <= acc < (1 << (ACC_W - 1)), f'accumulator overflow: {acc}'

with open('dotprod_codes.hex', 'w') as f:
    for w_code, a_code in codes:
        byte = (w_code << 4) | a_code
        f.write(f'{byte:02x}\n')

acc_uns = acc if acc >= 0 else acc + (1 << ACC_W)
with open('dotprod_expected.hex', 'w') as f:
    f.write(f'{acc_uns:08x}\n')

print(f'K={K} random MACs, seed={SEED}')
print(f'expected accumulated dot product (Q{F} fixed point, exact) = {acc}  '
      f'(real value = {acc / (2**F)})')
print('Wrote dotprod_codes.hex, dotprod_expected.hex')
