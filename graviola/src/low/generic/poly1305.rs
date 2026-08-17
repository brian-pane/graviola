// Written for Graviola by Joe Birr-Pixton, 2024.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
// Originally from cifra, but later adopting the 64x64
// multiplication layout from poly1305-donna.

use super::blockwise::Blockwise;
pub(crate) struct Poly1305 {
    /// Current accumulator
    h: [u64; 3],

    /// Block multiplier
    r: [u64; 3],

    /// r[1..3] times (5 << 2), precomputed
    r5: [u64; 2],

    /// Final XOR offset
    s: [u8; 16],

    /// Unprocessed input
    bw: Blockwise<16>,
}

impl Poly1305 {
    pub(crate) fn new(key: &[u8; 32]) -> Self {
        let h = [0; 3];
        let mut r = to_limbs(&key[0..16].try_into().unwrap());
        r[0] &= 0xffc0fffffff;
        r[1] &= 0xfffffc0ffff;
        r[2] &= 0x00ffffffc0f;
        const MULTIPLIER: u64 = 5 << 2;
        let r5 = [r[1] * MULTIPLIER, r[2] * MULTIPLIER];
        let s = key[16..32].try_into().unwrap();
        Self {
            h,
            r,
            r5,
            s,
            bw: Blockwise::new(),
        }
    }

    pub(crate) fn add_bytes(&mut self, bytes: &[u8]) {
        let bytes = self.bw.add_leading(bytes);

        if let Some(block) = self.bw.take() {
            self.process_whole_block(&block, false);
        }

        let mut full_blocks = bytes.chunks_exact(16);
        for block in full_blocks.by_ref() {
            self.process_whole_block(block.try_into().unwrap(), false);
        }

        self.bw.add_trailing(full_blocks.remainder());
    }

    pub(crate) fn finish(mut self) -> [u8; 16] {
        if let Some(block) = self.bw.clone().peek_remaining() {
            self.process_last_block(block);
        }

        full_reduce(&mut self.h);

        // add s with carry
        let s = to_limbs(&self.s);
        self.h[0] += s[0];
        let carry = self.h[0] >> 44;
        self.h[0] &= 0xfffffffffff;
        self.h[1] += s[1] + carry;
        let carry = self.h[1] >> 44;
        self.h[1] &= 0xfffffffffff;
        self.h[2] += s[2] + carry;
        self.h[2] &= 0x3ffffffffff;

        // redistribute into 2 words
        self.h[0] |= self.h[1] << 44;
        self.h[1] = (self.h[1] >> 20) | (self.h[2] << 24);

        let mut r = [0; 16];
        r[0..8].copy_from_slice(&self.h[0].to_le_bytes());
        r[8..16].copy_from_slice(&self.h[1].to_le_bytes());
        r
    }

    fn process_whole_block(&mut self, inp: &[u8; 16], is_final: bool) {
        let mut block = to_limbs(inp);
        block[2] |= ((!is_final) as u64) << 40;
        self.process_block(&block);
    }

    fn process_last_block(&mut self, inp: &[u8]) {
        let mut bytes = [0u8; 16];
        bytes[..inp.len()].copy_from_slice(inp);
        bytes[inp.len()] = 0x01;
        self.process_whole_block(&bytes, true);
    }

    fn process_block(&mut self, block: &[u64; 3]) {
        add(&mut self.h, block);
        mul(&mut self.h, &self.r, &self.r5);
    }
}

fn read64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn to_limbs(bytes: &[u8; 16]) -> [u64; 3] {
    let low = read64(&bytes[0..8]);
    let high = read64(&bytes[8..16]);
    [
        low & 0xfffffffffff,
        ((low >> 44) | (high << 20)) & 0xfffffffffff,
        (high >> 24) & 0x3ffffffffff,
    ]
}

fn add(h: &mut [u64; 3], x: &[u64; 3]) {
    h[0] = h[0].wrapping_add(x[0]);
    h[1] = h[1].wrapping_add(x[1]);
    h[2] = h[2].wrapping_add(x[2]);
}

fn mul(h: &mut [u64; 3], r: &[u64; 3], s: &[u64; 2]) {
    fn mul64(a: u64, b: u64) -> u128 {
        u128::from(a) * u128::from(b)
    }

    let d0 = mul64(h[0], r[0])
        + mul64(h[1], s[1])
        + mul64(h[2], s[0]);
    let d1 = mul64(h[0], r[1])
        + mul64(h[1], r[0])
        + mul64(h[2], s[1]);
    let d2 = mul64(h[0], r[2])
        + mul64(h[1], r[1])
        + mul64(h[2], r[0]);

    // partial reduction
    let carry = d0 >> 44;
    h[0] = d0 as u64 & 0xfffffffffff;
    let d1 = d1 + carry;

    let carry = d1 >> 44;
    h[1] = d1 as u64 & 0xfffffffffff;
    let d2 = d2 + carry;

    let carry = d2 >> 42;
    h[2] = d2 as u64 & 0x3ffffffffff;

    let carry = carry as u64;
    h[0] += carry * 5;

    let carry = h[0] >> 44;
    h[0] &= 0xfffffffffff;
    h[1] += carry;
}

fn full_reduce(h: &mut [u64; 3]) {
    min_reduce(h);
    maybe_sub_130_5(h);
}

fn min_reduce(h: &mut [u64; 3]) {
    let carry = h[1] >> 44;
    h[1] &= 0xfffffffffff;
    h[2] = h[2].wrapping_add(carry);

    let carry = h[2] >> 42;
    h[2] &= 0x3ffffffffff;
    h[0] = h[0].wrapping_add(carry * 5);

    let carry = h[0] >> 44;
    h[0] &= 0xfffffffffff;
    h[1] = h[1].wrapping_add(carry);

    let carry = h[1] >> 44;
    h[1] &= 0xfffffffffff;
    h[2] = h[2].wrapping_add(carry);

    let carry = h[2] >> 42;
    h[2] &= 0x3ffffffffff;
    h[0] = h[0].wrapping_add(carry * 5);

    let carry = h[0] >> 44;
    h[0] &= 0xfffffffffff;
    h[1] = h[1].wrapping_add(carry);
}

fn maybe_sub_130_5(h: &mut [u64; 3]) {
    let g0 = h[0].wrapping_add(5);
    let carry = g0 >> 44;
    let g0 = g0 & 0xfffffffffff;

    let g1 = h[1].wrapping_add(carry);
    let carry = g1 >> 44;
    let g1 = g1 & 0xfffffffffff;

    let g2 = h[2].wrapping_add(carry).wrapping_sub(1u64 << 42);

    const HIGH_BIT: u64 = 1u64 << 63;
    let negative_mask = equal_mask(g2 & HIGH_BIT, HIGH_BIT);
    let positive_mask = !negative_mask;

    h[0] = (h[0] & negative_mask) | (g0 & positive_mask);
    h[1] = (h[1] & negative_mask) | (g1 & positive_mask);
    h[2] = (h[2] & negative_mask) | (g2 & positive_mask);
}

/// Produce 0xffffffff if x == y, zero
fn equal_mask(x: u64, y: u64) -> u64 {
    let diff = x ^ y;
    let diff_is_zero = !diff & diff.wrapping_sub(1);
    0u64.wrapping_sub(diff_is_zero >> 63)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_vectors() {
        // From draft-agl-tls-chacha20poly1305-04 section 7
        let key = &[
            0x74, 0x68, 0x69, 0x73, 0x20, 0x69, 0x73, 0x20, 0x33, 0x32, 0x2d, 0x62, 0x79, 0x74,
            0x65, 0x20, 0x6b, 0x65, 0x79, 0x20, 0x66, 0x6f, 0x72, 0x20, 0x50, 0x6f, 0x6c, 0x79,
            0x31, 0x33, 0x30, 0x35,
        ];

        let mut p = Poly1305::new(key);
        p.add_bytes(&[0u8; 32]);
        assert_eq!(
            p.finish(),
            [
                0x49, 0xec, 0x78, 0x09, 0x0e, 0x48, 0x1e, 0xc6, 0xc2, 0x6b, 0x33, 0xb9, 0x1c, 0xcc,
                0x03, 0x07
            ]
        );

        let mut p = Poly1305::new(key);
        p.add_bytes(&[
            0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x21,
        ]);
        assert_eq!(
            p.finish(),
            [
                0xa6, 0xf7, 0x45, 0x00, 0x8f, 0x81, 0xc9, 0x16, 0xa2, 0x0d, 0xcc, 0x74, 0xee, 0xf2,
                0xb2, 0xf0
            ]
        );
    }
}
