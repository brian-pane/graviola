// Written for Graviola by Joe Birr-Pixton, 2024.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
// Originally from cifra

use core::arch::aarch64::{
    uint8x16_t, uint32x4_t, vaddq_u32, vdupq_n_u32, veorq_u32, vextq_u32, vorrq_u32, vqtbl1q_u8,
    vreinterpretq_u8_u32, vreinterpretq_u16_u32, vreinterpretq_u32_u8, vreinterpretq_u32_u16,
    vreinterpretq_u32_u64, vreinterpretq_u64_u32, vrev32q_u16, vshlq_n_u32, vshrq_n_u32,
    vzip1q_u32, vzip1q_u64, vzip2q_u32, vzip2q_u64,
};
use core::mem;

pub(crate) struct ChaCha20 {
    key0: [u32; 4],
    key1: [u32; 4],
    nonce: [u32; 4],
}

fn four(b: &[u8; 16]) -> [u32; 4] {
    [
        u32::from_le_bytes(b[0..4].try_into().unwrap()),
        u32::from_le_bytes(b[4..8].try_into().unwrap()),
        u32::from_le_bytes(b[8..12].try_into().unwrap()),
        u32::from_le_bytes(b[12..16].try_into().unwrap()),
    ]
}

impl ChaCha20 {
    pub(crate) fn new(key: &[u8; 32], nonce: &[u8; 16]) -> Self {
        Self {
            key0: four(key[0..16].try_into().unwrap()),
            key1: four(key[16..32].try_into().unwrap()),
            nonce: four(nonce),
        }
    }

    pub(crate) fn cipher(&mut self, buffer: &mut [u8]) {
        let (by5, remainder) = buffer.as_chunks_mut::<320>();
        for block_x5 in by5 {
            // SAFETY: this crate depends on the `neon` cpu feature.
            unsafe { core_x5(&self.key0, &self.key1, &mut self.nonce, block_x5) };
        }
        let (blocks, remainder) = remainder.as_chunks_mut::<64>();
        for block in blocks {
            // SAFETY: this crate depends on the `neon` cpu feature.
            unsafe { core(&self.key0, &self.key1, &mut self.nonce, block) };
        }
        if !remainder.is_empty() {
            let mut stream = [0u8; 64];
            // SAFETY: this crate depends on the `neon` cpu feature.
            unsafe { core(&self.key0, &self.key1, &mut self.nonce, &mut stream) };
            for (out, key) in remainder.iter_mut().zip(stream.iter()) {
                *out ^= *key;
            }
        }
    }
}

pub(crate) struct XChaCha20(ChaCha20);

impl XChaCha20 {
    pub(crate) fn new(key: &[u8; 32], nonce: &[u8; 24]) -> Self {
        let hchacha_nonce = four(nonce[..16].try_into().unwrap());
        let mut key0 = four(key[0..16].try_into().unwrap());
        let mut key1 = four(key[16..32].try_into().unwrap());

        hchacha(&mut key0, &mut key1, &hchacha_nonce);

        let mut chacha_nonce = [0u8; 16];
        chacha_nonce[8..16].copy_from_slice(&nonce[16..24]);
        let chacha_nonce = four(&chacha_nonce);

        Self(ChaCha20 {
            key0,
            key1,
            nonce: chacha_nonce,
        })
    }

    pub(crate) fn cipher(&mut self, buffer: &mut [u8]) {
        self.0.cipher(buffer)
    }
}

// Rotate each element of a uint32x4_t left by a specified number of bits.
macro_rules! rotate_left {
    ($reg:expr, 8) => {{
        const LEFT_SHIFT_8: uint8x16_t =
            // SAFETY: the memory layout of [u8; 16] is compatible with uint8x16_t.
            unsafe {
                mem::transmute::<[u8; 16], uint8x16_t>([
                    3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
                ])
            };
        vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32($reg), LEFT_SHIFT_8))
    }};
    ($reg:expr, 16) => {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32($reg)))
    };
    ($reg:expr, $rot:literal) => {
        vorrq_u32(vshlq_n_u32($reg, $rot), vshrq_n_u32($reg, 32 - $rot))
    };
}

// Interleaved computation of a round of ChaCha for one or more blocks, each expressed as
// a quartet of vectors (a, b, c, d)
macro_rules! round {
    ($($a:ident, $b:ident, $c:ident, $d:ident),+) => {
        $( $a = vaddq_u32($a, $b); )*
        $( $d = veorq_u32($d, $a); )*
        $( $d = rotate_left!($d, 16); )*
        $( $c = vaddq_u32($c, $d); )*
        $( $b = veorq_u32($b, $c); )*
        $( $b = rotate_left!($b, 12); )*
        $( $a = vaddq_u32($a, $b); )*
        $( $d = veorq_u32($d, $a); )*
        $( $d = rotate_left!($d, 8); )*
        $( $c = vaddq_u32($c, $d); )*
        $( $b = veorq_u32($b, $c); )*
        $( $b = rotate_left!($b, 7); )*
    }
}

fn add_xor(value: uint32x4_t, initial: uint32x4_t, dst: &mut [u8; 16]) {
    // SAFETY: the memory layout of `uint32x4_t` is compatible with `u128`.
    let value128: u128 = unsafe { mem::transmute(vaddq_u32(value, initial)) };
    let current = u128::from_le_bytes(*dst);
    let result = value128 ^ current;
    dst.copy_from_slice(&result.to_le_bytes());
}

// Process one ChaCha20 block and store the result in `out`.
#[target_feature(enable = "neon")]
fn core(key0: &[u32; 4], key1: &[u32; 4], nonce: &mut [u32; 4], out: &mut [u8; 64]) {
    let mut a = u32_slice_to_uint32x4_t(&SIGMA);
    let mut b = u32_slice_to_uint32x4_t(key0);
    let mut c = u32_slice_to_uint32x4_t(key1);
    let mut d = u32_slice_to_uint32x4_t(nonce);

    let a_initial = a;
    let b_initial = b;
    let c_initial = c;
    let d_initial = d;

    for _ in 0..10 {
        round!(a, b, c, d);
        b = vextq_u32(b, b, 1);
        c = vextq_u32(c, c, 2);
        d = vextq_u32(d, d, 3);
        round!(a, b, c, d);
        b = vextq_u32(b, b, 3);
        c = vextq_u32(c, c, 2);
        d = vextq_u32(d, d, 1);
    }

    add_xor(a, a_initial, (&mut out[0..16]).try_into().unwrap());
    add_xor(b, b_initial, (&mut out[16..32]).try_into().unwrap());
    add_xor(c, c_initial, (&mut out[32..48]).try_into().unwrap());
    add_xor(d, d_initial, (&mut out[48..64]).try_into().unwrap());

    nonce[0] = nonce[0].wrapping_add(1);
}

// Process five ChaCha20 blocks in an interleaved manner and exclusive-or
// the result onto the contents of `out`.
#[target_feature(enable = "neon")]
fn core_x5(key0: &[u32; 4], key1: &[u32; 4], nonce: &mut [u32; 4], out: &mut [u8; 320]) {
    // Each block is computed from four inputs, all of which are [u32; 4]
    // a = SIGMA
    // b = key0
    // c = key1
    // d = 32-bit block counter and 96-bit nonce
    //
    // The ChaCha20 computation has two properties that facilitate SIMD parallelization:
    // 1. Within a block, all operations happen on individual elements within each
    //    of those [u32; 4] arrays, with no carry from one element to the next.
    // 2. The computation of a block does not depend on the result of any
    //    previous block. The only input that varies among blocks is d, and it
    //    can be calculated from the nonce and the block's sequence number.
    //
    // This implementation operates on five blocks at a time. Initially, the blocks are:
    //
    // Block 0:
    //  a0_0, a0_1, a0_2, a0_3 = SIGMA
    //  b0_0, b0_1, b0_2, b0_3 = key0
    //  c0_0, c0_1, c0_2, c0_3 = key1
    //  d0_0, d0_1, d0_2, d0_3 = nonce
    //
    // ...
    //
    // Block 4:
    //  a4_0, a4_1, a4_2, a4_3 = SIGMA
    //  b4_0, b4_1, b4_2, b4_3 = key0
    //  c4_0, c4_1, c4_2, c4_3 = key1
    //  d4_0, d4_1, d4_2, d4_3 = block counter and nonce, where d4_0 = d0_0.wrapping_add(4)
    //
    // We group the first four blocks into 16 vertical columns, each of which fits in a
    // uint32x4_t vector register.
    // col_a_0 = [ a0_0, a1_0, a2_0, a3_0]
    // col_a_1 = [ a0_1, a1_1, a2_1, a3_1]
    // ...
    // col_b_2 = [ b0_2, b1_2, b2_2, b3_2]
    // ...
    let mut col_a_0 = vdupq_n_u32(SIGMA[0]);
    let mut col_a_1 = vdupq_n_u32(SIGMA[1]);
    let mut col_a_2 = vdupq_n_u32(SIGMA[2]);
    let mut col_a_3 = vdupq_n_u32(SIGMA[3]);

    let mut col_b_0 = vdupq_n_u32(key0[0]);
    let mut col_b_1 = vdupq_n_u32(key0[1]);
    let mut col_b_2 = vdupq_n_u32(key0[2]);
    let mut col_b_3 = vdupq_n_u32(key0[3]);

    let mut col_c_0 = vdupq_n_u32(key1[0]);
    let mut col_c_1 = vdupq_n_u32(key1[1]);
    let mut col_c_2 = vdupq_n_u32(key1[2]);
    let mut col_c_3 = vdupq_n_u32(key1[3]);

    let mut col_d_0 = u32_slice_to_uint32x4_t(&[
        nonce[0],
        nonce[0].wrapping_add(1),
        nonce[0].wrapping_add(2),
        nonce[0].wrapping_add(3),
    ]);
    let mut col_d_1 = vdupq_n_u32(nonce[1]);
    let mut col_d_2 = vdupq_n_u32(nonce[2]);
    let mut col_d_3 = vdupq_n_u32(nonce[3]);

    // The fifth block is organised as rows. (Note: It's more efficient to do the
    // ChaCha computations with column vectors than rows, but we don't have enough
    // SIMD registers to do another set of four blocks in the inner loop. So we
    // interleave four blocks structured as columns and one block structured as
    // rows.)
    let mut row_a = u32_slice_to_uint32x4_t(&SIGMA);
    let mut row_b = u32_slice_to_uint32x4_t(key0);
    let mut row_c = u32_slice_to_uint32x4_t(key1);
    let mut row_d = vaddq_u32(
        u32_slice_to_uint32x4_t(nonce),
        u32_slice_to_uint32x4_t(&[4, 0, 0, 0]),
    );

    // Save the initial values, which we'll need to add to the block result later.
    let row_a_initial = row_a;
    let row_b_initial = row_b;
    let row_c_initial = row_c;
    let row_d_initial = u32_slice_to_uint32x4_t(nonce);

    // Compute 20 rounds of ChaCha.
    for _ in 0..10 {
        // Block n
        round!(
            col_a_0, col_b_0, col_c_0, col_d_0, col_a_1, col_b_1, col_c_1, col_d_1, col_a_2,
            col_b_2, col_c_2, col_d_2, col_a_3, col_b_3, col_c_3, col_d_3, row_a, row_b, row_c,
            row_d
        );

        // Block n+1: same operations as block n, but with the inputs shuffled.
        // For the columns, it's easy to shuffle the inputs as they're in different
        // vector registers. For the rows, we have to rearrange the lanes within
        // each vector.
        row_b = vextq_u32(row_b, row_b, 1);
        row_c = vextq_u32(row_c, row_c, 2);
        row_d = vextq_u32(row_d, row_d, 3);
        round!(
            col_a_0, col_b_1, col_c_2, col_d_3, col_a_1, col_b_2, col_c_3, col_d_0, col_a_2,
            col_b_3, col_c_0, col_d_1, col_a_3, col_b_0, col_c_1, col_d_2, row_a, row_b, row_c,
            row_d
        );
        row_b = vextq_u32(row_b, row_b, 3);
        row_c = vextq_u32(row_c, row_c, 2);
        row_d = vextq_u32(row_d, row_d, 1);
    }

    // Transpose four adjacent column vectors into row vectors.
    #[target_feature(enable = "neon")]
    fn cols_to_rows(
        c0: uint32x4_t,
        c1: uint32x4_t,
        c2: uint32x4_t,
        c3: uint32x4_t,
    ) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
        let zip0 = vzip1q_u32(c0, c1);
        let zip1 = vzip1q_u32(c2, c3);
        let row0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(zip0),
            vreinterpretq_u64_u32(zip1),
        ));
        let row1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(zip0),
            vreinterpretq_u64_u32(zip1),
        ));

        let zip0 = vzip2q_u32(c0, c1);
        let zip1 = vzip2q_u32(c2, c3);
        let row2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(zip0),
            vreinterpretq_u64_u32(zip1),
        ));
        let row3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(zip0),
            vreinterpretq_u64_u32(zip1),
        ));
        (row0, row1, row2, row3)
    }

    // The column vectors now contain the result of the ChaCha20 computation on
    // four successive blocks. We transpose them into rows to XOR the result onto
    // the output buffer.
    let (row_a_0, row_a_1, row_a_2, row_a_3) = cols_to_rows(col_a_0, col_a_1, col_a_2, col_a_3);
    add_xor(
        row_a_0,
        row_a_initial,
        (&mut out[0..16]).try_into().unwrap(),
    );
    add_xor(
        row_a_1,
        row_a_initial,
        (&mut out[64..80]).try_into().unwrap(),
    );
    add_xor(
        row_a_2,
        row_a_initial,
        (&mut out[128..144]).try_into().unwrap(),
    );
    add_xor(
        row_a_3,
        row_a_initial,
        (&mut out[192..208]).try_into().unwrap(),
    );

    let (row_b_0, row_b_1, row_b_2, row_b_3) = cols_to_rows(col_b_0, col_b_1, col_b_2, col_b_3);
    add_xor(
        row_b_0,
        row_b_initial,
        (&mut out[16..32]).try_into().unwrap(),
    );
    add_xor(
        row_b_1,
        row_b_initial,
        (&mut out[80..96]).try_into().unwrap(),
    );
    add_xor(
        row_b_2,
        row_b_initial,
        (&mut out[144..160]).try_into().unwrap(),
    );
    add_xor(
        row_b_3,
        row_b_initial,
        (&mut out[208..224]).try_into().unwrap(),
    );

    let (row0, row1, row2, row3) = cols_to_rows(col_c_0, col_c_1, col_c_2, col_c_3);
    add_xor(row0, row_c_initial, (&mut out[32..48]).try_into().unwrap());
    add_xor(row1, row_c_initial, (&mut out[96..112]).try_into().unwrap());
    add_xor(
        row2,
        row_c_initial,
        (&mut out[160..176]).try_into().unwrap(),
    );
    add_xor(
        row3,
        row_c_initial,
        (&mut out[224..240]).try_into().unwrap(),
    );

    let (row0, row1, row2, row3) = cols_to_rows(col_d_0, col_d_1, col_d_2, col_d_3);
    add_xor(row0, row_d_initial, (&mut out[48..64]).try_into().unwrap());
    const ONE: uint32x4_t = u32_slice_to_uint32x4_t(&[1, 0, 0, 0]);
    let row_d_initial = vaddq_u32(row_d_initial, ONE);
    add_xor(
        row1,
        row_d_initial,
        (&mut out[112..128]).try_into().unwrap(),
    );
    let row_d_initial = vaddq_u32(row_d_initial, ONE);
    add_xor(
        row2,
        row_d_initial,
        (&mut out[176..192]).try_into().unwrap(),
    );
    let row_d_initial = vaddq_u32(row_d_initial, ONE);
    add_xor(
        row3,
        row_d_initial,
        (&mut out[240..256]).try_into().unwrap(),
    );

    // The row vectors contain the result of the ChaCha computation on the fifth block.
    add_xor(
        row_a,
        row_a_initial,
        (&mut out[256..272]).try_into().unwrap(),
    );
    add_xor(
        row_b,
        row_b_initial,
        (&mut out[272..288]).try_into().unwrap(),
    );
    add_xor(
        row_c,
        row_c_initial,
        (&mut out[288..304]).try_into().unwrap(),
    );
    let row_d_initial = vaddq_u32(row_d_initial, ONE);
    add_xor(
        row_d,
        row_d_initial,
        (&mut out[304..320]).try_into().unwrap(),
    );

    nonce[0] = nonce[0].wrapping_add(5);
}

fn hchacha(key0: &mut [u32; 4], key1: &mut [u32; 4], nonce: &[u32; 4]) {
    let [mut z0, mut z1, mut z2, mut z3] = SIGMA;
    let &mut [mut z4, mut z5, mut z6, mut z7] = key0;
    let &mut [mut z8, mut z9, mut za, mut zb] = key1;
    let &[mut zc, mut zd, mut ze, mut zf] = nonce;

    macro_rules! quarter {
        ($a:ident, $b:ident, $c:ident, $d:ident) => {
            $a = $a.wrapping_add($b);
            $d = ($d ^ $a).rotate_left(16);
            $c = $c.wrapping_add($d);
            $b = ($b ^ $c).rotate_left(12);
            $a = $a.wrapping_add($b);
            $d = ($d ^ $a).rotate_left(8);
            $c = $c.wrapping_add($d);
            $b = ($b ^ $c).rotate_left(7);
        };
    }

    for _ in 0..10 {
        quarter!(z0, z4, z8, zc);
        quarter!(z1, z5, z9, zd);
        quarter!(z2, z6, za, ze);
        quarter!(z3, z7, zb, zf);
        quarter!(z0, z5, za, zf);
        quarter!(z1, z6, zb, zc);
        quarter!(z2, z7, z8, zd);
        quarter!(z3, z4, z9, ze);
    }

    *key0 = [z0, z1, z2, z3];
    *key1 = [zc, zd, ze, zf];
}

const fn u32_slice_to_uint32x4_t(value: &[u32; 4]) -> uint32x4_t {
    // SAFETY: the memory layout of [u32; 4] is compatible with uint32x4_t.
    unsafe { mem::transmute(*value) }
}

// b"expand 32-byte k" in little-endian
const SIGMA: [u32; 4] = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574];

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_vectors() {
        // From draft-agl-tls-chacha20poly1305-04 section 7
        let mut c = ChaCha20::new(&[0u8; 32], &[0u8; 16]);
        let mut block = [0u8; 64];
        c.cipher(&mut block);
        assert_eq!(
            block,
            [
                0x76, 0xb8, 0xe0, 0xad, 0xa0, 0xf1, 0x3d, 0x90, 0x40, 0x5d, 0x6a, 0xe5, 0x53, 0x86,
                0xbd, 0x28, 0xbd, 0xd2, 0x19, 0xb8, 0xa0, 0x8d, 0xed, 0x1a, 0xa8, 0x36, 0xef, 0xcc,
                0x8b, 0x77, 0x0d, 0xc7, 0xda, 0x41, 0x59, 0x7c, 0x51, 0x57, 0x48, 0x8d, 0x77, 0x24,
                0xe0, 0x3f, 0xb8, 0xd8, 0x4a, 0x37, 0x6a, 0x43, 0xb8, 0xf4, 0x15, 0x18, 0xa1, 0x1c,
                0xc3, 0x87, 0xb6, 0x69, 0xb2, 0xee, 0x65, 0x86
            ]
        );

        let mut key = [0u8; 32];
        key[31] = 0x01;
        let mut c = ChaCha20::new(&key, &[0u8; 16]);
        let mut block = [0u8; 64];
        c.cipher(&mut block);
        assert_eq!(
            block,
            [
                0x45, 0x40, 0xf0, 0x5a, 0x9f, 0x1f, 0xb2, 0x96, 0xd7, 0x73, 0x6e, 0x7b, 0x20, 0x8e,
                0x3c, 0x96, 0xeb, 0x4f, 0xe1, 0x83, 0x46, 0x88, 0xd2, 0x60, 0x4f, 0x45, 0x09, 0x52,
                0xed, 0x43, 0x2d, 0x41, 0xbb, 0xe2, 0xa0, 0xb6, 0xea, 0x75, 0x66, 0xd2, 0xa5, 0xd1,
                0xe7, 0xe2, 0x0d, 0x42, 0xaf, 0x2c, 0x53, 0xd7, 0x92, 0xb1, 0xc4, 0x3f, 0xea, 0x81,
                0x7e, 0x9a, 0xd2, 0x75, 0xae, 0x54, 0x69, 0x63
            ]
        );

        let mut nonce = [0u8; 16];
        nonce[15] = 0x01;
        let mut c = ChaCha20::new(&[0u8; 32], &nonce);
        let mut block = [0u8; 64];
        c.cipher(&mut block);
        assert_eq!(
            block[..60],
            [
                0xde, 0x9c, 0xba, 0x7b, 0xf3, 0xd6, 0x9e, 0xf5, 0xe7, 0x86, 0xdc, 0x63, 0x97, 0x3f,
                0x65, 0x3a, 0x0b, 0x49, 0xe0, 0x15, 0xad, 0xbf, 0xf7, 0x13, 0x4f, 0xcb, 0x7d, 0xf1,
                0x37, 0x82, 0x10, 0x31, 0xe8, 0x5a, 0x05, 0x02, 0x78, 0xa7, 0x08, 0x45, 0x27, 0x21,
                0x4f, 0x73, 0xef, 0xc7, 0xfa, 0x5b, 0x52, 0x77, 0x06, 0x2e, 0xb7, 0xa0, 0x43, 0x3e,
                0x44, 0x5f, 0x41, 0xe3
            ]
        );

        let mut nonce = [0u8; 16];
        nonce[8] = 0x01;
        let mut c = ChaCha20::new(&[0u8; 32], &nonce);
        let mut block = [0u8; 64];
        c.cipher(&mut block);
        assert_eq!(
            block,
            [
                0xef, 0x3f, 0xdf, 0xd6, 0xc6, 0x15, 0x78, 0xfb, 0xf5, 0xcf, 0x35, 0xbd, 0x3d, 0xd3,
                0x3b, 0x80, 0x09, 0x63, 0x16, 0x34, 0xd2, 0x1e, 0x42, 0xac, 0x33, 0x96, 0x0b, 0xd1,
                0x38, 0xe5, 0x0d, 0x32, 0x11, 0x1e, 0x4c, 0xaf, 0x23, 0x7e, 0xe5, 0x3c, 0xa8, 0xad,
                0x64, 0x26, 0x19, 0x4a, 0x88, 0x54, 0x5d, 0xdc, 0x49, 0x7a, 0x0b, 0x46, 0x6e, 0x7d,
                0x6b, 0xbd, 0xb0, 0x04, 0x1b, 0x2f, 0x58, 0x6b
            ]
        );

        let mut c = ChaCha20::new(
            &[
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                0x1c, 0x1d, 0x1e, 0x1f,
            ],
            &[
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                0x06, 0x07,
            ],
        );

        let mut block = [0u8; 256];
        c.cipher(&mut block);

        assert_eq!(
            block,
            [
                0xf7, 0x98, 0xa1, 0x89, 0xf1, 0x95, 0xe6, 0x69, 0x82, 0x10, 0x5f, 0xfb, 0x64, 0x0b,
                0xb7, 0x75, 0x7f, 0x57, 0x9d, 0xa3, 0x16, 0x02, 0xfc, 0x93, 0xec, 0x01, 0xac, 0x56,
                0xf8, 0x5a, 0xc3, 0xc1, 0x34, 0xa4, 0x54, 0x7b, 0x73, 0x3b, 0x46, 0x41, 0x30, 0x42,
                0xc9, 0x44, 0x00, 0x49, 0x17, 0x69, 0x05, 0xd3, 0xbe, 0x59, 0xea, 0x1c, 0x53, 0xf1,
                0x59, 0x16, 0x15, 0x5c, 0x2b, 0xe8, 0x24, 0x1a, 0x38, 0x00, 0x8b, 0x9a, 0x26, 0xbc,
                0x35, 0x94, 0x1e, 0x24, 0x44, 0x17, 0x7c, 0x8a, 0xde, 0x66, 0x89, 0xde, 0x95, 0x26,
                0x49, 0x86, 0xd9, 0x58, 0x89, 0xfb, 0x60, 0xe8, 0x46, 0x29, 0xc9, 0xbd, 0x9a, 0x5a,
                0xcb, 0x1c, 0xc1, 0x18, 0xbe, 0x56, 0x3e, 0xb9, 0xb3, 0xa4, 0xa4, 0x72, 0xf8, 0x2e,
                0x09, 0xa7, 0xe7, 0x78, 0x49, 0x2b, 0x56, 0x2e, 0xf7, 0x13, 0x0e, 0x88, 0xdf, 0xe0,
                0x31, 0xc7, 0x9d, 0xb9, 0xd4, 0xf7, 0xc7, 0xa8, 0x99, 0x15, 0x1b, 0x9a, 0x47, 0x50,
                0x32, 0xb6, 0x3f, 0xc3, 0x85, 0x24, 0x5f, 0xe0, 0x54, 0xe3, 0xdd, 0x5a, 0x97, 0xa5,
                0xf5, 0x76, 0xfe, 0x06, 0x40, 0x25, 0xd3, 0xce, 0x04, 0x2c, 0x56, 0x6a, 0xb2, 0xc5,
                0x07, 0xb1, 0x38, 0xdb, 0x85, 0x3e, 0x3d, 0x69, 0x59, 0x66, 0x09, 0x96, 0x54, 0x6c,
                0xc9, 0xc4, 0xa6, 0xea, 0xfd, 0xc7, 0x77, 0xc0, 0x40, 0xd7, 0x0e, 0xaf, 0x46, 0xf7,
                0x6d, 0xad, 0x39, 0x79, 0xe5, 0xc5, 0x36, 0x0c, 0x33, 0x17, 0x16, 0x6a, 0x1c, 0x89,
                0x4c, 0x94, 0xa3, 0x71, 0x87, 0x6a, 0x94, 0xdf, 0x76, 0x28, 0xfe, 0x4e, 0xaa, 0xf2,
                0xcc, 0xb2, 0x7d, 0x5a, 0xaa, 0xe0, 0xad, 0x7a, 0xd0, 0xf9, 0xd4, 0xb6, 0xad, 0x3b,
                0x54, 0x09, 0x87, 0x46, 0xd4, 0x52, 0x4d, 0x38, 0x40, 0x7a, 0x6d, 0xeb, 0x3a, 0xb7,
                0x8f, 0xab, 0x78, 0xc9
            ]
        );
    }

    #[test]
    fn hchacha_test_vectors() {
        // From draft-irtf-cfrg-xchacha-03

        let key = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let nonce = [
            0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x31, 0x41,
            0x59, 0x27, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
        ];

        let c = XChaCha20::new(&key, &nonce);
        assert_eq!(
            c.0.key0,
            [0x423b4182, 0xfe7bb227, 0x50420ed3, 0x737d878a],
            "{:x?}",
            c.0.key0
        );
        assert_eq!(
            c.0.key1,
            [0xd5e4f9a0, 0x53a8748a, 0x13c42ec1, 0xdcecd326],
            "{:x?}",
            c.0.key1
        );
        assert_eq!(
            c.0.nonce,
            [0, 0, 0x98badcfe, 0x10325476],
            "{:x?}",
            c.0.nonce
        );
    }

    #[test]
    fn xchacha_test_vectors() {
        // From draft-irtf-cfrg-xchacha-03, A.2
        // https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha-03#appendix-A.2

        let key = [
            0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
            0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
            0x9c, 0x9d, 0x9e, 0x9f,
        ];
        let nonce = *b"@ABCDEFGHIJKLMNOPQRSTUVX";
        let mut c = XChaCha20::new(&key, &nonce);

        let mut buffer = [
            0x54, 0x68, 0x65, 0x20, 0x64, 0x68, 0x6f, 0x6c, 0x65, 0x20, 0x28, 0x70, 0x72, 0x6f,
            0x6e, 0x6f, 0x75, 0x6e, 0x63, 0x65, 0x64, 0x20, 0x22, 0x64, 0x6f, 0x6c, 0x65, 0x22,
            0x29, 0x20, 0x69, 0x73, 0x20, 0x61, 0x6c, 0x73, 0x6f, 0x20, 0x6b, 0x6e, 0x6f, 0x77,
            0x6e, 0x20, 0x61, 0x73, 0x20, 0x74, 0x68, 0x65, 0x20, 0x41, 0x73, 0x69, 0x61, 0x74,
            0x69, 0x63, 0x20, 0x77, 0x69, 0x6c, 0x64, 0x20, 0x64, 0x6f, 0x67, 0x2c, 0x20, 0x72,
            0x65, 0x64, 0x20, 0x64, 0x6f, 0x67, 0x2c, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x77, 0x68,
            0x69, 0x73, 0x74, 0x6c, 0x69, 0x6e, 0x67, 0x20, 0x64, 0x6f, 0x67, 0x2e, 0x20, 0x49,
            0x74, 0x20, 0x69, 0x73, 0x20, 0x61, 0x62, 0x6f, 0x75, 0x74, 0x20, 0x74, 0x68, 0x65,
            0x20, 0x73, 0x69, 0x7a, 0x65, 0x20, 0x6f, 0x66, 0x20, 0x61, 0x20, 0x47, 0x65, 0x72,
            0x6d, 0x61, 0x6e, 0x20, 0x73, 0x68, 0x65, 0x70, 0x68, 0x65, 0x72, 0x64, 0x20, 0x62,
            0x75, 0x74, 0x20, 0x6c, 0x6f, 0x6f, 0x6b, 0x73, 0x20, 0x6d, 0x6f, 0x72, 0x65, 0x20,
            0x6c, 0x69, 0x6b, 0x65, 0x20, 0x61, 0x20, 0x6c, 0x6f, 0x6e, 0x67, 0x2d, 0x6c, 0x65,
            0x67, 0x67, 0x65, 0x64, 0x20, 0x66, 0x6f, 0x78, 0x2e, 0x20, 0x54, 0x68, 0x69, 0x73,
            0x20, 0x68, 0x69, 0x67, 0x68, 0x6c, 0x79, 0x20, 0x65, 0x6c, 0x75, 0x73, 0x69, 0x76,
            0x65, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x73, 0x6b, 0x69, 0x6c, 0x6c, 0x65, 0x64, 0x20,
            0x6a, 0x75, 0x6d, 0x70, 0x65, 0x72, 0x20, 0x69, 0x73, 0x20, 0x63, 0x6c, 0x61, 0x73,
            0x73, 0x69, 0x66, 0x69, 0x65, 0x64, 0x20, 0x77, 0x69, 0x74, 0x68, 0x20, 0x77, 0x6f,
            0x6c, 0x76, 0x65, 0x73, 0x2c, 0x20, 0x63, 0x6f, 0x79, 0x6f, 0x74, 0x65, 0x73, 0x2c,
            0x20, 0x6a, 0x61, 0x63, 0x6b, 0x61, 0x6c, 0x73, 0x2c, 0x20, 0x61, 0x6e, 0x64, 0x20,
            0x66, 0x6f, 0x78, 0x65, 0x73, 0x20, 0x69, 0x6e, 0x20, 0x74, 0x68, 0x65, 0x20, 0x74,
            0x61, 0x78, 0x6f, 0x6e, 0x6f, 0x6d, 0x69, 0x63, 0x20, 0x66, 0x61, 0x6d, 0x69, 0x6c,
            0x79, 0x20, 0x43, 0x61, 0x6e, 0x69, 0x64, 0x61, 0x65, 0x2e,
        ];
        c.cipher(&mut buffer);

        let expected = [
            0x45, 0x59, 0xab, 0xba, 0x4e, 0x48, 0xc1, 0x61, 0x02, 0xe8, 0xbb, 0x2c, 0x05, 0xe6,
            0x94, 0x7f, 0x50, 0xa7, 0x86, 0xde, 0x16, 0x2f, 0x9b, 0x0b, 0x7e, 0x59, 0x2a, 0x9b,
            0x53, 0xd0, 0xd4, 0xe9, 0x8d, 0x8d, 0x64, 0x10, 0xd5, 0x40, 0xa1, 0xa6, 0x37, 0x5b,
            0x26, 0xd8, 0x0d, 0xac, 0xe4, 0xfa, 0xb5, 0x23, 0x84, 0xc7, 0x31, 0xac, 0xbf, 0x16,
            0xa5, 0x92, 0x3c, 0x0c, 0x48, 0xd3, 0x57, 0x5d, 0x4d, 0x0d, 0x2c, 0x67, 0x3b, 0x66,
            0x6f, 0xaa, 0x73, 0x10, 0x61, 0x27, 0x77, 0x01, 0x09, 0x3a, 0x6b, 0xf7, 0xa1, 0x58,
            0xa8, 0x86, 0x42, 0x92, 0xa4, 0x1c, 0x48, 0xe3, 0xa9, 0xb4, 0xc0, 0xda, 0xec, 0xe0,
            0xf8, 0xd9, 0x8d, 0x0d, 0x7e, 0x05, 0xb3, 0x7a, 0x30, 0x7b, 0xbb, 0x66, 0x33, 0x31,
            0x64, 0xec, 0x9e, 0x1b, 0x24, 0xea, 0x0d, 0x6c, 0x3f, 0xfd, 0xdc, 0xec, 0x4f, 0x68,
            0xe7, 0x44, 0x30, 0x56, 0x19, 0x3a, 0x03, 0xc8, 0x10, 0xe1, 0x13, 0x44, 0xca, 0x06,
            0xd8, 0xed, 0x8a, 0x2b, 0xfb, 0x1e, 0x8d, 0x48, 0xcf, 0xa6, 0xbc, 0x0e, 0xb4, 0xe2,
            0x46, 0x4b, 0x74, 0x81, 0x42, 0x40, 0x7c, 0x9f, 0x43, 0x1a, 0xee, 0x76, 0x99, 0x60,
            0xe1, 0x5b, 0xa8, 0xb9, 0x68, 0x90, 0x46, 0x6e, 0xf2, 0x45, 0x75, 0x99, 0x85, 0x23,
            0x85, 0xc6, 0x61, 0xf7, 0x52, 0xce, 0x20, 0xf9, 0xda, 0x0c, 0x09, 0xab, 0x6b, 0x19,
            0xdf, 0x74, 0xe7, 0x6a, 0x95, 0x96, 0x74, 0x46, 0xf8, 0xd0, 0xfd, 0x41, 0x5e, 0x7b,
            0xee, 0x2a, 0x12, 0xa1, 0x14, 0xc2, 0x0e, 0xb5, 0x29, 0x2a, 0xe7, 0xa3, 0x49, 0xae,
            0x57, 0x78, 0x20, 0xd5, 0x52, 0x0a, 0x1f, 0x3f, 0xb6, 0x2a, 0x17, 0xce, 0x6a, 0x7e,
            0x68, 0xfa, 0x7c, 0x79, 0x11, 0x1d, 0x88, 0x60, 0x92, 0x0b, 0xc0, 0x48, 0xef, 0x43,
            0xfe, 0x84, 0x48, 0x6c, 0xcb, 0x87, 0xc2, 0x5f, 0x0a, 0xe0, 0x45, 0xf0, 0xcc, 0xe1,
            0xe7, 0x98, 0x9a, 0x9a, 0xa2, 0x20, 0xa2, 0x8b, 0xdd, 0x48, 0x27, 0xe7, 0x51, 0xa2,
            0x4a, 0x6d, 0x5c, 0x62, 0xd7, 0x90, 0xa6, 0x63, 0x93, 0xb9, 0x31, 0x11, 0xc1, 0xa5,
            0x5d, 0xd7, 0x42, 0x1a, 0x10, 0x18, 0x49, 0x74, 0xc7, 0xc5,
        ];

        assert_eq!(buffer, expected);
    }
}
