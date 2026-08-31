// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0

// SHA-512 implementation for CPUs with the x86_64 SHA-512 extension

use core::arch::x86_64::*;

pub(in crate::low) fn sha512_compress_blocks_shaext(
    state: &mut [u64; 8],
    block: &[u8],
    _token: super::cpu::HaveSha512
) {
    todo!("sha512_ext");
}
