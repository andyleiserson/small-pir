#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::vaddvq_u32;
#[cfg(target_feature = "avx512f")]
use std::arch::x86_64::{
    _mm512_add_epi32, _mm512_cmpge_epu32_mask, _mm512_load_si512, _mm512_mask_sub_epi32,
    _mm512_maskz_sub_epi32, _mm512_mul_epu32, _mm512_mullo_epi32, _mm512_set1_epi32,
    _mm512_set1_epi64, _mm512_setzero_si512, _mm512_shuffle_ps, _mm512_srli_epi64,
    _mm512_sub_epi32, _mm512_unpackhi_epi32, _mm512_unpacklo_epi32,
};
#[cfg(target_feature = "avx512ifma")]
use std::arch::x86_64::{_mm512_add_epi64, _mm512_madd52hi_epu64};
#[cfg(target_feature = "avx2")]
use std::arch::{
    asm,
    x86_64::{
        __m256, _addcarry_u64, _mm256_add_epi32, _mm256_add_epi64, _mm256_blendv_epi8,
        _mm256_cmpgt_epi32, _mm256_extract_epi64, _mm256_extractf128_si256, _mm256_load_si256,
        _mm256_mul_epu32, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi32,
        _mm256_set1_epi64x, _mm256_set_m128i, _mm256_setzero_si256, _mm256_shuffle_ps,
        _mm256_srli_epi64, _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpacklo_epi32,
        _mm_setzero_si128,
    },
};
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
use std::mem::transmute;
use std::{
    array::{self, TryFromSliceError},
    fmt::Debug,
    iter::zip,
    ops::{Index, IndexMut},
};

pub const Q: u32 = 0x7fffd801;

pub const CHUNK: usize = 4;

// std Simd is a struct, not a trait.
pub trait Simd:
    IndexMut<usize, Output = Self::Item>
    + for<'a> TryFrom<&'a [Self::Item], Error = TryFromSliceError>
    + Clone
    + Debug
    + Default
    + Eq
{
    type Item: Copy;

    // std calls this `LEN`
    const DIM: usize;

    // std calls this `splat`
    fn repeat(value: Self::Item) -> Self {
        Self::repeat_with(|| value)
    }

    fn repeat_with<F: FnMut() -> Self::Item>(f: F) -> Self;
}

#[allow(non_camel_case_types)]
#[repr(align(32))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct u32x8(pub [u32; 8]);

impl Index<usize> for u32x8 {
    type Output = u32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl IndexMut<usize> for u32x8 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl Simd for u32x8 {
    type Item = u32;
    const DIM: usize = 8;

    fn repeat_with<F: FnMut() -> u32>(mut f: F) -> Self {
        Self(array::from_fn(|_| f()))
    }
}

impl TryFrom<&[u32]> for u32x8 {
    type Error = TryFromSliceError;

    fn try_from(value: &[u32]) -> Result<Self, Self::Error> {
        <_>::try_from(value).map(Self)
    }
}

#[allow(non_camel_case_types)]
#[repr(align(64))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct u32x16(pub [u32; 16]);

impl Index<usize> for u32x16 {
    type Output = u32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl IndexMut<usize> for u32x16 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl Simd for u32x16 {
    type Item = u32;
    const DIM: usize = 16;

    fn repeat_with<F: FnMut() -> u32>(mut f: F) -> Self {
        Self(array::from_fn(|_| f()))
    }
}

impl TryFrom<&[u32]> for u32x16 {
    type Error = TryFromSliceError;

    fn try_from(value: &[u32]) -> Result<Self, Self::Error> {
        <_>::try_from(value).map(Self)
    }
}

// TODO: some of the additions in this file need to be changed to `wrapping_add`.

#[inline(never)]
pub fn accum_only(db: &[u32x8], _query: &u32x8, acc: &mut [u32x8]) {
    let mut tmp_acc = u32x8::default();
    for db_row in db {
        for i in 0..8 {
            tmp_acc[i] += db_row[i];
        }
    }
    acc[0] = tmp_acc;
}

// When only AVX2 is enabled, this compiles as expected and runs at 87 GB/s. When
// AVX512 is enabled, this compiles to something awful using gather instructions, and
// only gets single digit GB/s. Using u32x16 instead of u32x8 doesn't improve the
// AVX512 performance significantly.
#[inline(never)]
pub fn accum_only_unrolled<T: Simd<Item = u32>>(db: &[T], _query: &T, acc: &mut [T]) {
    assert_eq!(db.len() % 8, 0, "database length should be a multiple of 8");
    let mut tmp_acc = T::default();
    for db_chunk in db.array_chunks::<8>() {
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[0][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[1][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[2][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[3][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[4][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[5][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[6][i];
        }
        for i in 0..T::DIM {
            tmp_acc[i] += db_chunk[7][i];
        }
    }
    acc[0] = tmp_acc;
}

#[cfg(target_feature = "avx2")]
#[inline(never)]
pub fn mul_accum_vert_noreduce_asm(db: &[u32x8], _query: &u32x8, acc: &mut [u32x8]) {
    assert!(db.len() >= 128);
    assert_eq!(db.len(), acc.len());
    for (db_chunk, acc_chunk) in zip(db.array_chunks::<16>(), acc.array_chunks_mut::<16>()) {
        unsafe {
            let result: __m256;
            asm!(
                "vmovdqa (%rsi), %ymm0",
                "vmovdqa 0x20(%rsi), %ymm1",
                "vmovdqa 0x40(%rsi), %ymm2",
                "vmovdqa 0x60(%rsi), %ymm3",
                "vmovdqa 0x80(%rsi), %ymm4",
                "vmovdqa 0xa0(%rsi), %ymm5",
                "vmovdqa 0xc0(%rsi), %ymm6",
                "vmovdqa 0xe0(%rsi), %ymm7",
                "vmovdqa 0x100(%rsi), %ymm8",
                "vmovdqa 0x120(%rsi), %ymm9",
                "vmovdqa 0x140(%rsi), %ymm10",
                "vmovdqa 0x160(%rsi), %ymm11",
                "vmovdqa 0x180(%rsi), %ymm12",
                "vmovdqa 0x1a0(%rsi), %ymm13",
                "vmovdqa 0x1c0(%rsi), %ymm14",
                "vmovdqa 0x1e0(%rsi), %ymm15",
                in("rsi") &db_chunk[0] as *const u32x8,
                out("ymm0") result,
                options(att_syntax),
            );
            acc_chunk[0] = std::mem::transmute(result);
        }
    }
}

#[inline(never)]
pub fn mul_accum_vert_noreduce(db: &[u32x8], query: &u32x8, acc: &mut [u32x8]) {
    for (db_row, acc_row) in zip(db, acc) {
        for i in 0..u32x8::DIM {
            acc_row[i] = u32::wrapping_add(acc_row[i], u32::wrapping_mul(db_row[i], query[i]));
        }
    }
}

#[inline(never)]
pub fn mul_accum_vert_division(db: &[u32x8], query: &u32x8, acc: &mut [u32x8]) {
    for (db_row, acc_row) in zip(db, acc) {
        for i in 0..u32x8::DIM {
            let tmp = u64::from(db_row[i]) * u64::from(query[i]) + u64::from(acc_row[i]);
            acc_row[i] = (tmp % u64::from(Q)) as u32;
        }
    }
}

#[inline(never)]
pub fn mul_accum_vert_barrett(db: &[u32x8], query: &u32x8, acc: &mut [u32x8]) {
    let p_barrett = ((1u64 << 62) / Q as u64) as u32;
    for (db_row, acc_row) in zip(db, acc) {
        for i in 0..u32x8::DIM {
            let tmp = u64::from(db_row[i]) * u64::from(query[i]);
            let tmp_lo = (tmp & 0xffff_ffff) as u32;
            let tmp_hi = tmp >> 30;
            let c = ((tmp_hi * u64::from(p_barrett)) >> 32) as u32;
            let c_times_big_q = c.wrapping_mul(Q);
            let q = tmp_lo.wrapping_sub(c_times_big_q);

            let q1 = if q >= Q { q - Q } else { q };
            let q2 = acc_row[i] + q1;
            acc_row[i] = if q2 > Q { q2 - Q } else { q2 };
        }
    }
}

#[cfg(target_feature = "avx2")]
#[inline(never)]
pub fn mul_accum_vert_barrett_asm(db: &[u32x8], query: &u32x8, acc: &mut [u32x8]) {
    assert_eq!(
        db.len() % CHUNK,
        0,
        "database length should be a multiple of {CHUNK}"
    );
    assert_eq!(db.len(), acc.len());
    assert_eq!(CHUNK, 4);
    unsafe {
        let p_barrett = _mm256_set1_epi64x(((1u64 << 62) / Q as u64) as i64);
        let big_q = _mm256_set1_epi32(Q as i32);
        let big_q_m1 = _mm256_set1_epi32((Q - 1) as i32);
        let zero = _mm256_setzero_si256();
        let query = _mm256_load_si256(query as *const u32x8 as _);
        let q_lo = _mm256_unpacklo_epi32(query, zero);
        let q_hi = _mm256_unpackhi_epi32(query, zero);
        for (db_chunk, acc_chunk) in zip(db.array_chunks::<4>(), acc.array_chunks_mut::<4>()) {
            let a = _mm256_load_si256(&acc_chunk[0] as *const u32x8 as _);
            let d = _mm256_load_si256(&db_chunk[0] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod0 = _mm256_mul_epu32(q_lo, db_lo);
            let prod1 = _mm256_mul_epu32(q_hi, db_hi);
            let prod0_hi = _mm256_srli_epi64(prod0, 30);
            let prod1_hi = _mm256_srli_epi64(prod1, 30);
            let tmp_lo = transmute(_mm256_shuffle_ps(transmute(prod0), transmute(prod1), 0x88));
            let c0 = _mm256_mul_epu32(prod0_hi, p_barrett);
            let c1 = _mm256_mul_epu32(prod1_hi, p_barrett);
            let c = transmute(_mm256_shuffle_ps(transmute(c0), transmute(c1), 0xdd));
            let c_times_big_q = _mm256_mullo_epi32(c, big_q);
            let q1 = _mm256_sub_epi32(tmp_lo, c_times_big_q);
            // TODO: this compare be done with one less instruction as
            // (q1 ^ 0x8000_0000) > X
            let cmp1a = _mm256_cmpgt_epi32(q1, big_q_m1);
            let cmp1b = _mm256_cmpgt_epi32(zero, q1);
            let cmp1 = _mm256_or_si256(cmp1a, cmp1b);
            let q1_sub = _mm256_sub_epi32(q1, big_q);
            let q2 = _mm256_blendv_epi8(q1, q1_sub, cmp1);
            let q3 = _mm256_add_epi32(q2, a);
            let cmp3a = _mm256_cmpgt_epi32(q3, big_q_m1);
            let cmp3b = _mm256_cmpgt_epi32(zero, q3);
            let cmp3 = _mm256_or_si256(cmp3a, cmp3b);
            let q3_sub = _mm256_sub_epi32(q3, big_q);
            let q4 = _mm256_blendv_epi8(q3, q3_sub, cmp3);
            acc_chunk[0] = transmute(q4);

            let a = _mm256_load_si256(&acc_chunk[1] as *const u32x8 as _);
            let d = _mm256_load_si256(&db_chunk[1] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod0 = _mm256_mul_epu32(q_lo, db_lo);
            let prod1 = _mm256_mul_epu32(q_hi, db_hi);
            let prod0_hi = _mm256_srli_epi64(prod0, 30);
            let prod1_hi = _mm256_srli_epi64(prod1, 30);
            let tmp_lo = transmute(_mm256_shuffle_ps(transmute(prod0), transmute(prod1), 0x88));
            let c0 = _mm256_mul_epu32(prod0_hi, p_barrett);
            let c1 = _mm256_mul_epu32(prod1_hi, p_barrett);
            let c = transmute(_mm256_shuffle_ps(transmute(c0), transmute(c1), 0xdd));
            let c_times_big_q = _mm256_mullo_epi32(c, big_q);
            let q1 = _mm256_sub_epi32(tmp_lo, c_times_big_q);
            let cmp1a = _mm256_cmpgt_epi32(q1, big_q_m1);
            let cmp1b = _mm256_cmpgt_epi32(zero, q1);
            let cmp1 = _mm256_or_si256(cmp1a, cmp1b);
            let q1_sub = _mm256_sub_epi32(q1, big_q);
            let q2 = _mm256_blendv_epi8(q1, q1_sub, cmp1);
            let q3 = _mm256_add_epi32(q2, a);
            let cmp3a = _mm256_cmpgt_epi32(q3, big_q_m1);
            let cmp3b = _mm256_cmpgt_epi32(zero, q3);
            let cmp3 = _mm256_or_si256(cmp3a, cmp3b);
            let q3_sub = _mm256_sub_epi32(q3, big_q);
            let q4 = _mm256_blendv_epi8(q3, q3_sub, cmp3);
            acc_chunk[1] = transmute(q4);

            let a = _mm256_load_si256(&acc_chunk[2] as *const u32x8 as _);
            let d = _mm256_load_si256(&db_chunk[2] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod0 = _mm256_mul_epu32(q_lo, db_lo);
            let prod1 = _mm256_mul_epu32(q_hi, db_hi);
            let prod0_hi = _mm256_srli_epi64(prod0, 30);
            let prod1_hi = _mm256_srli_epi64(prod1, 30);
            let tmp_lo = transmute(_mm256_shuffle_ps(transmute(prod0), transmute(prod1), 0x88));
            let c0 = _mm256_mul_epu32(prod0_hi, p_barrett);
            let c1 = _mm256_mul_epu32(prod1_hi, p_barrett);
            let c = transmute(_mm256_shuffle_ps(transmute(c0), transmute(c1), 0xdd));
            let c_times_big_q = _mm256_mullo_epi32(c, big_q);
            let q1 = _mm256_sub_epi32(tmp_lo, c_times_big_q);
            let cmp1a = _mm256_cmpgt_epi32(q1, big_q_m1);
            let cmp1b = _mm256_cmpgt_epi32(zero, q1);
            let cmp1 = _mm256_or_si256(cmp1a, cmp1b);
            let q1_sub = _mm256_sub_epi32(q1, big_q);
            let q2 = _mm256_blendv_epi8(q1, q1_sub, cmp1);
            let q3 = _mm256_add_epi32(q2, a);
            let cmp3a = _mm256_cmpgt_epi32(q3, big_q_m1);
            let cmp3b = _mm256_cmpgt_epi32(zero, q3);
            let cmp3 = _mm256_or_si256(cmp3a, cmp3b);
            let q3_sub = _mm256_sub_epi32(q3, big_q);
            let q4 = _mm256_blendv_epi8(q3, q3_sub, cmp3);
            acc_chunk[2] = transmute(q4);

            let a = _mm256_load_si256(&acc_chunk[3] as *const u32x8 as _);
            let d = _mm256_load_si256(&db_chunk[3] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod0 = _mm256_mul_epu32(q_lo, db_lo);
            let prod1 = _mm256_mul_epu32(q_hi, db_hi);
            let prod0_hi = _mm256_srli_epi64(prod0, 30);
            let prod1_hi = _mm256_srli_epi64(prod1, 30);
            let tmp_lo = transmute(_mm256_shuffle_ps(transmute(prod0), transmute(prod1), 0x88));
            let c0 = _mm256_mul_epu32(prod0_hi, p_barrett);
            let c1 = _mm256_mul_epu32(prod1_hi, p_barrett);
            let c = transmute(_mm256_shuffle_ps(transmute(c0), transmute(c1), 0xdd));
            let c_times_big_q = _mm256_mullo_epi32(c, big_q);
            let q1 = _mm256_sub_epi32(tmp_lo, c_times_big_q);
            let cmp1a = _mm256_cmpgt_epi32(q1, big_q_m1);
            let cmp1b = _mm256_cmpgt_epi32(zero, q1);
            let cmp1 = _mm256_or_si256(cmp1a, cmp1b);
            let q1_sub = _mm256_sub_epi32(q1, big_q);
            let q2 = _mm256_blendv_epi8(q1, q1_sub, cmp1);
            let q3 = _mm256_add_epi32(q2, a);
            let cmp3a = _mm256_cmpgt_epi32(q3, big_q_m1);
            let cmp3b = _mm256_cmpgt_epi32(zero, q3);
            let cmp3 = _mm256_or_si256(cmp3a, cmp3b);
            let q3_sub = _mm256_sub_epi32(q3, big_q);
            let q4 = _mm256_blendv_epi8(q3, q3_sub, cmp3);
            acc_chunk[3] = transmute(q4);
        }
    }
}

#[cfg(target_feature = "avx512f")]
#[inline(never)]
pub fn mul_accum_vert_barrett_avx512(db: &[u32x16], query: &u32x16, acc: &mut [u32x16]) {
    unsafe {
        let p_barrett = _mm512_set1_epi64(((1u64 << 62) / Q as u64) as i64);
        let big_q = _mm512_set1_epi32(Q as i32);
        let zero = _mm512_setzero_si512();
        let query = _mm512_load_si512(query as *const u32x16 as _);
        let q_lo = _mm512_unpacklo_epi32(query, zero);
        let q_hi = _mm512_unpackhi_epi32(query, zero);
        // Unlike the AVX2 implementation, unrolling this loop gives only slight performance improvements.
        for (db_row, acc_row) in zip(db, acc) {
            let a = _mm512_load_si512(acc_row as *const u32x16 as _);
            let d = _mm512_load_si512(db_row as *const u32x16 as _);
            let db_lo = _mm512_unpacklo_epi32(d, zero);
            let db_hi = _mm512_unpackhi_epi32(d, zero);
            let prod0 = _mm512_mul_epu32(q_lo, db_lo);
            let prod1 = _mm512_mul_epu32(q_hi, db_hi);
            let prod0_hi = _mm512_srli_epi64(prod0, 30);
            let prod1_hi = _mm512_srli_epi64(prod1, 30);
            let tmp_lo = transmute(_mm512_shuffle_ps(transmute(prod0), transmute(prod1), 0x88));
            let c0 = _mm512_mul_epu32(prod0_hi, p_barrett);
            let c1 = _mm512_mul_epu32(prod1_hi, p_barrett);
            let c = transmute(_mm512_shuffle_ps(transmute(c0), transmute(c1), 0xdd));
            let c_times_big_q = _mm512_mullo_epi32(c, big_q);
            let q = _mm512_sub_epi32(tmp_lo, c_times_big_q);
            let cmp = _mm512_cmpge_epu32_mask(q, big_q);
            let q1 = _mm512_mask_sub_epi32(q, cmp, q, big_q);
            let q2 = _mm512_add_epi32(q1, a);
            let cmp = _mm512_cmpge_epu32_mask(q2, big_q);
            let q3 = _mm512_mask_sub_epi32(q2, cmp, q2, big_q);
            *acc_row = transmute(q3);
        }
    }
}

#[inline(never)]
pub fn mul_accum_hv_division(db: &[u32x16], query: &[u32x16; 4], acc: &mut [u32x16]) {
    assert_eq!(
        db.len() % CHUNK,
        0,
        "database length should be a multiple of {CHUNK}"
    );
    assert_eq!(db.len() / CHUNK, acc.len());
    assert_eq!(CHUNK, 4);
    for (d_chunk, a) in zip(db.array_chunks::<4>(), acc) {
        for j in 0..4 {
            for i in 0..u32x16::DIM {
                a[i] = ((u64::from(a[i]) + u64::from(d_chunk[j][i]) * u64::from(query[j][i]))
                    % u64::from(Q)) as u32;
            }
        }
    }
}

pub fn inner_prod_31<
    'a,
    const N: usize,
    const WIDTH: usize,
    const STRIDE: usize,
    const OFFSET: usize,
    const Q: u32,
>(
    lhs: &'a [u32x16],
    rhs: &'a [u32x16],
    result: &mut [u32x16],
) {
    //const WIDTH: usize = 1024 / u32x16::DIM;
    assert_eq!(lhs.len(), N * STRIDE, "incorrect lhs length");
    assert_eq!(rhs.len(), N * STRIDE, "incorrect rhs length");
    let p_barrett = ((1u128 << 65) / Q as u128) as u64;
    for i in 0..WIDTH {
        let mut acc_tmp: [u64; u32x16::DIM] = <_>::default();
        for j in 0..N {
            for k in 0..u32x16::DIM {
                acc_tmp[k] += u64::from(lhs[j * STRIDE + OFFSET + i][k])
                    * u64::from(rhs[j * STRIDE + OFFSET + i][k]);
            }
            if j % 4 == (N - 1) % 4 {
                for k in 0..u32x16::DIM {
                    let tmp_lo = (acc_tmp[k] & 0xffff_ffff) as u32;
                    let tmp_hi = acc_tmp[k] >> 29;
                    let c = ((u128::from(tmp_hi) * u128::from(p_barrett)) >> 36) as u32;
                    let c_times_big_q = c.wrapping_mul(Q);
                    let q = tmp_lo.wrapping_sub(c_times_big_q);
                    acc_tmp[k] = u64::from(if q >= Q { q - Q } else { q });
                }
            }
        }
        for k in 0..u32x16::DIM {
            result[OFFSET + i][k] = acc_tmp[k] as u32;
        }
    }
}

#[cfg(target_feature = "avx512ifma")]
pub fn inner_prod_31_avx512<
    'a,
    const N: usize,
    const WIDTH: usize,
    const STRIDE: usize,
    const OFFSET: usize,
    const Q: u32,
>(
    lhs: &'a [u32x16],
    rhs: &'a [u32x16],
    result: &mut [u32x16],
) {
    assert_eq!(lhs.len(), N * STRIDE, "incorrect lhs length");
    assert_eq!(rhs.len(), N * STRIDE, "incorrect rhs length");
    unsafe {
        // Safety: this should only be called on CPUs that support AVX512IFMA
        let p_barrett = _mm512_set1_epi64(((1u128 << 65) / Q as u128) as i64);
        let big_q = _mm512_set1_epi64(Q as i64);
        let zero = _mm512_setzero_si512();
        for i in 0..WIDTH {
            let mut acc_lo = _mm512_setzero_si512();
            let mut acc_hi = _mm512_setzero_si512();
            for j in 0..N {
                let lhs = _mm512_load_si512(&lhs[j * STRIDE + OFFSET + i] as *const u32x16 as _);
                let rhs = _mm512_load_si512(&rhs[j * STRIDE + OFFSET + i] as *const u32x16 as _);
                let lhs_lo = _mm512_unpacklo_epi32(lhs, zero);
                let lhs_hi = _mm512_unpackhi_epi32(lhs, zero);
                let rhs_lo = _mm512_unpacklo_epi32(rhs, zero);
                let rhs_hi = _mm512_unpackhi_epi32(rhs, zero);
                let prod_lo = _mm512_mul_epu32(lhs_lo, rhs_lo);
                let prod_hi = _mm512_mul_epu32(lhs_hi, rhs_hi);
                acc_lo = _mm512_add_epi64(acc_lo, prod_lo);
                acc_hi = _mm512_add_epi64(acc_hi, prod_hi);

                if j % 4 == (N - 1) % 4 {
                    let acc_lo_hi = _mm512_srli_epi64(acc_lo, 13);
                    let acc_hi_hi = _mm512_srli_epi64(acc_hi, 13);
                    let c_lo = _mm512_madd52hi_epu64(zero, acc_lo_hi, p_barrett);
                    let c_hi = _mm512_madd52hi_epu64(zero, acc_hi_hi, p_barrett);
                    let c_times_big_q_lo = _mm512_mullo_epi32(c_lo, big_q);
                    let c_times_big_q_hi = _mm512_mullo_epi32(c_hi, big_q);
                    let q_lo = _mm512_maskz_sub_epi32(0x5555, acc_lo, c_times_big_q_lo);
                    let q_hi = _mm512_maskz_sub_epi32(0x5555, acc_hi, c_times_big_q_hi);
                    let cmp_lo = _mm512_cmpge_epu32_mask(q_lo, big_q);
                    let cmp_hi = _mm512_cmpge_epu32_mask(q_hi, big_q);
                    acc_lo = _mm512_mask_sub_epi32(q_lo, cmp_lo, q_lo, big_q);
                    acc_hi = _mm512_mask_sub_epi32(q_hi, cmp_hi, q_hi, big_q);
                }
            }
            result[OFFSET + i] = transmute(_mm512_shuffle_ps(
                transmute(acc_lo),
                transmute(acc_hi),
                0x88,
            ));
        }
    }
}

pub fn mul_accum_hv_barrett_31<'a, J, const Q: u32>(
    db: &'a [u32x16],
    query: [&'a u32x16; CHUNK],
    acc: J,
) where
    J: ExactSizeIterator<Item = &'a mut u32x16>,
{
    assert_eq!(
        db.len() % CHUNK,
        0,
        "database length should be a multiple of {CHUNK}"
    );
    assert_eq!(db.len() / CHUNK, acc.len());
    let p_barrett = ((1u128 << 65) / Q as u128) as u64;
    for (db_chunk, acc) in zip(db.array_chunks::<CHUNK>(), acc) {
        let mut acc_tmp: [u64; u32x16::DIM] = array::from_fn(|i| u64::from(acc[i]));
        for j in 0..CHUNK {
            for i in 0..u32x16::DIM {
                acc_tmp[i] += u64::from(db_chunk[j][i]) * u64::from(query[j][i]);
            }
        }
        for i in 0..u32x16::DIM {
            let tmp_lo = (acc_tmp[i] & 0xffff_ffff) as u32;
            let tmp_hi = acc_tmp[i] >> 29;
            let c = ((u128::from(tmp_hi) * u128::from(p_barrett)) >> 36) as u32;
            let c_times_big_q = c.wrapping_mul(Q);
            let q = tmp_lo.wrapping_sub(c_times_big_q);
            acc[i] = if q >= Q { q - Q } else { q };
        }
    }
}

/*
 * This is not complete. Still need a replacement for madd52hi.

#[cfg(target_feature = "avx2")]
#[inline(never)]
pub fn mul_accum_hv_barrett_avx2_31(db: &[u32x8], query: &[u32x8; 4], acc: &mut [u32x8]) {
    assert_eq!(db.len() % CHUNK, 0, "database length should be a multiple of {CHUNK}");
    assert_eq!(db.len() / CHUNK, acc.len());
    assert_eq!(CHUNK, 4);
    unsafe {
        let p_barrett = _mm256_set1_epi64x(((1u128 << 65) / Q as u128) as i64);
        let big_q = _mm256_set1_epi32(Q as i32);
        let big_q_m1x = _mm256_set1_epi32(((Q - 1) ^ 0x8000_0000) as i32);
        let msb = _mm256_set1_epi32(0x8000_0000 as i32);
        let zero = _mm256_setzero_si256();
        let query0 = _mm256_load_si256(&query[0] as *const u32x8 as _);
        let query1 = _mm256_load_si256(&query[1] as *const u32x8 as _);
        let query2 = _mm256_load_si256(&query[2] as *const u32x8 as _);
        let query3 = _mm256_load_si256(&query[3] as *const u32x8 as _);
        let q0_lo = _mm256_unpacklo_epi32(query0, zero);
        let q0_hi = _mm256_unpackhi_epi32(query0, zero);
        let q1_lo = _mm256_unpacklo_epi32(query1, zero);
        let q1_hi = _mm256_unpackhi_epi32(query1, zero);
        let q2_lo = _mm256_unpacklo_epi32(query2, zero);
        let q2_hi = _mm256_unpackhi_epi32(query2, zero);
        let q3_lo = _mm256_unpacklo_epi32(query3, zero);
        let q3_hi = _mm256_unpackhi_epi32(query3, zero);
        for (db_chunk, acc) in zip(db.array_chunks::<4>(), acc) {
            let a = _mm256_load_si256(acc as *const u32x8 as _);
            let d = _mm256_load_si256(&db_chunk[0] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod00 = _mm256_mul_epu32(q0_lo, db_lo);
            let prod01 = _mm256_mul_epu32(q0_hi, db_hi);

            let d = _mm256_load_si256(&db_chunk[1] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod10 = _mm256_mul_epu32(q1_lo, db_lo);
            let prod11 = _mm256_mul_epu32(q1_hi, db_hi);

            let sum00 = _mm256_add_epi64(prod00, prod10);
            let sum01 = _mm256_add_epi64(prod01, prod11);

            let d = _mm256_load_si256(&db_chunk[2] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod0 = _mm256_mul_epu32(q2_lo, db_lo);
            let prod1 = _mm256_mul_epu32(q2_hi, db_hi);

            let sum10 = _mm256_add_epi64(sum00, prod0);
            let sum11 = _mm256_add_epi64(sum01, prod1);

            let d = _mm256_load_si256(&db_chunk[3] as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod0 = _mm256_mul_epu32(q3_lo, db_lo);
            let prod1 = _mm256_mul_epu32(q3_hi, db_hi);

            let sum20 = _mm256_add_epi64(sum10, prod0);
            let sum21 = _mm256_add_epi64(sum11, prod1);

            let prod0_hi = _mm256_srli_epi64(sum20, 13);
            let prod1_hi = _mm256_srli_epi64(sum21, 13);
            let tmp_lo = transmute(_mm256_shuffle_ps(transmute(sum20), transmute(sum21), 0x88));
            let c0 = _mm256_madd52hi_epu64(zero, prod0_hi, p_barrett);
            let c1 = _mm256_madd52hi_epu64(zero, prod1_hi, p_barrett);
            let c = transmute(_mm256_shuffle_ps(transmute(c0), transmute(c1), 0x88));
            let c_times_big_q = _mm256_mullo_epi32(c, big_q);
            let q1 = _mm256_sub_epi32(tmp_lo, c_times_big_q);
            // AVX2 only has a signed 32-bit compare, we invert the MSB
            // to get an unsigned compare from the signed compare.
            let q1x = _mm256_xor_epi32(q1, msb);
            let cmp1 = _mm256_cmpgt_epi32(q1x, big_q_m1x);
            let q1_sub = _mm256_sub_epi32(q1, big_q);
            let q2 = _mm256_blendv_epi8(q1, q1_sub, cmp1);
            let q3 = _mm256_add_epi32(q2, a);
            let q3x = _mm256_xor_epi32(q3, msb);
            let cmp3 = _mm256_cmpgt_epi32(q3x, big_q_m1x);
            let q3_sub = _mm256_sub_epi32(q3, big_q);
            let q4 = _mm256_blendv_epi8(q3, q3_sub, cmp3);
            *acc = transmute(q4);
        }
    }
}
*/

#[cfg(target_feature = "avx512ifma")]
#[inline(never)]
pub fn mul_accum_hv_barrett_31_avx512<'a, J, const Q: u32>(
    db: &'a [u32x16],
    query: [&'a u32x16; 4],
    acc: J,
) where
    J: ExactSizeIterator<Item = &'a mut u32x16>,
{
    assert_eq!(
        db.len() % CHUNK,
        0,
        "database length should be a multiple of {CHUNK}"
    );
    assert_eq!(db.len() / CHUNK, acc.len());
    assert_eq!(CHUNK, 4);
    unsafe {
        // Safety: this should only be called on CPUs that support AVX512IFMA
        let p_barrett = _mm512_set1_epi64(((1u128 << 65) / Q as u128) as i64);
        let big_q = _mm512_set1_epi32(Q as i32);
        let zero = _mm512_setzero_si512();
        let query0 = _mm512_load_si512(query[0] as *const u32x16 as _);
        let query1 = _mm512_load_si512(query[1] as *const u32x16 as _);
        let query2 = _mm512_load_si512(query[2] as *const u32x16 as _);
        let query3 = _mm512_load_si512(query[3] as *const u32x16 as _);
        let q0_lo = _mm512_unpacklo_epi32(query0, zero);
        let q0_hi = _mm512_unpackhi_epi32(query0, zero);
        let q1_lo = _mm512_unpacklo_epi32(query1, zero);
        let q1_hi = _mm512_unpackhi_epi32(query1, zero);
        let q2_lo = _mm512_unpacklo_epi32(query2, zero);
        let q2_hi = _mm512_unpackhi_epi32(query2, zero);
        let q3_lo = _mm512_unpacklo_epi32(query3, zero);
        let q3_hi = _mm512_unpackhi_epi32(query3, zero);
        for (db_chunk, acc) in zip(db.array_chunks::<4>(), acc) {
            let a = _mm512_load_si512(acc as *const u32x16 as _);
            let d = _mm512_load_si512(&db_chunk[0] as *const u32x16 as _);
            let db_lo = _mm512_unpacklo_epi32(d, zero);
            let db_hi = _mm512_unpackhi_epi32(d, zero);
            let prod00 = _mm512_mul_epu32(q0_lo, db_lo);
            let prod01 = _mm512_mul_epu32(q0_hi, db_hi);

            let d = _mm512_load_si512(&db_chunk[1] as *const u32x16 as _);
            let db_lo = _mm512_unpacklo_epi32(d, zero);
            let db_hi = _mm512_unpackhi_epi32(d, zero);
            let prod10 = _mm512_mul_epu32(q1_lo, db_lo);
            let prod11 = _mm512_mul_epu32(q1_hi, db_hi);

            let sum00 = _mm512_add_epi64(prod00, prod10);
            let sum01 = _mm512_add_epi64(prod01, prod11);

            let d = _mm512_load_si512(&db_chunk[2] as *const u32x16 as _);
            let db_lo = _mm512_unpacklo_epi32(d, zero);
            let db_hi = _mm512_unpackhi_epi32(d, zero);
            let prod0 = _mm512_mul_epu32(q2_lo, db_lo);
            let prod1 = _mm512_mul_epu32(q2_hi, db_hi);

            let sum10 = _mm512_add_epi64(sum00, prod0);
            let sum11 = _mm512_add_epi64(sum01, prod1);

            let d = _mm512_load_si512(&db_chunk[3] as *const u32x16 as _);
            let db_lo = _mm512_unpacklo_epi32(d, zero);
            let db_hi = _mm512_unpackhi_epi32(d, zero);
            let prod0 = _mm512_mul_epu32(q3_lo, db_lo);
            let prod1 = _mm512_mul_epu32(q3_hi, db_hi);

            let sum20 = _mm512_add_epi64(sum10, prod0);
            let sum21 = _mm512_add_epi64(sum11, prod1);

            let prod0_hi = _mm512_srli_epi64(sum20, 13);
            let prod1_hi = _mm512_srli_epi64(sum21, 13);
            let tmp_lo = transmute(_mm512_shuffle_ps(transmute(sum20), transmute(sum21), 0x88));
            let c0 = _mm512_madd52hi_epu64(zero, prod0_hi, p_barrett);
            let c1 = _mm512_madd52hi_epu64(zero, prod1_hi, p_barrett);
            let c = transmute(_mm512_shuffle_ps(transmute(c0), transmute(c1), 0x88));
            let c_times_big_q = _mm512_mullo_epi32(c, big_q);
            let q1 = _mm512_sub_epi32(tmp_lo, c_times_big_q);
            let cmp1 = _mm512_cmpge_epu32_mask(q1, big_q);
            let q2 = _mm512_mask_sub_epi32(q1, cmp1, q1, big_q);
            let q3 = _mm512_add_epi32(q2, a);
            let cmp3 = _mm512_cmpge_epu32_mask(q3, big_q);
            let q4 = _mm512_mask_sub_epi32(q3, cmp3, q3, big_q);
            *acc = transmute(q4);
        }
    }
}

#[inline(never)]
pub fn mul_accum_horiz_noreduce(db: &[u32x8], query: &u32x8, acc: &mut [u32]) {
    for (db_row, acc_row) in zip(db, acc) {
        let mut acc_tmp = 0;
        for i in 0..4 {
            acc_tmp = u32::wrapping_add(acc_tmp, u32::wrapping_mul(db_row[i], query[i]));
        }
        *acc_row = u32::wrapping_add(*acc_row, acc_tmp);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn mul_accum_horiz_noreduce_asm(db: &[u32x8], query: &u32x8, acc: &mut [u32]) {
    for (db_chunk, acc_chunk) in zip(db.array_chunks::<4>(), acc.array_chunks_mut::<4>()) {
        let mut p00 = [0; 4];
        let mut p01 = [0; 4];
        let mut p10 = [0; 4];
        let mut p11 = [0; 4];
        let mut p20 = [0; 4];
        let mut p21 = [0; 4];
        let mut p30 = [0; 4];
        let mut p31 = [0; 4];
        for i in 0..4 {
            p00[i] = u32::wrapping_mul(db_chunk[0][i], query[i]);
            p01[i] = u32::wrapping_mul(db_chunk[0][4 + i], query[4 + i]);
        }
        for i in 0..4 {
            p10[i] = u32::wrapping_mul(db_chunk[1][i], query[i]);
            p11[i] = u32::wrapping_mul(db_chunk[1][4 + i], query[4 + i]);
        }
        for i in 0..4 {
            p20[i] = u32::wrapping_mul(db_chunk[2][i], query[i]);
            p21[i] = u32::wrapping_mul(db_chunk[2][4 + i], query[4 + i]);
        }
        for i in 0..4 {
            p30[i] = u32::wrapping_mul(db_chunk[3][i], query[i]);
            p31[i] = u32::wrapping_mul(db_chunk[3][4 + i], query[4 + i]);
        }
        #[allow(clippy::missing_transmute_annotations)] // TODO: fix
        unsafe {
            acc_chunk[0] = acc_chunk[0]
                + vaddvq_u32(std::mem::transmute(p00))
                + vaddvq_u32(std::mem::transmute(p01));
            acc_chunk[1] = acc_chunk[1]
                + vaddvq_u32(std::mem::transmute(p10))
                + vaddvq_u32(std::mem::transmute(p11));
            acc_chunk[2] = acc_chunk[2]
                + vaddvq_u32(std::mem::transmute(p20))
                + vaddvq_u32(std::mem::transmute(p21));
            acc_chunk[3] = acc_chunk[3]
                + vaddvq_u32(std::mem::transmute(p30))
                + vaddvq_u32(std::mem::transmute(p31));
        }
    }
}

#[inline(never)]
pub fn mul_accum_horiz_division(db: &[u32x8], query: &u32x8, acc: &mut [u32]) {
    for (db_row, row_acc) in zip(db, acc) {
        for i in 0..u32x8::DIM {
            let tmp = u64::from(db_row[i]) * u64::from(query[i]) + u64::from(*row_acc);
            *row_acc = (tmp % u64::from(Q)) as u32;
        }
    }
}

#[inline(never)]
pub fn mul_accum_horiz_barrett(db: &[u32x8], query: &u32x8, acc: &mut [u32]) {
    let p_barrett = ((1u64 << 62) / Q as u64) as u32;
    for (db_row, row_acc) in zip(db, acc) {
        for i in 0..u32x8::DIM {
            let tmp = u64::from(db_row[i]) * u64::from(query[i]);
            let tmp_lo = (tmp & 0xffff_ffff) as u32;
            let tmp_hi = tmp >> 30;
            let c = ((tmp_hi * u64::from(p_barrett)) >> 32) as u32;
            let c_times_big_q = c.wrapping_mul(Q);
            let q = tmp_lo.wrapping_sub(c_times_big_q);

            let q1 = if q >= Q { q - Q } else { q };
            let q = *row_acc + q1;
            *row_acc = if q >= Q { q - Q } else { q };
        }
    }
}

#[inline(never)]
pub fn mul_accum_horiz_deferred(db: &[u32x8], query: &u32x8, acc: &mut [u32]) {
    let p_barrett = ((1u128 << 65) / Q as u128) as u64;
    for (db_row, row_acc) in zip(db, acc) {
        let tmp = u64::from(db_row[0]) * u64::from(query[0])
            + u64::from(db_row[1]) * u64::from(query[1])
            + u64::from(db_row[2]) * u64::from(query[2])
            + u64::from(db_row[3]) * u64::from(query[3]);
        let tmp_lo = (tmp & 0xffff_ffff) as u32;
        let tmp_hi = tmp >> 29;
        let c = ((u128::from(tmp_hi) * u128::from(p_barrett)) >> 36) as u32;
        let c_times_big_q = c.wrapping_mul(Q);
        let q = tmp_lo.wrapping_sub(c_times_big_q);
        let result1 = if q >= Q { q - Q } else { q };

        let tmp = u64::from(db_row[4]) * u64::from(query[4])
            + u64::from(db_row[5]) * u64::from(query[5])
            + u64::from(db_row[6]) * u64::from(query[6])
            + u64::from(db_row[7]) * u64::from(query[7]);
        let tmp_lo = (tmp & 0xffff_ffff) as u32;
        let tmp_hi = tmp >> 29;
        let c = ((u128::from(tmp_hi) * u128::from(p_barrett)) >> 36) as u32;
        let c_times_big_q = c.wrapping_mul(Q);
        let q = tmp_lo.wrapping_sub(c_times_big_q);
        let result2 = if q >= Q { q - Q } else { q };

        let q = result1 + result2;
        let result3 = if q >= Q { q - Q } else { q };

        let q = result3 + *row_acc;
        *row_acc = if q >= Q { q - Q } else { q };
    }
}

#[cfg(target_feature = "avx2")]
#[inline(never)]
pub fn mul_accum_horiz_deferred_avx2(db: &[u32x8], query: &u32x8, acc: &mut [u32]) {
    unsafe {
        let p_barrett = ((1u128 << 65) / Q as u128) as u64;
        let zero = _mm256_setzero_si256();
        let q = _mm256_load_si256(query as *const u32x8 as _);
        let q_lo = _mm256_unpacklo_epi32(q, zero);
        let q_hi = _mm256_unpackhi_epi32(q, zero);
        let carry_adjust = ((1u128 << 64) % (Q as u128)) as u64;
        for (db_row, row_acc) in zip(db, acc) {
            let d = _mm256_load_si256(db_row as *const u32x8 as _);
            let db_lo = _mm256_unpacklo_epi32(d, zero);
            let db_hi = _mm256_unpackhi_epi32(d, zero);
            let prod_lo = _mm256_mul_epu32(q_lo, db_lo);
            let prod_hi = _mm256_mul_epu32(q_hi, db_hi);
            let sum0a = _mm256_add_epi64(prod_lo, prod_hi);
            let sum0b = _mm256_set_m128i(_mm_setzero_si128(), _mm256_extractf128_si256(sum0a, 1));
            let sum1 = _mm256_add_epi64(sum0a, sum0b);
            let sum1a = _mm256_extract_epi64(sum1, 0) as u64;
            let sum1b = _mm256_extract_epi64(sum1, 1) as u64;
            let mut sum2 = 0;
            if _addcarry_u64(0, sum1a, sum1b, &mut sum2) != 0 {
                sum2 += carry_adjust;
            }
            let tmp_lo = (sum2 & 0xffff_ffff) as u32;
            let tmp_hi = sum2 >> 29;
            let c = ((u128::from(tmp_hi) * u128::from(p_barrett)) >> 36) as u32;
            let c_times_big_q = c.wrapping_mul(Q);
            let q = tmp_lo.wrapping_sub(c_times_big_q);

            let q1 = if q >= Q { q - Q } else { q };
            let q = *row_acc + q1;
            *row_acc = if q >= Q { q - Q } else { q };
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;

    fn test_mul_accum_vert<T: Simd<Item = u32>, F: Fn(&[T], &T, &mut [T])>(mul_accum_fn: F) {
        let mut rng = thread_rng();
        let db = [
            T::repeat(1),
            T::repeat(Q - 1),
            T::repeat_with(|| rng.gen::<u32>() % Q),
            T::repeat_with(|| rng.gen::<u32>() % Q),
        ];
        let mut query_vec = vec![1, Q - 1];
        query_vec.resize_with(T::DIM, || rng.gen::<u32>() % Q);
        let query = T::try_from(&query_vec).unwrap();

        let acc: [T; 4] = array::from_fn(|_| T::repeat_with(|| rng.gen::<u32>() % Q));
        let mut acc_out = acc.clone();

        let expected = zip(&db, &acc)
            .map(|(d, a)| {
                let mut result = T::default();
                for i in 0..T::DIM {
                    result[i] = ((u64::from(a[i]) + u64::from(d[i]) * u64::from(query[i]))
                        % u64::from(Q)) as u32;
                }
                result
            })
            .collect::<Vec<_>>();

        mul_accum_fn(&db, &query, &mut acc_out);

        assert_eq!(acc_out.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_mul_accum_vert_division() {
        test_mul_accum_vert(mul_accum_vert_division);
    }

    #[test]
    fn test_mul_accum_vert_barrett() {
        test_mul_accum_vert(mul_accum_vert_barrett);
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_mul_accum_vert_barrett_asm() {
        test_mul_accum_vert(mul_accum_vert_barrett_asm);
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn test_mul_accum_vert_barrett_avx512() {
        test_mul_accum_vert(mul_accum_vert_barrett_avx512);
    }

    /*
     * TODO: fix and restore these tests
     *
    fn test_mul_accum_hv<T: Simd<Item = u32>, F: Fn(&[T], &[T; 4], &mut [T])>(mul_accum_fn: F) {
        let mut rng = thread_rng();
        let db = [
            T::repeat(1),
            T::repeat(Q - 1),
            T::repeat_with(|| rng.gen::<u32>() % Q),
            T::repeat_with(|| rng.gen::<u32>() % Q),
        ];
        let mut query_vec = vec![1, Q - 1];
        query_vec.resize_with(T::DIM, || rng.gen::<u32>() % Q);
        let query = array::from_fn(|_| T::try_from(query_vec.clone()).unwrap());

        let acc: [T; 1] = array::from_fn(|_| T::repeat_with(|| rng.gen::<u32>() % Q));
        let mut acc_out = acc.clone();

        let expected = zip(db.array_chunks::<4>(), &acc).map(|(d_chunk, a)| {
            let mut result = a.clone();
            for j in 0..4 {
                for i in 0..T::DIM {
                    result[i] = ((u64::from(result[i]) + u64::from(d_chunk[j][i]) * u64::from(query[j][i])) % u64::from(Q)) as u32;
                }
            }
            result
        })
        .collect::<Vec<_>>();

        mul_accum_fn(&db, &query, &mut acc_out);

        assert_eq!(acc_out.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_mul_accum_hv_division() {
        test_mul_accum_hv(mul_accum_hv_division);
    }

    #[cfg(target_feature = "avx512ifma")]
    #[test]
    fn test_mul_accum_hv_barrett_31_avx512() {
        test_mul_accum_hv(mul_accum_hv_barrett_31_avx512);
    }
    */

    fn test_mul_accum_horiz<T: Simd<Item = u32>, F: Fn(&[T], &T, &mut [u32])>(mul_accum_fn: F) {
        let mut rng = thread_rng();
        let db = [
            T::repeat(1),
            T::repeat(Q - 1),
            T::repeat_with(|| rng.gen::<u32>() % Q),
        ];
        let mut query_vec = vec![1, Q - 1];
        query_vec.resize_with(T::DIM, || rng.gen::<u32>() % Q);
        let query = T::try_from(&query_vec).unwrap();

        let acc: [u32; 3] = array::from_fn(|_| rng.gen::<u32>() % Q);
        let mut acc_out = acc;

        let expected = zip(&db, &acc)
            .map(|(d, a)| {
                let mut result = *a;
                for i in 0..T::DIM {
                    result = ((u64::from(result) + u64::from(d[i]) * u64::from(query[i]))
                        % u64::from(Q)) as u32;
                }
                result
            })
            .collect::<Vec<_>>();

        mul_accum_fn(&db, &query, &mut acc_out);

        assert_eq!(acc_out.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_mul_accum_horiz_division() {
        test_mul_accum_horiz(mul_accum_horiz_division);
    }

    #[test]
    fn test_mul_accum_horiz_barrett() {
        test_mul_accum_horiz(mul_accum_horiz_barrett);
    }

    #[test]
    fn test_mul_accum_horiz_deferred() {
        test_mul_accum_horiz(mul_accum_horiz_deferred);
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_mul_accum_horiz_deferred_avx2() {
        test_mul_accum_horiz(mul_accum_horiz_deferred_avx2);
    }

    fn test_inner_prod<
        const W: usize,
        const N: usize,
        T: Simd<Item = u32>,
        F: for<'a> Fn(&'a [T], &'a [T], &'a mut [T]),
    >(
        inner_prod_fn: F,
    ) {
        let mut rng = thread_rng();
        let mut lhs = vec![T::repeat(1), T::repeat(Q - 1)];
        lhs.resize_with(N * W, || T::repeat_with(|| rng.gen::<u32>() % Q));
        let mut rhs = vec![T::repeat(1), T::repeat(Q - 1)];
        rhs.resize_with(N * W, || T::repeat_with(|| rng.gen::<u32>() % Q));

        let mut result: [T; W] = array::from_fn(|_| T::default());

        let expected = zip(lhs.chunks(W), rhs.chunks(W)).fold(
            array::from_fn::<_, W, _>(|_| T::default()),
            |mut acc, (l, r)| {
                for i in 0..W {
                    for j in 0..T::DIM {
                        acc[i][j] = ((u64::from(acc[i][j])
                            + u64::from(l[i][j]) * u64::from(r[i][j]))
                            % u64::from(Q)) as u32;
                    }
                }
                acc
            },
        );

        inner_prod_fn(&lhs, &rhs, &mut result);

        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_inner_prod_31() {
        test_inner_prod::<64, 2, _, _>(inner_prod_31::<2, 64, 64, 0, Q>);
    }
}
