#![allow(clippy::needless_range_loop)]

use std::{marker::PhantomData, slice};

#[cfg(not(target_feature = "avx512ifma"))]
use crate::math::{inner_prod_31, mul_accum_hv_barrett_31};
#[cfg(target_feature = "avx512ifma")]
use crate::math::{inner_prod_31_avx512, mul_accum_hv_barrett_31_avx512};
use crate::{
    lwe::{LweParams, LwePrivate},
    math::{u32x16, Simd},
    poly::{Ntt, Plan, Poly, ResidueNtt, ResidueNttBasis},
};

#[allow(clippy::len_without_is_empty)]
pub trait PirBackend {
    type LweParams: LweParams;

    fn name() -> &'static str {
        Self::LweParams::name()
    }

    fn new(src: &[Poly<Self::LweParams>], query_len: usize) -> Self;

    fn len(&self) -> usize;

    fn execute_bfv(
        &self,
        query: &[<Self::LweParams as LwePrivate>::Ntt],
        reduced_db: &mut [<Self::LweParams as LwePrivate>::Ntt],
    );

    fn inner_prod<const N: usize>(
        lhs: &[<Self::LweParams as LwePrivate>::Ntt; N],
        rhs: &[<Self::LweParams as LwePrivate>::Ntt; N],
        result: &mut <Self::LweParams as LwePrivate>::Ntt,
    );
}

pub struct Pir31<L: LweParams, const Q: u32> {
    db: Vec<u32x16>,
    query_len: usize,
    phantom_data: PhantomData<L>,
}

impl<L, const Q: u32> PirBackend for Pir31<L, Q>
where
    L: LweParams<Storage = u32, Array = [u32; 1024], Ntt = Ntt<L>>,
{
    type LweParams = L;

    fn new(src: &[Poly<L>], query_len: usize) -> Self {
        const WIDTH: usize = 1024 / u32x16::DIM;
        assert_eq!(
            query_len % 4,
            0,
            "query length should be a multiple of four"
        );

        let mut src = src.iter().cloned().map(L::Ntt::from).collect::<Vec<_>>();

        // Pad the database to a multiple of the query length
        for _ in src.len()..src.len().next_multiple_of(query_len) {
            src.push(L::Ntt::default());
        }

        // Viewing the database as `D = size` rows by `W = WIDTH` columns, and denoting
        // elements as `<row>.<col>`, to process a query with length `Q`, we want the
        // database in the following order:
        //
        // 0.0     1.0 2.0 3.0 (Q  ).0 (Q+1).0 (Q+2).0 (Q+3).0 ... (D-Q+3).0       \
        // 0.1     1.1 2.1 3.1 (Q  ).1 (Q+1).1 (Q+2).1 (Q+3).1 ... (D-Q+3).1        \  Large chunk
        // ...                                                                      /
        // 0.(W-1) ...                                         ... (D-Q+3).(W-1)   /
        // 4.0     5.0 6.0 7.0 (Q+4).0 (Q+5).0 (Q+6).0 (Q+7).0 ... (D-Q+7).0
        // ...
        // (Q-4).(W-1) ...                                     ... (D  -1).(W-1)
        //
        // One line of that listing is a "small chunk". A block of WIDTH rows is a
        // "large chunk".
        let size = src.len();
        let mut db = vec![u32x16::default(); size * WIDTH];
        let small_chunk_size = 4 * size / query_len;
        let large_chunk_size = small_chunk_size * WIDTH;
        for i in 0..size {
            for j in 0..WIDTH {
                let elt_index = 4 * (i / query_len) + i % 4;
                let large_chunk_index = (i % query_len) / 4;
                let transposed_index =
                    large_chunk_size * large_chunk_index + small_chunk_size * j + elt_index;
                let src = &src[i].as_raw_storage()[u32x16::DIM * j..u32x16::DIM * (j + 1)];
                db[transposed_index] = u32x16::try_from(src).unwrap();
            }
        }

        Self {
            db,
            query_len,
            phantom_data: PhantomData,
        }
    }

    fn len(&self) -> usize {
        const WIDTH: usize = 1024 / u32x16::DIM;
        self.db.len() / WIDTH
    }

    fn execute_bfv<'a>(&'a self, query: &'a [L::Ntt], reduced_db: &'a mut [L::Ntt]) {
        const WIDTH: usize = 1024 / u32x16::DIM;
        assert_eq!(query.len(), self.query_len, "unexpected query length");
        assert_eq!(
            self.db.len(),
            WIDTH * reduced_db.len() * query.len(),
            "unexpected reduced_db length"
        );
        let depth = query.len() / 4;
        let small_chunk_size = 4 * self.db.len() / query.len() / WIDTH;

        unsafe {
            // Safety:
            // * `Ntt` and `Poly` specify the required `align(64)` for `u32x16` values.
            // * The correct lifetime `'a` is transferred from the arguments to this function.
            // * Binaries built with `avx512ifma` should only be run on CPUs with the necessary support.
            let query_simd =
                slice::from_raw_parts::<'a, u32x16>(query.as_ptr() as _, query.len() * WIDTH);
            let reduced_simd = slice::from_raw_parts_mut::<'a, u32x16>(
                reduced_db.as_mut_ptr() as _,
                reduced_db.len() * WIDTH,
            );

            for i in 0..depth {
                for j in 0..WIDTH {
                    let data_chunk = &self.db[(WIDTH * i + j) * small_chunk_size
                        ..(WIDTH * i + j + 1) * small_chunk_size];
                    let query_chunk = [
                        &query_simd[(4 * i) * WIDTH + j],
                        &query_simd[(4 * i + 1) * WIDTH + j],
                        &query_simd[(4 * i + 2) * WIDTH + j],
                        &query_simd[(4 * i + 3) * WIDTH + j],
                    ];
                    let reduced_chunk = reduced_simd[j..].iter_mut().step_by(WIDTH);
                    #[cfg(not(target_feature = "avx512ifma"))]
                    mul_accum_hv_barrett_31::<_, Q>(data_chunk, query_chunk, reduced_chunk);
                    #[cfg(target_feature = "avx512ifma")]
                    mul_accum_hv_barrett_31_avx512::<_, Q>(data_chunk, query_chunk, reduced_chunk);
                }
            }
        }
    }

    fn inner_prod<'a, const N: usize>(
        lhs: &'a [Ntt<Self::LweParams>; N],
        rhs: &'a [Ntt<Self::LweParams>; N],
        result: &'a mut Ntt<Self::LweParams>,
    ) {
        const WIDTH: usize = 1024 / u32x16::DIM;
        unsafe {
            // Safety:
            // * `Ntt` and `Poly` specify the required `align(64)` for `u32x16` values.
            // * The correct lifetime `'a` is transferred from the arguments to this function.
            // * Binaries built with `avx512ifma` should only be run on CPUs with the necessary support.
            let lhs_simd = slice::from_raw_parts::<'a, u32x16>(lhs.as_ptr() as _, N * WIDTH);
            let rhs_simd = slice::from_raw_parts::<'a, u32x16>(rhs.as_ptr() as _, N * WIDTH);
            let result_simd =
                slice::from_raw_parts_mut::<'a, u32x16>(result as *mut _ as *mut u32x16, WIDTH);
            #[cfg(not(target_feature = "avx512ifma"))]
            inner_prod_31::<N, 64, 64, 0, Q>(lhs_simd, rhs_simd, result_simd);
            #[cfg(target_feature = "avx512ifma")]
            inner_prod_31_avx512::<N, 64, 64, 0, Q>(lhs_simd, rhs_simd, result_simd);
        }
    }
}

pub struct PirGeneric<L: LweParams<Ntt = Ntt<L>>> {
    db: Vec<Ntt<L>>,
    phantom_data: PhantomData<L>,
}

impl<L: LweParams<Ntt = Ntt<L>>> PirBackend for PirGeneric<L> {
    type LweParams = L;

    fn new(src: &[Poly<L>], _query_len: usize) -> Self {
        let db = src.iter().cloned().map(L::Ntt::from).collect::<Vec<_>>();

        Self {
            db,
            phantom_data: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.db.len()
    }

    fn execute_bfv(&self, query: &[L::Ntt], reduced_db: &mut [L::Ntt]) {
        let mut reduced_db_iter = reduced_db.iter_mut();
        let mut acc = reduced_db_iter.next().unwrap();
        let mut q_iter = query.iter();
        for db in &self.db {
            let q = if let Some(q) = q_iter.next() {
                q
            } else {
                acc = reduced_db_iter.next().unwrap();

                q_iter = query.iter();
                q_iter.next().unwrap()
            };
            L::plan().mul_accumulate(
                acc.as_raw_storage_mut(),
                db.as_raw_storage(),
                q.as_raw_storage(),
            );
        }
    }

    fn inner_prod<const N: usize>(
        lhs: &[Ntt<Self::LweParams>; N],
        rhs: &[Ntt<Self::LweParams>; N],
        result: &mut Ntt<Self::LweParams>,
    ) {
        for i in 0..N {
            // This clone is probably cheap relative to the overall cost of the
            // multiplication?
            *result += lhs[i].clone() * rhs[i].clone();
        }
    }
}

pub struct Pir62Crt<L: LweParams, B: ResidueNttBasis, const Q0: u32, const Q1: u32> {
    db_a: Vec<u32x16>,
    db_b: Vec<u32x16>,
    query_len: usize,
    phantom_data: PhantomData<(L, B)>,
}

impl<L, B, const Q0: u32, const Q1: u32> PirBackend for Pir62Crt<L, B, Q0, Q1>
where
    L: LweParams<Storage = u64, Array = [u64; 2048], Ntt = ResidueNtt<L, B>>,
    B: ResidueNttBasis,
{
    type LweParams = L;

    fn new(src: &[Poly<L>], query_len: usize) -> Self {
        const WIDTH: usize = 2048 / u32x16::DIM;
        assert_eq!(
            query_len % 4,
            0,
            "query length should be a multiple of four"
        );

        let mut src = src.iter().cloned().map(L::Ntt::from).collect::<Vec<_>>();

        // Pad the database to a multiple of the query length
        for _ in src.len()..src.len().next_multiple_of(query_len) {
            src.push(L::Ntt::default());
        }

        // Viewing the database as `D = size` rows by `W = WIDTH` columns, and denoting
        // elements as `<row>.<col>`, to process a query with length `Q`, we want the
        // database in the following order:
        //
        // 0.0     1.0 2.0 3.0 (Q  ).0 (Q+1).0 (Q+2).0 (Q+3).0 ... (D-Q+3).0       \
        // 0.1     1.1 2.1 3.1 (Q  ).1 (Q+1).1 (Q+2).1 (Q+3).1 ... (D-Q+3).1        \  Large chunk
        // ...                                                                      /
        // 0.(W-1) ...                                         ... (D-Q+3).(W-1)   /
        // 4.0     5.0 6.0 7.0 (Q+4).0 (Q+5).0 (Q+6).0 (Q+7).0 ... (D-Q+7).0
        // ...
        // (Q-4).(W-1) ...                                     ... (D  -1).(W-1)
        //
        // One line of that listing is a "small chunk". A block of WIDTH rows is a
        // "large chunk".
        let size = src.len();
        let mut db_a = vec![u32x16::default(); size * WIDTH];
        let mut db_b = vec![u32x16::default(); size * WIDTH];
        let small_chunk_size = 4 * size / query_len;
        let large_chunk_size = small_chunk_size * WIDTH;
        for i in 0..size {
            for j in 0..WIDTH {
                let elt_index = 4 * (i / query_len) + i % 4;
                let large_chunk_index = (i % query_len) / 4;
                let transposed_index =
                    large_chunk_size * large_chunk_index + small_chunk_size * j + elt_index;
                let src_pair = src[i].as_raw_storage();
                let src_a = &src_pair[0][u32x16::DIM * j..u32x16::DIM * (j + 1)];
                let src_b = &src_pair[1][u32x16::DIM * j..u32x16::DIM * (j + 1)];
                db_a[transposed_index] = u32x16::try_from(src_a).unwrap();
                db_b[transposed_index] = u32x16::try_from(src_b).unwrap();
            }
        }

        Self {
            db_a,
            db_b,
            query_len,
            phantom_data: PhantomData,
        }
    }

    fn len(&self) -> usize {
        const WIDTH: usize = 2048 / u32x16::DIM;
        self.db_a.len() / WIDTH
    }

    fn execute_bfv<'a>(&'a self, query: &'a [L::Ntt], reduced_db: &'a mut [L::Ntt]) {
        const WIDTH: usize = 2048 / u32x16::DIM;
        assert_eq!(query.len(), self.query_len, "unexpected query length");
        assert_eq!(
            self.db_a.len(),
            WIDTH * reduced_db.len() * query.len(),
            "unexpected reduced_db length"
        );
        let depth = query.len() / 4;
        let small_chunk_size = 4 * self.db_a.len() / query.len() / WIDTH;

        unsafe {
            // Safety:
            // * `Ntt` and `Poly` specify the required `align(64)` for `u32x16` values.
            // * The correct lifetime `'a` is transferred from the arguments to this function.
            // * Binaries built with `avx512ifma` should only be run on CPUs with the necessary support.
            let query_simd =
                slice::from_raw_parts::<'a, u32x16>(query.as_ptr() as _, 2 * query.len() * WIDTH);
            let reduced_simd = slice::from_raw_parts_mut::<'a, u32x16>(
                reduced_db.as_mut_ptr() as _,
                2 * reduced_db.len() * WIDTH,
            );

            for i in 0..depth {
                for j in 0..WIDTH {
                    let data_chunk_a = &self.db_a[(WIDTH * i + j) * small_chunk_size
                        ..(WIDTH * i + j + 1) * small_chunk_size];
                    let data_chunk_b = &self.db_b[(WIDTH * i + j) * small_chunk_size
                        ..(WIDTH * i + j + 1) * small_chunk_size];
                    let query_chunk_a = [
                        &query_simd[(8 * i) * WIDTH + j],
                        &query_simd[(8 * i + 2) * WIDTH + j],
                        &query_simd[(8 * i + 4) * WIDTH + j],
                        &query_simd[(8 * i + 6) * WIDTH + j],
                    ];
                    let query_chunk_b = [
                        &query_simd[(8 * i + 1) * WIDTH + j],
                        &query_simd[(8 * i + 3) * WIDTH + j],
                        &query_simd[(8 * i + 5) * WIDTH + j],
                        &query_simd[(8 * i + 7) * WIDTH + j],
                    ];

                    let reduced_chunk_a = reduced_simd[j..].iter_mut().step_by(2 * WIDTH);
                    #[cfg(not(target_feature = "avx512ifma"))]
                    mul_accum_hv_barrett_31::<_, Q0>(data_chunk_a, query_chunk_a, reduced_chunk_a);
                    #[cfg(target_feature = "avx512ifma")]
                    mul_accum_hv_barrett_31_avx512::<_, Q0>(
                        data_chunk_a,
                        query_chunk_a,
                        reduced_chunk_a,
                    );

                    let reduced_chunk_b = reduced_simd[WIDTH + j..].iter_mut().step_by(2 * WIDTH);
                    #[cfg(not(target_feature = "avx512ifma"))]
                    mul_accum_hv_barrett_31::<_, Q1>(data_chunk_b, query_chunk_b, reduced_chunk_b);
                    #[cfg(target_feature = "avx512ifma")]
                    mul_accum_hv_barrett_31_avx512::<_, Q1>(
                        data_chunk_b,
                        query_chunk_b,
                        reduced_chunk_b,
                    );
                }
            }
        }
    }

    fn inner_prod<'a, const N: usize>(
        lhs: &'a [L::Ntt; N],
        rhs: &'a [L::Ntt; N],
        result: &'a mut L::Ntt,
    ) {
        const WIDTH: usize = 2048 / u32x16::DIM;
        const STRIDE: usize = 2 * WIDTH;
        unsafe {
            // Safety:
            // * `Ntt` and `Poly` specify the required `align(64)` for `u32x16` values.
            // * The correct lifetime `'a` is transferred from the arguments to this function.
            // * Binaries built with `avx512ifma` should only be run on CPUs with the necessary support.
            let lhs_simd = slice::from_raw_parts::<'a, u32x16>(lhs.as_ptr() as _, N * STRIDE);
            let rhs_simd = slice::from_raw_parts::<'a, u32x16>(rhs.as_ptr() as _, N * STRIDE);
            let result_simd =
                slice::from_raw_parts_mut::<'a, u32x16>(result as *mut _ as *mut u32x16, STRIDE);
            #[cfg(not(target_feature = "avx512ifma"))]
            inner_prod_31::<N, WIDTH, STRIDE, 0, Q0>(lhs_simd, rhs_simd, result_simd);
            #[cfg(target_feature = "avx512ifma")]
            inner_prod_31_avx512::<N, WIDTH, STRIDE, 0, Q0>(lhs_simd, rhs_simd, result_simd);
            #[cfg(not(target_feature = "avx512ifma"))]
            inner_prod_31::<N, WIDTH, STRIDE, 128, Q1>(lhs_simd, rhs_simd, result_simd);
            #[cfg(target_feature = "avx512ifma")]
            inner_prod_31_avx512::<N, WIDTH, STRIDE, 128, Q1>(lhs_simd, rhs_simd, result_simd);
        }
    }
}
