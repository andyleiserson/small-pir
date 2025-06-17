use std::{array, iter::repeat_with, mem::size_of};

use criterion::{measurement::Measurement, BatchSize, BenchmarkGroup, Criterion, Throughput};
#[cfg(target_arch = "aarch64")]
use small_pir::math::mul_accum_horiz_noreduce_asm;
#[cfg(target_feature = "avx512ifma")]
use small_pir::math::mul_accum_hv_barrett_31_avx512;
#[cfg(target_feature = "avx512f")]
use small_pir::math::mul_accum_vert_barrett_avx512;
#[cfg(target_feature = "avx2")]
use small_pir::math::{
    mul_accum_horiz_deferred_avx2, mul_accum_vert_barrett_asm, mul_accum_vert_noreduce_asm,
};
use small_pir::{
    lwe::Q31,
    math::{
        accum_only, accum_only_unrolled, mul_accum_horiz_barrett, mul_accum_horiz_deferred,
        mul_accum_horiz_division, mul_accum_horiz_noreduce, mul_accum_hv_barrett_31,
        mul_accum_vert_barrett, mul_accum_vert_division, mul_accum_vert_noreduce, u32x8, Simd, Q,
    },
};
use rand::{thread_rng, Rng};

const N_SMALL: usize = 16 * 1024 / size_of::<u32>();
const N_MEDIUM: usize = 1024 * 1024 / size_of::<u32>();
const N_LARGE: usize = 16 * 1024 * 1024 / size_of::<u32>();
const N_HUGE: usize = 256 * 1024 * 1024 / size_of::<u32>();

// Sample counts for criterion
const S_SMALL: usize = 100;
const S_MEDIUM: usize = 50;
const S_LARGE: usize = 50;
const S_HUGE: usize = 10;

fn bench_vert<T, F, M>(criterion: &mut BenchmarkGroup<M>, name: &str, len: usize, f: F)
where
    T: Simd<Item = u32>,
    F: Fn(&mut (Vec<T>, T, Vec<T>)),
    M: Measurement,
{
    assert_eq!(
        len % (4 * T::DIM),
        0,
        "len should be a multiple of 4 vectors"
    );
    let adj_len = len / T::DIM;
    let f_ref = &f;
    criterion.bench_function(name, |b| {
        b.iter_batched_ref(
            || {
                let mut rng = thread_rng();
                let db = repeat_with(|| T::repeat_with(|| rng.gen::<u32>() % Q))
                    .take(adj_len)
                    .collect();
                let query = T::repeat_with(|| rng.gen::<u32>() % Q);
                let acc = repeat_with(|| T::repeat_with(|| rng.gen::<u32>() % Q))
                    .take(adj_len)
                    .collect();
                (db, query, acc)
            },
            f_ref,
            BatchSize::PerIteration,
        )
    });
}

fn bench_horiz<T, F, M>(criterion: &mut BenchmarkGroup<M>, name: &str, len: usize, f: F)
where
    T: Simd<Item = u32>,
    F: Fn(&mut (Vec<T>, T, Vec<u32>)),
    M: Measurement,
{
    assert_eq!(
        len % (4 * T::DIM),
        0,
        "len should be a multiple of 4 vectors"
    );
    let adj_len = len / T::DIM;
    let f_ref = &f;
    criterion.bench_function(name, |b| {
        b.iter_batched_ref(
            || {
                let mut rng = thread_rng();
                let db = repeat_with(|| T::repeat_with(|| rng.gen::<u32>() % Q))
                    .take(adj_len)
                    .collect();
                let query = T::repeat_with(|| rng.gen::<u32>() % Q);
                let acc = repeat_with(|| rng.gen::<u32>() % Q).take(adj_len).collect();
                (db, query, acc)
            },
            f_ref,
            BatchSize::PerIteration,
        )
    });
}

fn bench_hv<T, F, M>(criterion: &mut BenchmarkGroup<M>, name: &str, len: usize, f: F)
where
    T: Simd<Item = u32>,
    F: Fn(&mut (Vec<T>, [T; 4], Vec<T>)),
    M: Measurement,
{
    assert_eq!(
        len % (4 * T::DIM),
        0,
        "len should be a multiple of 4 vectors"
    );
    let adj_len = len / T::DIM;
    let f_ref = &f;
    criterion.bench_function(name, |b| {
        b.iter_batched_ref(
            || {
                let mut rng = thread_rng();
                let db = repeat_with(|| T::repeat_with(|| rng.gen::<u32>() % Q))
                    .take(adj_len)
                    .collect();
                let query = array::from_fn(|_| T::repeat_with(|| rng.gen::<u32>() % Q));
                let acc = repeat_with(|| T::repeat_with(|| rng.gen::<u32>() % Q))
                    .take(adj_len / 4)
                    .collect();
                (db, query, acc)
            },
            f_ref,
            BatchSize::PerIteration,
        )
    });
}

fn vert_group(criterion: &mut Criterion, name: &str, len: usize, sample_size: usize) {
    let mut group = criterion.benchmark_group(name);
    group.throughput(Throughput::Bytes(
        u64::try_from(len * size_of::<u32>()).unwrap(),
    ));
    group.sample_size(sample_size);
    bench_vert::<_, _, _>(&mut group, "accum_only", len, |(d, q, a)| {
        accum_only(d, q, a)
    });
    bench_vert::<u32x8, _, _>(&mut group, "accum_only_unrolled", len, |(d, q, a)| {
        accum_only_unrolled(d, q, a)
    });
    bench_vert::<_, _, _>(&mut group, "noreduce", len, |(d, q, a)| {
        mul_accum_vert_noreduce(d, q, a)
    });
    #[cfg(target_feature = "avx2")]
    bench_vert::<_, _, _>(&mut group, "noreduce_asm", len, |(d, q, a)| {
        mul_accum_vert_noreduce_asm(d, q, a)
    });
    bench_vert::<_, _, _>(&mut group, "division", len, |(d, q, a)| {
        mul_accum_vert_division(d, q, a)
    });
    bench_vert::<_, _, _>(&mut group, "barrett", len, |(d, q, a)| {
        mul_accum_vert_barrett(d, q, a)
    });
    #[cfg(target_feature = "avx2")]
    bench_vert::<_, _, _>(&mut group, "barrett_asm", len, |(d, q, a)| {
        mul_accum_vert_barrett_asm(d, q, a)
    });
    #[cfg(target_feature = "avx512f")]
    bench_vert::<_, _, _>(&mut group, "barrett_avx512", len, |(d, q, a)| {
        mul_accum_vert_barrett_avx512(d, q, a)
    });
    //bench_mul_accum(&mut group, "deferred", |(d, q, a)| mul_accum_deferred(d, q, a));
    group.finish();
}

fn horiz_group(criterion: &mut Criterion, name: &str, len: usize, sample_size: usize) {
    let mut group = criterion.benchmark_group(name);
    group.throughput(Throughput::Bytes(
        u64::try_from(len * size_of::<u32>()).unwrap(),
    ));
    group.sample_size(sample_size);
    bench_horiz::<_, _, _>(&mut group, "noreduce", len, |(d, q, a)| {
        mul_accum_horiz_noreduce(d, q, a)
    });
    #[cfg(target_arch = "aarch64")]
    bench_horiz::<_, _, _>(&mut group, "noreduce_asm", len, |(d, q, a)| {
        mul_accum_horiz_noreduce_asm(d, q, a)
    });
    bench_horiz::<_, _, _>(&mut group, "division", len, |(d, q, a)| {
        mul_accum_horiz_division(d, q, a)
    });
    bench_horiz::<_, _, _>(&mut group, "barrett", len, |(d, q, a)| {
        mul_accum_horiz_barrett(d, q, a)
    });
    bench_horiz::<_, _, _>(&mut group, "deferred", len, |(d, q, a)| {
        mul_accum_horiz_deferred(d, q, a)
    });
    #[cfg(target_feature = "avx2")]
    bench_horiz::<_, _, _>(&mut group, "deferred_avx2", len, |(d, q, a)| {
        mul_accum_horiz_deferred_avx2(d, q, a)
    });
    group.finish();
}

fn hv_group(criterion: &mut Criterion, name: &str, len: usize, sample_size: usize) {
    let mut group = criterion.benchmark_group(name);
    group.throughput(Throughput::Bytes(
        u64::try_from(len * size_of::<u32>()).unwrap(),
    ));
    group.sample_size(sample_size);
    bench_hv::<_, _, _>(&mut group, "barrett", len, |(d, q, a)| {
        mul_accum_hv_barrett_31::<_, Q31>(d, q.each_ref(), a.iter_mut())
    });
    #[cfg(target_feature = "avx512ifma")]
    bench_hv::<_, _, _>(&mut group, "barrett_avx512", len, |(d, q, a)| {
        mul_accum_hv_barrett_31_avx512::<_, Q31>(d, q, a)
    });
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    vert_group(&mut criterion, "vert_small", N_SMALL, S_SMALL);
    horiz_group(&mut criterion, "horiz_small", N_SMALL, S_SMALL);
    hv_group(&mut criterion, "hv_small", N_SMALL, S_SMALL);
    vert_group(&mut criterion, "vert_medium", N_MEDIUM, S_MEDIUM);
    horiz_group(&mut criterion, "horiz_medium", N_MEDIUM, S_MEDIUM);
    hv_group(&mut criterion, "hv_medium", N_MEDIUM, S_MEDIUM);
    vert_group(&mut criterion, "vert_large", N_LARGE, S_LARGE);
    horiz_group(&mut criterion, "horiz_large", N_LARGE, S_LARGE);
    hv_group(&mut criterion, "hv_large", N_LARGE, S_LARGE);
    vert_group(&mut criterion, "vert_huge", N_HUGE, S_HUGE);
    horiz_group(&mut criterion, "horiz_huge", N_HUGE, S_HUGE);
    hv_group(&mut criterion, "hv_huge", N_HUGE, S_HUGE);

    criterion.final_summary();
}
