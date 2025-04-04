use criterion::{measurement::Measurement, BatchSize, BenchmarkGroup, Criterion};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Standard};
use small_pir::{
    lwe::{
        BfvCiphertext, GswCiphertextNtt, Lwe1024Q30P8, Lwe1024Q31P8, Lwe1024Q32P8, Lwe2048Q56P8,
        LweParams,
    },
    poly::Poly,
};

fn extprod_bench<L: LweParams, const D: usize, M: Measurement, R: Rng>(
    group: &mut BenchmarkGroup<M>,
    rng: &mut R,
) where
    Standard: Distribution<GswCiphertextNtt<L, D>>
        + Distribution<BfvCiphertext<L>>
        + Distribution<Poly<L>>
        + Distribution<L::Ntt>,
{
    group.bench_function(format!("extprod_nc_l{D}"), |b| {
        b.iter_batched_ref(
            || {
                let gsw = rng.gen::<GswCiphertextNtt<L, D>>();
                let bfv = rng.gen::<BfvCiphertext<L>>();
                (gsw, bfv)
            },
            |(gsw, bfv)| (&*gsw) * (&*bfv),
            BatchSize::SmallInput,
        )
    });
}

fn lwe_bench<L: LweParams>(criterion: &mut Criterion)
where
    Standard: Distribution<L::Ntt>,
{
    let mut rng = thread_rng();

    // "nc" stands for NTT/Coefficient, i.e., the GSW ciphertext is in NTT
    // representation and the BFV ciphertext is in coefficient representation.
    // We also have a coefficient-coefficient external product implementation,
    // not included in this benchmark, and a special "one minus" implementation
    // that hopefully performs similarly to the N/C implementation, but probably
    // worth adding to the benchmark.
    let mut group = criterion.benchmark_group(L::name());
    extprod_bench::<L, 2, _, _>(&mut group, &mut rng);
    extprod_bench::<L, 3, _, _>(&mut group, &mut rng);
    extprod_bench::<L, 4, _, _>(&mut group, &mut rng);
    extprod_bench::<L, 11, _, _>(&mut group, &mut rng);
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    // For external product, the performance will not change based on the plaintext
    // modulus size.
    lwe_bench::<Lwe1024Q30P8>(&mut criterion);
    lwe_bench::<Lwe1024Q31P8>(&mut criterion);
    lwe_bench::<Lwe1024Q32P8>(&mut criterion);

    lwe_bench::<Lwe2048Q56P8>(&mut criterion);

    criterion.final_summary();
}
