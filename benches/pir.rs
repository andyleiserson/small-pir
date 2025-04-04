use criterion::{
    measurement::Measurement, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use rand::{thread_rng, Rng};
use small_pir::{
    lwe::{BfvCiphertextNtt, Lwe1024Q30P8, Lwe1024Q31P8, Lwe1024Q32P8, Lwe2048Q56P8, LweParams},
    pir::PirBackend,
};

const LEN: usize = 4096;

fn bfv<B, M>(g: &mut BenchmarkGroup<M>, pir_backend: &B, db_len: usize, query_len: usize)
where
    B: PirBackend,
    M: Measurement,
{
    g.bench_function(
        BenchmarkId::new("query", format!("lwe{}_ql{query_len}", B::name())),
        |b| {
            b.iter_batched_ref(
                || {
                    let mut rng = thread_rng();
                    let mut lwe = B::LweParams::gen_uniform(rng.clone());
                    let query_index = rng.gen_range(0..query_len);
                    let query = (0..query_len)
                        .map(|i| {
                            let v = if i == query_index { 1 } else { 0 };
                            lwe.encrypt_bfv(v).into()
                        })
                        .collect::<Vec<_>>();
                    let reduced_db =
                        vec![
                            BfvCiphertextNtt::from_raw(Default::default(), Default::default());
                            db_len / query_len
                        ];
                    (query, reduced_db)
                },
                |input| pir_backend.execute_bfv(&input.0, &mut input.1),
                BatchSize::PerIteration,
            )
        },
    );
}

fn test_bfv<L: LweParams>(criterion: &mut Criterion) {
    const QUERY_LEN: usize = 16;
    let mut rng = thread_rng();
    let mut data: Vec<u8> = vec![0u8; LEN * L::plaintext_bytes_per_ciphertext()];
    rng.fill(data.as_mut_slice());
    let pir = L::pir(&data, QUERY_LEN, 1, rng.gen());

    let mut group = criterion.benchmark_group("bfv");
    group.throughput(Throughput::Bytes(u64::try_from(data.len()).unwrap()));
    bfv(&mut group, pir.backend(), pir.len(), QUERY_LEN);
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    test_bfv::<Lwe1024Q30P8>(&mut criterion);
    test_bfv::<Lwe1024Q31P8>(&mut criterion);
    test_bfv::<Lwe1024Q32P8>(&mut criterion);

    test_bfv::<Lwe2048Q56P8>(&mut criterion);

    criterion.final_summary();
}
