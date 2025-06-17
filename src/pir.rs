pub(crate) mod backend;
mod query;

use std::{
    iter::{once, repeat_with, zip},
    time::Duration,
};

pub use backend::PirBackend;
use bitvec::{
    field::BitField,
    order::{BitOrder, Lsb0},
    slice::BitSlice,
    store::BitStore,
    vec::BitVec,
    view::BitView,
};
pub use query::{DecodedQuery, PackedQuery, QueryBfv, SimpleQuery};
use rand::{thread_rng, CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Standard};

use crate::{
    field::Field,
    lwe::{BfvCiphertext, Compressor, Lwe, LweParams, LwePrivate},
    pir::query::Query,
    poly::Poly,
    timer::Timer,
    Decompose, OneMinus,
};

// Debug controls
const PRINT_NOISE: bool = false;

trait PaddedChunksExact {
    type Iterator;

    fn padded_chunks_exact(self, chunk_size: usize) -> Self::Iterator;
}

impl<'a, T> PaddedChunksExact for &'a [T] {
    type Iterator = SlicePadLastChunk<'a, T>;

    fn padded_chunks_exact(self, chunk_size: usize) -> SlicePadLastChunk<'a, T> {
        SlicePadLastChunk {
            inner: Some(self.chunks_exact(chunk_size)),
            chunk_size,
        }
    }
}

// Unused and untested
struct SlicePadLastChunk<'a, T> {
    inner: Option<std::slice::ChunksExact<'a, T>>,
    chunk_size: usize,
}

impl<'a, T: Copy + Default + 'a> Iterator for SlicePadLastChunk<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(inner) = &mut self.inner else {
            return None;
        };
        match inner.next() {
            Some(item) => Some(item.to_owned()),
            None => {
                let remainder = inner.remainder();
                let result = (!remainder.is_empty()).then(|| {
                    let mut last = Vec::with_capacity(self.chunk_size);
                    last.extend(remainder.iter());
                    last.resize(self.chunk_size, T::default());
                    last
                });
                self.inner = None;
                result
            }
        }
    }
}

impl<'a, T: BitStore, O: BitOrder> PaddedChunksExact for &'a BitSlice<T, O> {
    type Iterator = BitSlicePadLastChunk<'a, T, O>;

    fn padded_chunks_exact(self, chunk_size: usize) -> BitSlicePadLastChunk<'a, T, O> {
        BitSlicePadLastChunk::<T, O> {
            inner: Some(self.chunks_exact(chunk_size)),
            chunk_size,
        }
    }
}

struct BitSlicePadLastChunk<'a, T: BitStore, O: BitOrder> {
    inner: Option<bitvec::slice::ChunksExact<'a, T, O>>,
    chunk_size: usize,
}

impl<'a, T, O> Iterator for BitSlicePadLastChunk<'a, T, O>
where
    T: BitStore + Clone + Default + 'a,
    O: BitOrder,
{
    type Item = BitVec<T, O>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(inner) = &mut self.inner else {
            return None;
        };
        match inner.next() {
            Some(item) => Some(item.to_owned()),
            None => {
                let remainder = inner.remainder();
                let result = (!remainder.is_empty()).then(|| {
                    let mut last = BitVec::with_capacity(self.chunk_size);
                    last.extend(remainder.iter());
                    last.resize(self.chunk_size, false);
                    last
                });
                self.inner = None;
                result
            }
        }
    }
}

#[allow(dead_code)]
fn padded_chunks_exact<I: Iterator, F: Fn() -> I::Item>(
    iter: I,
    chunk_size: usize,
    pad_fn: F,
) -> IteratorPadLastChunk<I, F> {
    IteratorPadLastChunk {
        inner: Some(iter),
        chunk_size,
        pad_fn,
    }
}

struct IteratorPadLastChunk<I, F> {
    inner: Option<I>,
    chunk_size: usize,
    pad_fn: F,
}

impl<I, F> Iterator for IteratorPadLastChunk<I, F>
where
    I: Iterator,
    I::Item: Clone + Default,
    F: Fn() -> I::Item,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(inner) = &mut self.inner else {
            return None;
        };
        match inner.next() {
            Some(item) => {
                let mut count = 1;
                let result = Some(
                    once(item)
                        .chain(inner.inspect(|_| count += 1))
                        .chain(repeat_with(|| (self.pad_fn)()))
                        .take(self.chunk_size)
                        .collect::<Vec<_>>(),
                );
                if count < self.chunk_size {
                    self.inner = None;
                }
                result
            }
            None => {
                self.inner = None;
                None
            }
        }
    }
}

pub type Seed = <ChaCha12Rng as SeedableRng>::Seed;

pub struct Pir<B: PirBackend, const ELL_GSW: usize, const ELL_KS: usize> {
    bfv_query_len: usize,
    database: Vec<BitVec<u8, Lsb0>>,
    backend: B,
    compressor: Compressor,
    #[allow(dead_code)]
    bfv_c0_db_ntt: Vec<<B::LweParams as LwePrivate>::Ntt>,
    bfv_c0_decompose: Vec<[<B::LweParams as LwePrivate>::Ntt; ELL_GSW]>,
}

#[derive(Default)]
pub struct QueryTimings {
    pub unpack_duration: Duration,
    pub bfv_query_duration: Duration,
    pub gsw_query_duration: Duration,
    pub e2e_query_duration: Duration,
}

fn precompute_query_ciphertexts<L: LweParams, const ELL_KS: usize>(
    compressor: &Compressor,
    pack_dim: usize,
    n_pack: usize,
) -> Vec<Poly<L>>
where
    Standard: Distribution<L::Ntt>,
{
    if pack_dim == 1 {
        // The easy case
        return (0..n_pack)
            .map(|i| compressor.get::<Poly<L>>(u64::try_from(i / pack_dim).unwrap()))
            .collect();
    }

    // This doesn't affect the returned values
    let mut lwe = L::gen_noise(thread_rng());

    // Need a new one, so we don't consume indices from the original
    let compressor = compressor.clone();

    let packed_bits = vec![Field::ZERO; pack_dim * n_pack];
    let query = PackedQuery::<_, ELL_KS>::encode(&mut lwe, &compressor, &packed_bits, pack_dim);
    query
        .decode(&compressor)
        .into_iter()
        .map(|c| c.into_raw().0)
        .collect()
}

pub struct PirResult<L: LweParams> {
    pub answer: BfvCiphertext<L>,
    pub timings: QueryTimings,
}

impl<B: PirBackend, const ELL_GSW: usize, const ELL_KS: usize> Pir<B, ELL_GSW, ELL_KS>
where
    Standard: Distribution<<B::LweParams as LwePrivate>::Ntt>,
{
    pub fn new(
        db_bytes: &[u8],
        bfv_query_len: usize,
        pack_dim: usize,
        n_pack: usize,
        seed: Seed,
    ) -> Self {
        assert!(bfv_query_len.is_power_of_two());

        let compressor = Compressor::new(seed);
        let _timer = Timer::new("Preprocess database");
        let size = 8 * db_bytes
            .len()
            .div_ceil(B::LweParams::P_BITS * B::LweParams::DEGREE);
        let mut database = Vec::with_capacity(size);
        let mut database_orig = Vec::with_capacity(size);
        let db_bits = db_bytes.view_bits::<Lsb0>();
        for element in db_bits.padded_chunks_exact(B::LweParams::P_BITS * B::LweParams::DEGREE) {
            database_orig.push(element.clone());
            let poly: Poly<B::LweParams> = <B::LweParams as LwePrivate>::Array::try_from(
                element
                    .padded_chunks_exact(B::LweParams::P_BITS)
                    .map(|bits| bits.load_le::<<B::LweParams as LwePrivate>::Storage>())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .into();
            database.push(poly.clone());
        }

        let backend = B::new(&database, bfv_query_len);

        let mut bfv_c0_db_ntt =
            vec![<B::LweParams as LwePrivate>::Ntt::default(); size.div_ceil(bfv_query_len)];
        let mut query = precompute_query_ciphertexts::<_, ELL_KS>(&compressor, pack_dim, n_pack);
        assert!(query.len() >= bfv_query_len);
        query.truncate(bfv_query_len); // discard GSW syn, if any
        let query_c0 = query
            .into_iter()
            .map(<B::LweParams as LwePrivate>::Ntt::from)
            .collect::<Vec<_>>();
        backend.execute_bfv(&query_c0, &mut bfv_c0_db_ntt);

        let bfv_c0_decompose = bfv_c0_db_ntt
            .iter()
            .map(|c0| {
                <_ as Into<Poly<_>>>::into(c0.clone())
                    .decompose()
                    .map(<B::LweParams as LwePrivate>::Ntt::from)
            })
            .collect();

        if !self::PRINT_NOISE {
            bfv_c0_db_ntt.truncate(0);
        }

        Self {
            bfv_query_len,
            database: database_orig,
            backend,
            compressor,
            bfv_c0_db_ntt,
            bfv_c0_decompose,
        }
    }
}

impl<B: PirBackend, const ELL_GSW: usize, const ELL_KS: usize> Pir<B, ELL_GSW, ELL_KS> {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.database.len()
    }

    // This is used by benchmarks to access individual routines.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn compressor(&self) -> &Compressor {
        &self.compressor
    }

    #[allow(dead_code)]
    fn maybe_invert_plaintext(index: usize, plaintext: &Poly<B::LweParams>) -> Poly<B::LweParams> {
        if index % 2 == 0 {
            return plaintext.clone();
        }

        B::LweParams::array_from_fn(|i| {
            let val = plaintext.as_raw_storage()[i];
            if val != <B::LweParams as LwePrivate>::Storage::default() {
                B::LweParams::P_MODULUS - val
            } else {
                val
            }
        })
        .into()
    }

    pub fn query<R, LPrime>(
        &self,
        query: &Query<B::LweParams, ELL_GSW>,
        lwe: &mut Lwe<B::LweParams, R>,
        query_index: usize,
    ) -> PirResult<LPrime>
    where
        R: Rng + CryptoRng + Clone,
        LPrime: LweParams,
        LPrime::Storage: TryFrom<<B::LweParams as LwePrivate>::OpStorage>,
        <B::LweParams as LwePrivate>::OpStorage: TryFrom<LPrime::Storage>,
    {
        // TODO: check query structure vs. database size

        let mut timer = Timer::new("Unpack query");
        let e2e_timer = timer.copy("Total time for server to answer query");
        let unpack_timer = timer.copy("Unpack and NTT of query");

        let DecodedQuery {
            bfv_ciphertexts,
            gsw_ciphertexts,
        } = query.decode(self.compressor());

        assert_eq!(bfv_ciphertexts.len(), self.bfv_query_len);

        if PRINT_NOISE {
            for (i, q) in bfv_ciphertexts.iter().enumerate() {
                let pt_pre = if i == query_index % self.bfv_query_len {
                    Poly::from_scalar(Field::ONE)
                } else {
                    Poly::from_scalar(Field::ZERO)
                };
                let pt = pt_pre * B::LweParams::field(B::LweParams::floor_q_div_p());
                let norm = (lwe.raw_decrypt_bfv(q.clone()) - pt).inf_norm();
                println!(
                    "Unpacked query[{i}]: log₂|ε| = {:.1}",
                    ((<_ as TryInto<u64>>::try_into(norm).ok().unwrap() as f32).log2())
                );
            }
        }

        timer.report_and_reset("NTT of query");

        let bfv_ntt = bfv_ciphertexts
            .into_iter()
            .map(|c| <B::LweParams as LwePrivate>::Ntt::from(c.into_raw().1))
            .collect::<Vec<_>>();

        let unpack_duration = unpack_timer.end();

        timer.report_and_reset("BFV query output buffer allocation");
        let mut bfv_c1_ntt =
            vec![<B::LweParams as LwePrivate>::Ntt::default(); self.len().div_ceil(bfv_ntt.len())];
        timer.report_and_reset("BFV query");
        self.backend.execute_bfv(&bfv_ntt, &mut bfv_c1_ntt);
        let bfv_query_duration = timer.report_and_reset("BFV query inverse NTT");

        /*
        if PRINT_NOISE {
            let mut reduced_db_norm = <B::LweParams as LwePrivate>::Storage::default();
            for (i, (c0, c1)) in zip(&self.bfv_c0_db_ntt, &bfv_c1_ntt).enumerate() {
                let index = i * bfv_ntt.len() + (query_index & (bfv_ntt.len() - 1));
                if let Some(d) = self.database.get(index) {
                    let pt = Self::maybe_invert_plaintext(index, d) * B::LweParams::field(B::LweParams::floor_q_div_p());
                    let ct = BfvCiphertext::from(
                        BfvCiphertextNtt::from_raw(c0.clone(), c1.clone())
                    );
                    reduced_db_norm = reduced_db_norm.max((lwe.raw_decrypt_bfv(ct) - pt).inf_norm());
                }
            }
            println!("BFV query output: log₂|ε| = {:.1}", ((<_ as TryInto<u64>>::try_into(reduced_db_norm).ok().unwrap() as f32).log2()));
        }
        */

        let first_gsw_input = zip(&self.bfv_c0_decompose, bfv_c1_ntt)
            .map(|(c0d, c1)| {
                (
                    c0d,
                    <_ as Into<Poly<_>>>::into(c1)
                        .decompose()
                        .map(<B::LweParams as LwePrivate>::Ntt::from),
                )
            })
            .collect::<Vec<_>>();

        timer.report_and_reset("GSW query");

        let mut gsw_index = 0;

        let mut reduced_db = Vec::with_capacity(first_gsw_input.len().div_ceil(2));
        for chunk in first_gsw_input.chunks(2) {
            if let Ok([d0, d1]) = <&[_; 2]>::try_from(chunk) {
                // TODO: implement Add<BfvCiphertextNtt> and save half the NTTs
                let d1_prod = BfvCiphertext::from(&gsw_ciphertexts[gsw_index] * (d1.0, &d1.1));
                let d0_prod =
                    BfvCiphertext::from(OneMinus(&gsw_ciphertexts[gsw_index]) * (d0.0, &d0.1));
                let d = d1_prod + d0_prod;
                reduced_db.push(d);
            } else {
                // Odd final element
                let d = BfvCiphertext::from(
                    OneMinus(&gsw_ciphertexts[gsw_index]) * (chunk[0].0, &chunk[0].1),
                );
                reduced_db.push(d);
            }
        }

        gsw_index += 1;

        while reduced_db.len() > 1 {
            let mut reduced_db_next = Vec::with_capacity(reduced_db.len().div_ceil(2));
            for chunk in reduced_db.chunks(2) {
                if let Ok([d0, d1]) = <&[BfvCiphertext<_>; 2]>::try_from(chunk) {
                    // TODO: implement Add<BfvCiphertextNtt> and save half the NTTs
                    let d1_prod = BfvCiphertext::from(&gsw_ciphertexts[gsw_index] * d1);
                    let d0_prod = BfvCiphertext::from(OneMinus(&gsw_ciphertexts[gsw_index]) * d0);
                    let d = d1_prod + d0_prod;
                    reduced_db_next.push(d);
                } else {
                    // Odd final element
                    let d = BfvCiphertext::from(OneMinus(&gsw_ciphertexts[gsw_index]) * &chunk[0]);
                    reduced_db_next.push(d);
                }
            }

            reduced_db = reduced_db_next;
            gsw_index += 1;

            /*
            if PRINT_NOISE {
                let mut reduced_db_norm = <B::LweParams as LwePrivate>::Storage::default();
                for (i, v) in reduced_db.iter().enumerate() {
                    let gsw_query_index = (query_index / bfv_ntt.len()) & ((1 << gsw_index) - 1);
                    let db_index = i * (bfv_ntt.len() << gsw_index) + gsw_query_index * bfv_ntt.len() + (query_index & (bfv_ntt.len() - 1));
                    if let Some(d) = self.database.get(db_index) {
                        let pt = Self::maybe_invert_plaintext(db_index, d) * B::LweParams::field(B::LweParams::floor_q_div_p());
                        let norm = (lwe.raw_decrypt_bfv(v.clone()) - pt).inf_norm();
                        reduced_db_norm = reduced_db_norm.max(norm);
                    }
                }
                println!(
                    "GSW query {} output: log₂|ε| = {:.1}",
                    gsw_index - 1,
                    ((<_ as TryInto<u64>>::try_into(reduced_db_norm).ok().unwrap() as f32).log2()),
                );
            }
            */
        }

        let gsw_query_duration = timer.end();
        let e2e_query_duration = e2e_timer.end();

        let answer = reduced_db.pop().unwrap();
        let modswitch_answer = answer.switch_modulus();

        PirResult {
            answer: modswitch_answer,
            timings: QueryTimings {
                unpack_duration,
                bfv_query_duration,
                gsw_query_duration,
                e2e_query_duration,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use rand_distr::{Distribution, Standard};

    use super::*;
    use crate::{
        field::Field,
        lwe::{
            GswCiphertextCompressed, Lwe1024Q14P1, Lwe1024Q20P6, Lwe1024Q30P1, Lwe1024Q30P2,
            Lwe1024Q30P6, Lwe1024Q31P2, Lwe1024Q31P4, Lwe1024Q31P7, Lwe2048Q21P1, Lwe2048Q21P4,
            Lwe2048Q21P6, Lwe2048Q21P8, Lwe2048Q25P12, Lwe2048Q56P1, Lwe2048Q56P12, Lwe2048Q56P4,
            Lwe2048Q56P6, Lwe2048Q56P8,
        },
        pir::query::Query,
        Decompose,
    };

    fn pir<L, const ELL_GSW: usize, const ELL_KS: usize, LPrime>(
        total_db_bytes: usize,
        slices: usize,
        pack_dim: usize,
        n_pack: usize,
        synth_gsw_bits: Option<u32>,
    ) where
        L: LweParams,
        LPrime: LweParams,
        LPrime::Storage: TryFrom<L::Storage> + TryFrom<L::OpStorage>,
        L::OpStorage: TryFrom<LPrime::Storage>,
        Standard: Distribution<L::Ntt>,
    {
        let mut rng = thread_rng();

        let db_bytes = total_db_bytes / slices;
        let len = 8 * db_bytes.div_ceil(L::P_BITS * L::DEGREE);

        // TODO: n_pack is ignored here when gsw_synth is active, but it is used below in computing n_packed_bits
        let (bfv_bits, gsw_bits, gsw_synth) = if let Some(gsw_bits) = synth_gsw_bits {
            (len.next_power_of_two().ilog2() - gsw_bits, gsw_bits, true)
        } else {
            let bfv_bits = (pack_dim * n_pack).ilog2();
            (bfv_bits, len.next_power_of_two().ilog2() - bfv_bits, false)
        };

        println!("{bfv_bits} BFV bits, {gsw_bits} GSW bits");

        let bfv_query_len = 1 << bfv_bits;

        let mut databases: Vec<Vec<u8>> = vec![vec![0u8; db_bytes]; slices];
        println!("Q modulus bits: {}", L::Q_BITS);
        println!("P modulus bits: {}", L::P_BITS);
        println!("Query encoding: ell_ks = {ELL_KS}, ell_gsw = {ELL_GSW}, {n_pack} ciphertexts each containing {pack_dim} packed indices");

        let timer = Timer::new("RNG");
        for db in databases.iter_mut() {
            rng.fill(db.as_mut_slice());
        }
        timer.end();

        let seed = rng.gen::<Seed>();
        let pirs = databases
            .iter()
            .map(|db| L::pir::<ELL_GSW, ELL_KS>(db, bfv_query_len, pack_dim, n_pack, seed))
            .collect::<Vec<_>>();
        let encoded_db_bytes = len * size_of::<L::Array>();
        let bfv_output_bytes = encoded_db_bytes.div_ceil(bfv_query_len);
        println!(
            "Database is {} MiB before encoding, {} MiB after.",
            ((slices * databases[0].len()) as f32 / 104_857.6).round() / 10.0,
            ((slices * encoded_db_bytes) as f32 / 104_857.6).round() / 10.0,
        );
        println!(
            "BFV query output is {} MiB.",
            ((slices * bfv_output_bytes) as f32 / 104_857.6).round() / 10.0,
        );

        let query_index = rng.gen_range(0..len);

        let mut timer = Timer::new("LWE setup");
        let mut lwe = L::gen_noise(rng.clone());

        timer.report_and_reset("Construct query");

        // If we're doing gsw synthesis, we have to collect the bits before we pack the query.
        // But if we're not doing gsw synthesis, we can't encrypt the bits until after we pack
        // the query, because the query has to use the first ciphertext indices for the PRG.
        let mut gsw_components: Vec<Field<L>> = Vec::new();
        let mut gsw_bit_vec = Vec::new();
        let mut remaining_query_index = query_index >> bfv_bits;
        for _ in 0..gsw_bits {
            let gsw_bit = remaining_query_index & 1 == 1;
            if gsw_synth {
                let gsw_val = if gsw_bit {
                    <Field<L> as Decompose<ELL_GSW>>::basis().into_iter()
                } else {
                    [Field::ZERO; ELL_GSW].into_iter()
                };
                gsw_components.extend(
                    gsw_val.map(|b| L::field(b.to_raw() / L::Storage::from(pack_dim as u32))),
                );
            } else {
                gsw_bit_vec.push(gsw_bit);
            }
            remaining_query_index >>= 1;
        }

        let n_gsw_components = if gsw_synth {
            usize::try_from(gsw_bits).unwrap() * ELL_GSW
        } else {
            0
        };
        let n_packed_bits = pack_dim * n_pack;
        let needed_bits = (1 << bfv_bits) + n_gsw_components;
        assert!(
             needed_bits <= n_packed_bits,
            "n_pack of {n_pack} is not large enough. pack_dim * n_pack must be at least {needed_bits}."
        );
        let mut packed_bits = Vec::with_capacity(n_packed_bits);
        packed_bits.extend((0..usize::try_from(1 << bfv_bits).unwrap()).map(|i| {
            if i == query_index & ((1 << bfv_bits) - 1) {
                Field::new(L::floor_q_div_p() / L::Storage::from(pack_dim as u32))
            } else {
                Field::ZERO
            }
        }));
        packed_bits.resize(n_packed_bits - n_gsw_components, Field::ZERO);
        packed_bits.append(&mut gsw_components);

        let packed: Box<dyn QueryBfv<LweParams = L>> = if pack_dim == 1 {
            Box::new(SimpleQuery::encode(
                &mut lwe,
                pirs[0].compressor(),
                &packed_bits,
            ))
        } else {
            Box::new(PackedQuery::<_, ELL_KS>::encode(
                &mut lwe,
                pirs[0].compressor(),
                &packed_bits,
                pack_dim,
            ))
        };

        let mut query_gsw: Vec<GswCiphertextCompressed<_, ELL_GSW>> = Vec::new();
        if gsw_synth {
            query_gsw.push(lwe.gsw_encrypt_minus_secret(&pirs[0].compressor));
        } else {
            query_gsw.extend(gsw_bit_vec.into_iter().map(|bit| {
                let gsw_val = if bit {
                    Poly::from_scalar(Field::ONE)
                } else {
                    Poly::from_scalar(Field::ZERO)
                };
                lwe.encrypt_gsw_compressed(&pirs[0].compressor, gsw_val)
            }));
        }

        timer.end();
        println!("Database slice is {len} Rq elements. Query index is {query_index}.");

        let query = Query {
            gsw_bits: usize::try_from(gsw_bits).unwrap(),
            bfv_bits: usize::try_from(bfv_bits).unwrap(),
            gsw_synth,
            packed,
            gsw_ciphertexts: query_gsw,
        };

        println!(
            "Query size is {} kB, answer size is {} kB.",
            query.size() / 1024,
            2 * LPrime::Q_BITS * LPrime::DEGREE / 8192,
        );

        let results = pirs
            .iter()
            .map(|pir| pir.query::<_, LPrime>(&query, &mut lwe, query_index))
            .collect::<Vec<_>>();

        let mut total_time_detail = QueryTimings::default();

        for (i, result) in results.into_iter().enumerate() {
            let t = &result.timings;
            total_time_detail.unpack_duration += t.unpack_duration;
            total_time_detail.bfv_query_duration += t.bfv_query_duration;
            total_time_detail.gsw_query_duration += t.gsw_query_duration;
            total_time_detail.e2e_query_duration += t.e2e_query_duration - t.unpack_duration;

            let response = lwe.decrypt_bfv(result.answer);
            //println!("{} {}", response.0.as_ref()[0], pir.database[query_index].0.as_ref()[0]);
            //let expected = Pir::<L::PirBackend>::maybe_invert_plaintext(query_index, &pirs[i].database[query_index]);
            let expected = LPrime::array_from_fn(|j| {
                pirs[i].database[query_index]
                    .get(j * L::P_BITS..(j + 1) * L::P_BITS)
                    .unwrap()
                    .load_le()
            })
            .into();
            assert!(
                response == expected,
                "Query {i} failed: {} != {}",
                response.as_raw_storage()[0],
                expected.as_raw_storage()[0]
            );
        }

        let avg_bfv_query_duration =
            total_time_detail.bfv_query_duration.as_nanos() / u128::try_from(slices).unwrap();
        let avg_gsw_query_duration =
            total_time_detail.gsw_query_duration.as_micros() / u128::try_from(slices).unwrap();

        let total_unpack = total_time_detail.unpack_duration / u32::try_from(slices).unwrap();
        let total_time = total_time_detail.e2e_query_duration;
        println!(
            "Total query time for all slices, excluding unpacking: {:?}",
            total_time
        );
        println!(
            "Total query time for all slices, including unpacking: {:?}",
            total_unpack + total_time
        );
        println!(
            "BFV query throughput: {:.1} GiB/s",
            encoded_db_bytes as f32 / avg_bfv_query_duration as f32
        );
        println!(
            "GSW query throughput: {:.1} MiB/s",
            bfv_output_bytes as f32 / avg_gsw_query_duration as f32
        );
        println!(
            "Normalized GSW query throughput (ell_gsw = {ELL_GSW}): {:.1} MiB/s",
            (bfv_output_bytes * ELL_GSW) as f32 / avg_gsw_query_duration as f32,
        );
    }

    // Applying PIR independently to stripes of this size, gives a 512 kB total response
    // for the 128MB safe browsing database, before any modulus switching.
    const DB_SIZE_1024: usize = 4096 * 512;
    const DB_SIZE_2048: usize = 4096 * 2048;

    #[test]
    #[ignore]
    fn final_256mb_q30_q_cost() {
        pir::<Lwe1024Q30P1, 2, 10, Lwe1024Q14P1>(1 << 28, 64, 16, 16, None);
    }

    #[test]
    #[ignore]
    fn final_256mb_q30_query_sz() {
        pir::<Lwe1024Q30P1, 2, 8, Lwe1024Q14P1>(1 << 28, 64, 16, 4, None);
    }

    #[test]
    #[ignore]
    fn final_256mb_q30_t_cost() {
        pir::<Lwe1024Q30P1, 2, 10, Lwe1024Q14P1>(1 << 28, 8, 16, 16, None);
    }

    #[test]
    #[ignore]
    fn final_256mb_q30_t_comm() {
        pir::<Lwe1024Q30P1, 2, 8, Lwe1024Q14P1>(1 << 28, 8, 16, 4, None);
    }

    #[test]
    #[ignore]
    fn final_256mb_q30_compute() {
        pir::<Lwe1024Q30P6, 3, 3 /* unused */, Lwe1024Q20P6>(1 << 28, 8, 1, 128, None);
    }

    #[test]
    #[ignore]
    fn final_256mb_q56_q_cost() {
        pir::<Lwe2048Q56P8, 2, 6, Lwe2048Q21P8>(1 << 28, 16, 256, 2, None);
    }

    #[test]
    #[ignore]
    fn final_256mb_q56_query_sz() {
        pir::<Lwe2048Q56P1, 4, 7, Lwe2048Q21P1>(1 << 28, 16, 32, 2, Some(12))
    }

    #[test]
    #[ignore]
    fn final_256mb_q56_t_cost() {
        pir::<Lwe2048Q56P4, 6, 8, Lwe2048Q21P4>(1 << 28, 1, 128, 3, Some(10));
    }

    #[test]
    #[ignore]
    fn final_256mb_q56_t_comm() {
        pir::<Lwe2048Q56P1, 4, 8, Lwe2048Q21P1>(1 << 28, 1, 64, 2, Some(14));
    }

    #[test]
    #[ignore]
    fn final_256mb_q56_compute() {
        pir::<Lwe2048Q56P8, 2, 4, Lwe2048Q21P8>(1 << 28, 8, 16, 32, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q30_q_cost() {
        pir::<Lwe1024Q30P1, 2, 10, Lwe1024Q14P1>(1 << 30, 64, 16, 16, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q30_query_sz() {
        pir::<Lwe1024Q30P1, 2, 8, Lwe1024Q14P1>(1 << 30, 64, 16, 4, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q30_t_cost() {
        pir::<Lwe1024Q30P1, 2, 15, Lwe1024Q14P1>(1 << 30, 8, 16, 32, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q30_t_comm() {
        pir::<Lwe1024Q30P1, 2, 8, Lwe1024Q14P1>(1 << 30, 8, 16, 4, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q30_compute() {
        pir::<Lwe1024Q30P6, 3, 3 /* unused */, Lwe1024Q20P6>(1 << 30, 8, 1, 128, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q56_q_cost() {
        pir::<Lwe2048Q56P4, 6, 8, Lwe2048Q21P4>(1 << 30, 1, 128, 5, Some(11));
    }

    #[test]
    #[ignore]
    fn final_1gb_q56_query_sz() {
        pir::<Lwe2048Q56P1, 4, 8, Lwe2048Q21P1>(1 << 30, 1, 64, 2, Some(16));
    }

    #[test]
    #[ignore]
    fn final_1gb_q56_t_cost() {
        pir::<Lwe2048Q56P6, 7, 8, Lwe2048Q21P6>(1 << 30, 1, 128, 5, Some(11));
    }

    #[test]
    #[ignore]
    fn final_1gb_q56_t_comm() {
        pir::<Lwe2048Q56P1, 4, 8, Lwe2048Q21P1>(1 << 30, 1, 64, 2, Some(16));
    }

    #[test]
    #[ignore]
    fn final_1gb_q56_compute() {
        pir::<Lwe2048Q56P12, 2, 8, Lwe2048Q25P12>(1 << 30, 8, 16, 16, None);
    }

    // log_q    log_p    t_gsw    t_ks    l_gsw    l_ks    slices    i_bits    pf    pd    n_ksk    i0_bits    g_bits    g_mode    nse    n_mgn    query_sz    resp_sz    tot_sz    comp    score    cost
    // -------  -------  -------  ------  -------  ------  --------  --------  ----  ----  -------  ---------  --------  --------  -----  -------  ----------  ---------  --------  ------  -------  ------
    //      56        8       28      12        2       5         8        16     2     8        2          9         7         0   43.8      4.2      560          84      644       0.56      138       8
    //      56        6        8       8        7       7         1        20     2     8        2         10        10         1   45.8      4.2      462          10.5    472.5     1.16      124       7

    #[test]
    #[ignore]
    fn final_1gb_q56_a() {
        pir::<Lwe2048Q56P8, 2, 5, Lwe2048Q21P8>(1 << 30, 8, 256, 2, None);
    }

    #[test]
    #[ignore]
    fn final_1gb_q56_b() {
        pir::<Lwe2048Q56P6, 7, 7, Lwe2048Q21P6>(1 << 30, 1, 256, 5, Some(10));
    }

    #[test]
    #[ignore]
    fn final_128mb_a() {
        pir::<Lwe1024Q30P1, 2, 10, Lwe1024Q14P1>(1 << 27, 64, 16, 16, None);
    }

    #[test]
    #[ignore]
    fn final_128mb_b() {
        pir::<Lwe1024Q30P1, 2, 10, Lwe1024Q14P1>(1 << 27, 8, 16, 16, None);
    }

    #[test]
    #[ignore]
    fn final_128mb_c() {
        pir::<Lwe2048Q56P8, 2, 4, Lwe2048Q21P8>(1 << 27, 16, 64, 4, None);
    }

    #[test]
    fn q30_p1() {
        pir::<Lwe1024Q30P1, 2, 10, Lwe1024Q30P1>(DB_SIZE_1024, 1, 16, 16, None);
    }

    #[test]
    fn q30_p2_a() {
        pir::<Lwe1024Q30P2, 2, 8, Lwe1024Q30P2>(DB_SIZE_1024, 1, 4, 8, None);
    }

    #[test]
    fn q30_p2_b() {
        pir::<Lwe1024Q30P2, 2, 8, Lwe1024Q30P2>(DB_SIZE_1024, 1, 4, 16, None);
    }

    #[test]
    fn q30_p2_c() {
        pir::<Lwe1024Q30P2, 2, 10, Lwe1024Q30P2>(DB_SIZE_1024, 1, 4, 32, None);
    }

    #[test]
    fn q30_p6() {
        pir::<Lwe1024Q30P6, 3, 10, Lwe1024Q30P6>(DB_SIZE_1024, 1, 1, 32, None);
    }

    #[test]
    fn q31_p7() {
        pir::<Lwe1024Q31P7, 3, 1, Lwe1024Q31P7>(DB_SIZE_1024, 1, 1, 32, None);
    }

    #[test]
    fn q31_p4() {
        pir::<Lwe1024Q31P4, 3, 1, Lwe1024Q31P4>(DB_SIZE_1024, 1, 1, 32, None);
    }

    #[test]
    fn q31_p2_a() {
        pir::<Lwe1024Q31P2, 2, 11, Lwe1024Q31P2>(DB_SIZE_1024, 1, 8, 8, None);
    }

    #[test]
    fn q31_p2_b() {
        pir::<Lwe1024Q31P2, 2, 7, Lwe1024Q31P2>(DB_SIZE_1024, 1, 4, 16, None);
    }

    #[test]
    fn q31_p2_c() {
        pir::<Lwe1024Q31P2, 2, 16, Lwe1024Q31P2>(DB_SIZE_1024, 1, 8, 8, None);
    }

    #[test]
    fn q56_p4() {
        pir::<Lwe2048Q56P4, 7, 8, Lwe2048Q56P4>(DB_SIZE_2048, 1, 32, 6, Some(6));
    }

    #[test]
    fn q56_p8() {
        pir::<Lwe2048Q56P8, 2, 6, Lwe2048Q56P8>(DB_SIZE_2048, 1, 32, 4, None);
    }

    /*
     * These configs are broken because Q56 is hardcoded in `ResidueNtt`.
    #[test]
    fn q62_p8_a() {
        pir::<Lwe2048Q62P8, 3, 3, Lwe2048Q62P8>(DB_SIZE_2048, 1, 16, 4, None);
    }

    #[test]
    fn q62_p8_b() {
        pir::<Lwe2048Q62P8, 2, 4, Lwe2048Q62P8>(DB_SIZE_2048, 1, 64, 4, None);
    }

    #[test]
    fn q62_p16() {
        pir::<Lwe2048Q62P16, 5, 13, Lwe2048Q62P16>(DB_SIZE_2048, 1, 16, 2, None);
    }
    */
}
