use std::iter::zip;

use rand::{CryptoRng, Rng};
use rand_distr::{Distribution, Standard};

use crate::{
    field::Field,
    lwe::{
        BfvCiphertext, BfvCiphertextCompressed, BfvCiphertextNtt, Compressor, Decompress,
        GswCiphertextCompressed, GswCiphertextNtt, Lwe, LweParams,
    },
    morph::{unpack_all, EncryptedMorphCompressed},
    poly::Poly,
};

pub struct Query<L: LweParams, const D: usize> {
    pub gsw_bits: usize,
    pub bfv_bits: usize,
    pub gsw_synth: bool,
    pub packed: Box<dyn QueryBfv<LweParams = L>>,
    pub gsw_ciphertexts: Vec<GswCiphertextCompressed<L, D>>,
}

impl<L: LweParams, const D: usize> Query<L, D> {
    pub fn size(&self) -> usize {
        self.packed.size() + self.gsw_ciphertexts.len() * 2 * D * L::Q_BITS * L::DEGREE / 8
    }
}

pub struct DecodedQuery<L: LweParams, const D: usize> {
    pub bfv_ciphertexts: Vec<BfvCiphertext<L>>,
    pub gsw_ciphertexts: Vec<GswCiphertextNtt<L, D>>,
}

fn gsw_synth<L: LweParams, const D: usize>(
    elts: &[BfvCiphertext<L>],
    minus_s: &GswCiphertextNtt<L, D>,
) -> GswCiphertextNtt<L, D> {
    let (c00, c01): (Vec<_>, Vec<_>) = elts.iter().map(|elt| (minus_s * elt).into_raw()).unzip();
    let (c10, c11): (Vec<_>, Vec<_>) = elts
        .iter()
        .map(|elt| BfvCiphertextNtt::from(elt.clone()).into_raw())
        .unzip();

    GswCiphertextNtt::from_raw(
        c00.try_into().unwrap(),
        c01.try_into().unwrap(),
        c10.try_into().unwrap(),
        c11.try_into().unwrap(),
    )
}

impl<L: LweParams, const D: usize> Query<L, D> {
    pub fn decode(&self, compressor: &Compressor) -> DecodedQuery<L, D> {
        let mut unpacked = self.packed.decode(compressor);
        if self.gsw_synth {
            let Query {
                gsw_bits,
                bfv_bits,
                gsw_synth: _,
                packed: _,
                gsw_ciphertexts,
            } = self;

            assert_eq!(gsw_ciphertexts.len(), 1);
            let minus_s =
                GswCiphertextNtt::from(gsw_ciphertexts.last().unwrap().decompress(compressor));

            assert!(unpacked.len() >= (1 << bfv_bits) + gsw_bits * D);

            let synth_gsw_ciphertexts = unpacked[unpacked.len() - gsw_bits * D..]
                .chunks(D)
                .map(|src_set| gsw_synth(src_set, &minus_s))
                .collect::<Vec<_>>();

            unpacked.truncate(1 << bfv_bits);

            DecodedQuery {
                bfv_ciphertexts: unpacked,
                gsw_ciphertexts: synth_gsw_ciphertexts,
            }
        } else {
            assert_eq!(self.gsw_ciphertexts.len(), self.gsw_bits);
            assert_eq!(unpacked.len(), 1 << self.bfv_bits);

            DecodedQuery {
                bfv_ciphertexts: unpacked,
                gsw_ciphertexts: self
                    .gsw_ciphertexts
                    .iter()
                    .map(|c| GswCiphertextNtt::from(c.decompress(compressor)))
                    .collect::<Vec<_>>(),
            }
        }
    }
}

pub trait QueryBfv {
    type LweParams: LweParams;

    fn bits(&self) -> u32;

    fn decode(&self, compressor: &Compressor) -> Vec<BfvCiphertext<Self::LweParams>>;

    fn size(&self) -> usize;
}

impl<L: LweParams> QueryBfv for Box<dyn QueryBfv<LweParams = L>> {
    type LweParams = L;

    fn bits(&self) -> u32 {
        self.as_ref().bits()
    }

    fn decode(&self, compressor: &Compressor) -> Vec<BfvCiphertext<Self::LweParams>> {
        self.as_ref().decode(compressor)
    }

    fn size(&self) -> usize {
        self.as_ref().size()
    }
}

pub struct SimpleQuery<L: LweParams> {
    pub query: Vec<BfvCiphertextCompressed<L>>,
}

impl<L: LweParams> SimpleQuery<L> {
    pub fn encode<R: Rng + CryptoRng + 'static>(
        lwe: &mut Lwe<L, R>,
        compressor: &Compressor,
        packed: &[Field<L>],
    ) -> Self {
        let query = packed
            .iter()
            .map(|&v| lwe.encrypt_bfv_poly_compressed(compressor, Poly::from_scalar(v)))
            .collect();
        Self { query }
    }
}

impl<L: LweParams> QueryBfv for SimpleQuery<L> {
    type LweParams = L;

    fn bits(&self) -> u32 {
        self.query.len().ilog2()
    }

    fn decode(&self, compressor: &Compressor) -> Vec<BfvCiphertext<L>> {
        self.query
            .iter()
            .map(|c| c.decompress(compressor))
            .collect()
    }

    fn size(&self) -> usize {
        self.query.len() * L::Q_BITS * L::DEGREE / 8
    }
}

pub struct PackedQuery<L: LweParams, const ELL_KS: usize> {
    pub pack_dim: usize,
    pub morph: EncryptedMorphCompressed<L, ELL_KS>,
    pub packed_query: Vec<BfvCiphertextCompressed<L>>,
}

impl<L: LweParams, const ELL_KS: usize> PackedQuery<L, ELL_KS>
where
    Standard: Distribution<L::Ntt>,
{
    pub fn encode<R: Rng + CryptoRng + 'static>(
        lwe: &mut Lwe<L, R>,
        compressor: &Compressor,
        packed: &[Field<L>],
        pack_dim: usize,
    ) -> Self {
        assert!(pack_dim.is_power_of_two());

        let mut packed_query = Vec::with_capacity(packed.len());
        for (i, pack_chunk) in packed.chunks(pack_dim).enumerate() {
            let mut plaintext = L::array_zero();
            for (dest, v) in zip(plaintext.as_mut().iter_mut(), pack_chunk) {
                *dest = v.to_raw();
            }
            let q = lwe.encrypt_bfv_poly_compressed(compressor, plaintext.clone().into());
            let qd = q.decompress(compressor);
            packed_query.push(q.clone());
            if super::PRINT_NOISE {
                println!(
                    "query[{i}]: log₂|ε| = {:.1}",
                    (<_ as TryInto<u64>>::try_into(
                        (lwe.raw_decrypt_bfv(qd) - plaintext.into()).inf_norm()
                    )
                    .ok()
                    .unwrap() as f32)
                        .log2(),
                );
            }
        }
        let morph = lwe
            .unpack_morph_compressed::<ELL_KS>(compressor, pack_dim)
            .unwrap();

        Self {
            pack_dim,
            morph,
            packed_query,
        }
    }
}

impl<L: LweParams, const ELL_KS: usize> QueryBfv for PackedQuery<L, ELL_KS>
where
    Standard: Distribution<L::Ntt>,
{
    type LweParams = L;

    fn bits(&self) -> u32 {
        self.pack_dim.ilog2() + self.packed_query.len().ilog2()
    }

    fn decode(&self, compressor: &Compressor) -> Vec<BfvCiphertext<L>> {
        let morph = self.morph.decompress(compressor);

        self.packed_query
            .iter()
            .flat_map(|c| unpack_all::<_, ELL_KS>(&morph, self.pack_dim, &c.decompress(compressor)))
            .collect()
    }

    fn size(&self) -> usize {
        self.morph.size() + self.packed_query.len() * L::Q_BITS * L::DEGREE / 8
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;
    use crate::{
        field::Field,
        lwe::{Lwe2048, LwePrivate},
        poly::Poly,
        Decompose,
    };

    #[test]
    fn test_gsw_synth() {
        type LweParams = Lwe2048;
        const D: usize = 7;

        let mut rng = thread_rng();
        let mut lwe = LweParams::gen_noise(rng.clone());

        let basis: Box<[_; D]> = Field::<LweParams>::basis();
        let synth_bfv_cts = basis
            .iter()
            .map(|&g| lwe.encrypt_bfv_poly(Poly::from_scalar(g)))
            .collect::<Vec<_>>();
        let gsw_minus_s = lwe.gsw_encrypt_minus_secret::<D>();

        let gsw_ct = gsw_synth(&synth_bfv_cts, &GswCiphertextNtt::from(gsw_minus_s));

        let test_bfv_pt = rng.gen_range(0..4);
        let test_bfv_ct = lwe.encrypt_bfv(test_bfv_pt);

        let multiplied = BfvCiphertext::from(&gsw_ct * &test_bfv_ct);

        let plaintext_out = lwe.decrypt_bfv(multiplied);
        assert_eq!(
            plaintext_out.as_raw_storage()[0],
            <LweParams as LwePrivate>::Storage::from(test_bfv_pt),
        );
    }
}
