use std::{
    array,
    fmt::Debug,
    iter::{repeat_with, zip},
    ops::{Add, AddAssign, Mul, Neg, Sub},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

use concrete_ntt::{prime32::Plan as Prime32Plan, prime64::Plan as Prime64Plan};
use rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::Normal;

use crate::{
    field::Field,
    map_boxed_array,
    morph::{EncryptedMorph, EncryptedMorphCompressed, Morph},
    pir::{
        backend::{Pir31, Pir62Crt, PirGeneric},
        Pir, Seed,
    },
    poly::{Ntt, NttTrait, Poly, ResidueNtt},
    Decompose, InnerProduct, OneMinus,
};

const SIGMA: f32 = 6.4;

pub trait LweParams: LwePrivate {
    fn plaintext_bytes_per_ciphertext() -> usize {
        Self::P_BITS * Self::DEGREE / 8
    }

    fn p_modulus() -> usize {
        Self::P_MODULUS.try_into().ok().unwrap()
    }

    fn q_modulus() -> usize {
        Self::Q_MODULUS.try_into().ok().unwrap()
    }

    // Generates an `Lwe` instance with the secret drawn from the uniform distribution.
    fn gen_uniform<R: Rng + CryptoRng + 'static>(rng: R) -> Lwe<Self, R>;

    // Generates an `Lwe` instance with the secret drawn from the noise distribution.
    fn gen_noise<R: Rng + CryptoRng + 'static>(rng: R) -> Lwe<Self, R>;

    // Generates an `Lwe` instance with a binary secret.
    fn gen_binary<R: Rng + CryptoRng + 'static>(rng: R) -> Lwe<Self, R>;

    fn pir<const ELL_GSW: usize, const ELL_KS: usize>(
        db_bytes: &[u8],
        bfv_query_len: usize,
        pack_dim: usize,
        n_pack: usize,
        seed: Seed,
    ) -> Pir<Self::PirBackend, ELL_GSW, ELL_KS>
    where
        Standard: Distribution<Self::Ntt>,
    {
        Pir::new(db_bytes, bfv_query_len, pack_dim, n_pack, seed)
    }
}

mod sealed {
    use std::{
        fmt::Debug,
        ops::{Add, Mul, Neg, Sub, SubAssign},
    };

    use funty::Integral;
    use rand_distr::uniform::SampleUniform;

    use crate::{
        field::Field,
        pir::PirBackend,
        poly::{NttTrait, Plan, Poly},
        InnerProduct,
    };

    // Note: sealing this trait prevents implementations in other crates, but it does
    // not truly hide the members of the trait from exposure via the supertrait bound
    // of `LweParams`.
    pub trait LwePrivate: Copy + Debug + Default + Eq + PartialEq + Sized + 'static {
        const DEGREE: usize;
        const P_MODULUS: Self::Storage;
        const P_BITS: usize;
        const Q_MODULUS: Self::Storage;
        const Q_BITS: usize;

        fn name() -> &'static str;

        /// Integer type that can represent `Field<L>`.
        type Storage: Integral + SampleUniform + From<u32>;

        /// A (usually wider) integer type that can store intermediate values when
        /// operating on `Field<L>` values.
        type OpStorage: Integral + From<Self::Storage> + From<u32>;

        fn reduce(value: Self::OpStorage) -> Self::Storage;

        fn field(value: Self::Storage) -> Field<Self> {
            Field::new(value)
        }

        fn floor_q_div_p() -> Self::Storage {
            Self::Q_MODULUS / Self::P_MODULUS
        }

        /// An array of Self::Storage holding polynomial coefficients. We use an array
        /// of the integer type, because that is what `concrete-ntt` wants.
        type Array: Clone
            + Debug
            + PartialEq
            + Eq
            + Into<Poly<Self>>
            + TryFrom<Vec<Self::Storage>, Error = Vec<Self::Storage>>
            + AsRef<[Self::Storage]>
            + AsMut<[Self::Storage]>;

        fn array_zero() -> Self::Array {
            Self::array_from_fn(|_| Default::default())
        }

        fn array_from_fn<F: FnMut(usize) -> Self::Storage>(f: F) -> Self::Array;

        type Morph: AsRef<[i16]> + Clone;

        fn morph_from_fn<F: Fn(usize) -> i16>(f: F) -> Self::Morph;

        type Plan: Plan<Storage = Self::Storage>;

        fn plan() -> &'static Self::Plan;

        type PirBackend: PirBackend<LweParams = Self>;

        type Ntt: NttTrait<Self>
            + Add<Self::Ntt, Output = Self::Ntt>
            + Sub<Self::Ntt, Output = Self::Ntt>
            + for<'a> Sub<&'a Self::Ntt, Output = Self::Ntt>
            + for<'a> SubAssign<&'a Self::Ntt>
            + Mul<Self::Ntt, Output = Self::Ntt>
            + Neg<Output = Self::Ntt>
            + InnerProduct
            + From<Poly<Self>>
            + Into<Poly<Self>>
            + Clone
            + Default
            + Debug;
    }
}
pub(crate) use sealed::LwePrivate;

macro_rules! lwe_impl {
    ($ty:ident, $name:literal, $degree:expr, $p_modulus:expr, $q_modulus:expr, $store:ty, $op_store:ty, $plan:ty, $backend:ty, $ntt:ty) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $ty;

        impl sealed::LwePrivate for $ty {
            const DEGREE: usize = $degree;
            const P_MODULUS: Self::Storage = $p_modulus;
            const P_BITS: usize = Self::P_MODULUS.ilog2() as usize;
            const Q_MODULUS: Self::Storage = $q_modulus;
            const Q_BITS: usize = (Self::Q_MODULUS.ilog2() + 1) as usize;

            fn name() -> &'static str {
                $name
            }

            type Storage = $store;
            type OpStorage = $op_store;
            type Array = [$store; Self::DEGREE];

            fn reduce(value: Self::OpStorage) -> Self::Storage {
                <$store>::try_from(value % <$op_store>::from(Self::Q_MODULUS)).unwrap()
            }

            fn array_from_fn<F: FnMut(usize) -> Self::Storage>(f: F) -> Self::Array {
                array::from_fn(f)
            }

            type Morph = [i16; Self::DEGREE];
            fn morph_from_fn<F: Fn(usize) -> i16>(f: F) -> Self::Morph {
                array::from_fn(f)
            }

            type Plan = $plan;
            fn plan() -> &'static $plan {
                static PLAN: OnceLock<$plan> = OnceLock::new();
                PLAN.get_or_init(|| <$plan>::try_new(Self::DEGREE, Self::Q_MODULUS).unwrap())
            }

            type PirBackend = $backend;

            type Ntt = $ntt;
        }
    };

    ($ty:ident, $name:literal, 32, $q_modulus:expr, $p_modulus:expr, PirGeneric) => {
        lwe_impl!($ty, $name, 1024, $p_modulus, $q_modulus, u32, u64, Prime32Plan, PirGeneric<$ty>, Ntt<$ty>);
    };

    ($ty:ident, $name:literal, 32, $q_modulus:expr, $p_modulus:expr, Pir31) => {
        lwe_impl!($ty, $name, 1024, $p_modulus, $q_modulus, u32, u64, Prime32Plan, Pir31<$ty, $q_modulus>, Ntt<$ty>);
    };

    // TODO: 65!? This is awful.
    ($ty:ident, $name:literal, 65, $q_modulus:expr, $p_modulus:expr, PirGeneric) => {
        lwe_impl!($ty, $name, 2048, $p_modulus, $q_modulus, u64, u128, Prime64Plan, PirGeneric<$ty>, Ntt<$ty>);
    };

    ($ty:ident, $name:literal, 64, $q_factor_a:expr, $q_factor_b:expr, $p_modulus:expr) => {
        lwe_impl!($ty, $name, 2048, $p_modulus, $q_factor_a as u64 * $q_factor_b as u64, u64, u128, Prime64Plan, Pir62Crt<$ty, $q_factor_a, $q_factor_b>, ResidueNtt<$ty>);
    };
}

// Note: an impl `From<[$store; $degree]> for Poly<L>` is required in `poly.rs` for each
// unique combination of `[$store; $degree]` (i.e. `[u32; 1024]`, `[u64; 2048]`).

pub const Q30: u32 = 0x3fff_7801;
pub const Q31: u32 = 0x7fff_d801;
pub const Q32: u32 = 0xffff_d801;

pub const Q56_F0: u32 = 0xfff_0001;
pub const Q56_F1: u32 = 0xffe_e001;

pub const Q62_F0: u32 = 0x7ffe_9001;
pub const Q62_F1: u32 = 0x7ffe_6001;

lwe_impl!(Lwe1024Q32P8, "1024q32p8", 32, Q32, 256, PirGeneric);

lwe_impl!(Lwe1024Q31P1, "1024q31p1", 32, Q31, 2, Pir31);
lwe_impl!(Lwe1024Q31P2, "1024q31p2", 32, Q31, 4, Pir31);
lwe_impl!(Lwe1024Q31P4, "1024q31p4", 32, Q31, 16, Pir31);
lwe_impl!(Lwe1024Q31P6, "1024q31p6", 32, Q31, 64, Pir31);
lwe_impl!(Lwe1024Q31P7, "1024q31p7", 32, Q31, 128, Pir31);
lwe_impl!(Lwe1024Q31P8, "1024q31p8", 32, Q31, 256, Pir31);

lwe_impl!(Lwe1024Q30P1, "1024q30p1", 32, Q30, 2, Pir31);
lwe_impl!(Lwe1024Q30P2, "1024q30p2", 32, Q30, 4, Pir31);
lwe_impl!(Lwe1024Q30P4, "1024q30p4", 32, Q30, 16, Pir31);
lwe_impl!(Lwe1024Q30P6, "1024q30p6", 32, Q30, 64, Pir31);
lwe_impl!(Lwe1024Q30P8, "1024q30p8", 32, Q30, 256, Pir31);

lwe_impl!(Lwe1024Q14P1, "1024q14p1", 32, 0x3001, 2, Pir31);
lwe_impl!(Lwe1024Q14P4, "1024q14p4", 32, 0x3001, 16, Pir31);

lwe_impl!(Lwe1024Q20P6, "1024q20p6", 32, 0xfd801, 64, Pir31);

pub type Lwe1024 = Lwe1024Q31P4;

lwe_impl!(Lwe2048Q56P1, "2048q56p1", 64, Q56_F0, Q56_F1, 2);
lwe_impl!(Lwe2048Q56P4, "2048q56p4", 64, Q56_F0, Q56_F1, 16);
lwe_impl!(Lwe2048Q56P6, "2048q56p6", 64, Q56_F0, Q56_F1, 64);
lwe_impl!(Lwe2048Q56P8, "2048q56p8", 64, Q56_F0, Q56_F1, 256);
lwe_impl!(Lwe2048Q56P12, "2048q56p12", 64, Q56_F0, Q56_F1, 4096);
lwe_impl!(Lwe2048Q56P16, "2048q56p16", 64, Q56_F0, Q56_F1, 65536);

lwe_impl!(Lwe2048Q21P1, "2048q21p1", 65, 0x1f6001, 2, PirGeneric);
lwe_impl!(Lwe2048Q21P4, "2048q21p4", 65, 0x1f6001, 16, PirGeneric);
lwe_impl!(Lwe2048Q21P6, "2048q21p6", 65, 0x1f6001, 64, PirGeneric);
lwe_impl!(Lwe2048Q21P8, "2048q21p8", 65, 0x1f6001, 256, PirGeneric);

lwe_impl!(
    Lwe2048Q25P12,
    "2048q25p12",
    65,
    0x1ff_f001,
    4096,
    PirGeneric
);

// TODO: these configs are currently broken because Q56 is hardcoded in `ResidueNtt`.
// (Could be fixed by resolving that, or by tweaking the macro so Q62 configs use
// the PirGeneric backend.)
lwe_impl!(Lwe2048Q62P8, "2048q62p8", 64, Q62_F0, Q62_F1, 256);
lwe_impl!(Lwe2048Q62P16, "2048q62p16", 64, Q62_F0, Q62_F1, 65536);

pub type Lwe2048 = Lwe2048Q56P8;

fn noise<L: LwePrivate, R: Rng + CryptoRng + 'static>(rng: &mut R) -> Poly<L> {
    // This only fails if stdev is non-finite.
    let mut distr = rng.sample_iter(Normal::new(0 as f32, SIGMA).unwrap());
    L::array_from_fn(|_| {
        let val = distr.next().unwrap().round();
        if val >= 0.0 {
            L::Storage::from(val as u32)
        } else {
            L::Q_MODULUS - L::Storage::from(-val as u32)
        }
    })
    .into()
}

impl<L: LwePrivate> LweParams for L
where
    Standard: Distribution<Poly<L>>,
{
    fn gen_uniform<R: Rng + CryptoRng + 'static>(mut rng: R) -> Lwe<Self, R> {
        let secret = rng.gen();
        Lwe { secret, rng, noise }
    }

    fn gen_noise<R: Rng + CryptoRng + 'static>(mut rng: R) -> Lwe<Self, R> {
        let noise = noise;
        let secret = noise(&mut rng);
        Lwe { secret, rng, noise }
    }

    fn gen_binary<R: Rng + CryptoRng + 'static>(mut rng: R) -> Lwe<Self, R> {
        let secret =
            L::array_from_fn(|_| L::Storage::from(if rng.gen::<bool>() { 1 } else { 0 })).into();
        Lwe { secret, rng, noise }
    }
}

pub struct Compressor {
    index: AtomicU64,
    prg: ChaCha12Rng,
}

impl Clone for Compressor {
    fn clone(&self) -> Self {
        Self {
            index: AtomicU64::default(),
            prg: self.prg.clone(),
        }
    }
}

impl Compressor {
    pub fn new(seed: <ChaCha12Rng as SeedableRng>::Seed) -> Self {
        Self {
            index: AtomicU64::default(),
            prg: ChaCha12Rng::from_seed(seed),
        }
    }

    pub fn gen<T>(&self) -> (u64, T)
    where
        Standard: Distribution<T>,
    {
        let index = self.index.fetch_add(1, Ordering::SeqCst);
        let mut prg = self.prg.clone();
        //println!("gen with index = {index}");
        prg.set_stream(index);
        let result = prg.gen();
        (index, result)
    }

    pub fn get<T>(&self, index: u64) -> T
    where
        Standard: Distribution<T>,
    {
        let mut prg = self.prg.clone();
        //println!("get with index = {index}");
        prg.set_stream(index);
        prg.gen()
    }
}

pub trait Decompress {
    type Decompressed;

    fn decompress(&self, compressor: &Compressor) -> Self::Decompressed;
}

/// Decrypt a (possibly modulus-switched) ciphertext, without rounding.
fn raw_decrypt_bfv<L, LPrime>(secret: &Poly<L>, c0: Poly<LPrime>, c1: Poly<LPrime>) -> Poly<LPrime>
where
    L: LweParams,
    LPrime: LweParams,
    LPrime::Storage: TryFrom<L::Storage>,
{
    if !(LPrime::DEGREE == L::DEGREE
        && LPrime::p_modulus() == L::p_modulus()
        && LPrime::q_modulus() <= L::q_modulus())
    {
        panic!("{} and {} are not compatible", L::name(), LPrime::name());
    }
    let half_l_modulus = L::Q_MODULUS / L::Storage::from(2);
    let secret = LPrime::array_from_fn(|i| {
        let term = secret.as_raw_storage()[i];
        if term > half_l_modulus {
            LPrime::Q_MODULUS - LPrime::Storage::try_from(L::Q_MODULUS - term).ok().unwrap()
        } else {
            term.try_into().ok().unwrap()
        }
    })
    .into();
    c1 - c0 * secret
}

pub struct Lwe<L: LweParams, R: Rng + CryptoRng + 'static> {
    secret: Poly<L>,
    rng: R,
    noise: fn(&mut R) -> Poly<L>,
}

impl<L: LweParams, R: Rng + CryptoRng + 'static> Lwe<L, R>
where
    Standard: Distribution<Poly<L>>,
{
    pub fn gen_uniform(rng: R) -> Self {
        L::gen_uniform(rng)
    }

    pub fn encrypt_bfv(&mut self, plaintext: u32) -> BfvCiphertext<L> {
        let plaintext_stor = L::Storage::from(plaintext);
        assert!(plaintext_stor < L::P_MODULUS);
        let plaintext_poly = Poly::from_scalar(L::field(plaintext_stor * L::floor_q_div_p()));

        self.encrypt_bfv_poly(plaintext_poly)
    }

    pub fn encrypt_bfv_compressed(
        &mut self,
        compressor: &Compressor,
        plaintext: u32,
    ) -> BfvCiphertextCompressed<L> {
        let plaintext_stor = L::Storage::from(plaintext);
        assert!(plaintext_stor < L::P_MODULUS);
        let plaintext_poly = Poly::from_scalar(L::field(plaintext_stor * L::floor_q_div_p()));

        self.encrypt_bfv_poly_compressed(compressor, plaintext_poly)
    }

    pub fn encrypt_bfv_poly(&mut self, plaintext: Poly<L>) -> BfvCiphertext<L> {
        let c0 = self.rng.gen::<Poly<L>>();
        let c1 = c0.clone() * self.secret.clone() + plaintext + (self.noise)(&mut self.rng);
        BfvCiphertext(c0, c1)
    }

    pub fn encrypt_bfv_poly_compressed(
        &mut self,
        compressor: &Compressor,
        plaintext: Poly<L>,
    ) -> BfvCiphertextCompressed<L> {
        let (index, c0) = compressor.gen();
        let c1 = c0 * self.secret.clone() + plaintext + (self.noise)(&mut self.rng);
        BfvCiphertextCompressed(index, c1)
    }

    /// Decrypt a (possibly modulus-switched) ciphertext
    pub fn decrypt_bfv<LPrime>(&self, ciphertext: BfvCiphertext<LPrime>) -> Poly<LPrime>
    where
        LPrime: LweParams,
        LPrime::Storage: TryFrom<L::Storage>,
    {
        self.raw_decrypt_bfv(ciphertext)
            .round(LPrime::floor_q_div_p())
    }

    /// Decrypt a (possibly modulus-switched) ciphertext, without rounding.
    ///
    /// This is public to facilitate noise measurements.
    pub fn raw_decrypt_bfv<LPrime>(&self, ciphertext: BfvCiphertext<LPrime>) -> Poly<LPrime>
    where
        LPrime: LweParams,
        LPrime::Storage: TryFrom<L::Storage>,
    {
        let BfvCiphertext(c0, c1) = ciphertext;
        raw_decrypt_bfv(&self.secret, c0, c1)
    }

    pub fn encrypt_gsw<const D: usize>(&mut self, plaintext: Poly<L>) -> GswCiphertext<L, D> {
        let zero: Box<[_; D]> = repeat_with(Default::default)
            .take(D)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut c00 = zero.clone();
        let mut c01 = zero.clone();
        let mut c10 = zero.clone();
        let mut c11 = zero.clone();

        for i in 0..D {
            let BfvCiphertext(c0, c1) = self.encrypt_bfv(Default::default());
            c00[i] = c0;
            c01[i] = c1;
            let BfvCiphertext(c0, c1) = self.encrypt_bfv(Default::default());
            c10[i] = c0;
            c11[i] = c1;
        }

        let z_matrix = GswCiphertext { c00, c01, c10, c11 };

        let mut c00 = zero.clone();
        let mut c11 = zero.clone();

        let mu_g_vector = <Poly<L> as Decompose<D>>::basis().map(|b| plaintext.clone() * b);
        for i in 0..D {
            c00[i] += &mu_g_vector[i];
            c11[i] += &mu_g_vector[i];
        }

        let mu_g_matrix = GswCiphertext {
            c00,
            c01: zero.clone(),
            c10: zero,
            c11,
        };

        z_matrix + mu_g_matrix
    }

    pub fn encrypt_gsw_compressed<const D: usize>(
        &mut self,
        compressor: &Compressor,
        plaintext: Poly<L>,
    ) -> GswCiphertextCompressed<L, D> {
        let zero: Box<[_; D]> = repeat_with(Default::default)
            .take(D)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut c00 = [0u64; D];
        let mut c01 = zero.clone();
        let mut c10 = [0u64; D];
        let mut c11 = zero.clone();

        for i in 0..D {
            let BfvCiphertextCompressed(c0, c1) =
                self.encrypt_bfv_compressed(compressor, Default::default());
            c00[i] = c0;
            c01[i] = c1;
            let BfvCiphertextCompressed(c0, c1) =
                self.encrypt_bfv_compressed(compressor, Default::default());
            c10[i] = c0;
            c11[i] = c1;
        }

        let mu_g_vector = <Poly<L> as Decompose<D>>::basis().map(|b| plaintext.clone() * b);
        for i in 0..D {
            c01[i] += -&self.secret * mu_g_vector[i].clone();
            c11[i] += &mu_g_vector[i];
        }

        GswCiphertextCompressed { c00, c01, c10, c11 }
    }

    pub fn decrypt_gsw<const D: usize>(&self, ciphertext: GswCiphertext<L, D>) -> Poly<L> {
        let bfv_ciphertext =
            BfvCiphertext::from_raw(ciphertext.c10[D - 1].clone(), ciphertext.c11[D - 1].clone());
        self.raw_decrypt_bfv(bfv_ciphertext)
    }

    /// Generate an `EncryptedMorph` (an evaluation key for an automorphism).
    ///
    /// If `power` is coprime with `2 * L::DEGREE`, returns the `EncryptedMorph` that
    /// maps $p(x) \mapsto p(x^j)$ for $j$ equal to `power`. Otherwise, returns `None`.
    pub fn morph<const D: usize>(&mut self, power: usize) -> Option<EncryptedMorph<L, D>> {
        let morph = Morph::new(power)?;

        let key0: Box<[_; D]> = repeat_with(|| L::Ntt::from(self.rng.gen()))
            .take(D)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let basis = <Poly<L> as Decompose<D>>::basis();
        let morph_secret = morph.apply(&self.secret);

        let key1 = (0..D)
            .map(|i| {
                L::Ntt::from(
                    (L::Ntt::from(self.secret.clone()) * key0[i].clone()).into()
                        - morph_secret.clone() * basis[i]
                        + (self.noise)(&mut self.rng),
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Some(EncryptedMorph {
            key: (key0, key1),
            morph,
        })
    }

    pub fn morph_compressed<const D: usize>(
        &mut self,
        compressor: &Compressor,
        power: usize,
    ) -> Option<EncryptedMorphCompressed<L, D>>
    where
        Standard: Distribution<L::Ntt>,
    {
        let morph = Morph::new(power)?;

        let (key0_ix_vec, key0_vec): (Vec<_>, Vec<_>) =
            repeat_with(|| compressor.gen::<L::Ntt>()).take(D).unzip();

        let key0_ix: [_; D] = key0_ix_vec.try_into().unwrap();
        let key0: Box<[_; D]> = key0_vec.try_into().unwrap();

        let basis = <Poly<L> as Decompose<D>>::basis();
        let morph_secret = morph.apply(&self.secret);

        let key1 = (0..D)
            .map(|i| {
                L::Ntt::from(
                    (L::Ntt::from(self.secret.clone()) * key0[i].clone()).into()
                        - morph_secret.clone() * basis[i]
                        + (self.noise)(&mut self.rng),
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Some(EncryptedMorphCompressed {
            key: (key0_ix, key1),
            morph,
        })
    }

    /// Generates the `EncryptedMorph` for pack factor of `pack`.
    ///
    /// Returns `None` if `pack` is not a power of two, is larger than L::DEGREE / 2, or
    /// is zero.
    pub fn unpack_morph<const D: usize>(&mut self, pack: usize) -> Option<EncryptedMorph<L, D>> {
        if !pack.is_power_of_two() || pack == 0 || pack > L::DEGREE / 2 {
            return None;
        }
        self.morph(2 * L::DEGREE / pack + 1)
    }

    pub fn unpack_morph_compressed<const D: usize>(
        &mut self,
        compressor: &Compressor,
        pack: usize,
    ) -> Option<EncryptedMorphCompressed<L, D>>
    where
        Standard: Distribution<L::Ntt>,
    {
        if !pack.is_power_of_two() || pack == 0 || pack > L::DEGREE / 2 {
            return None;
        }
        self.morph_compressed(compressor, 2 * L::DEGREE / pack + 1)
    }

    pub fn gsw_encrypt_minus_secret<const D: usize>(
        &mut self,
        compressor: &Compressor,
    ) -> GswCiphertextCompressed<L, D> {
        let minus_s = -&self.secret;
        self.encrypt_gsw_compressed(compressor, minus_s)
    }
}

#[derive(Clone, Debug)]
pub struct BfvCiphertext<L: LweParams>(Poly<L>, Poly<L>);

#[derive(Clone, Debug)]
pub struct BfvCiphertextCompressed<L: LweParams>(u64, Poly<L>);

impl<L: LweParams> Decompress for BfvCiphertextCompressed<L> {
    type Decompressed = BfvCiphertext<L>;

    fn decompress(&self, compressor: &Compressor) -> Self::Decompressed {
        let Self(index, c1) = self.clone();
        let c0 = compressor.get(index);
        BfvCiphertext(c0, c1)
    }
}

impl<L: LweParams> BfvCiphertextCompressed<L> {
    pub fn as_raw(&self) -> (u64, &Poly<L>) {
        (self.0, &self.1)
    }
}

#[derive(Clone, Debug)]
pub struct BfvCiphertextNtt<L: LweParams>(L::Ntt, L::Ntt);

#[derive(Clone, Debug)]
pub struct GswCiphertext<L: LweParams, const D: usize> {
    c00: Box<[Poly<L>; D]>,
    c01: Box<[Poly<L>; D]>,
    c10: Box<[Poly<L>; D]>,
    c11: Box<[Poly<L>; D]>,
}

#[derive(Clone, Debug)]
pub struct GswCiphertextCompressed<L: LweParams, const D: usize> {
    c00: [u64; D],
    c01: Box<[Poly<L>; D]>,
    c10: [u64; D],
    c11: Box<[Poly<L>; D]>,
}

impl<L: LweParams, const D: usize> Decompress for GswCiphertextCompressed<L, D> {
    type Decompressed = GswCiphertext<L, D>;

    fn decompress(&self, compressor: &Compressor) -> Self::Decompressed {
        let c00 = Box::new(self.c00.map(|i| compressor.get(i)));
        let c10 = Box::new(self.c10.map(|i| compressor.get(i)));
        GswCiphertext {
            c00,
            c01: self.c01.clone(),
            c10,
            c11: self.c11.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GswCiphertextNtt<L: LweParams, const D: usize> {
    c00: Box<[L::Ntt; D]>,
    c01: Box<[L::Ntt; D]>,
    c10: Box<[L::Ntt; D]>,
    c11: Box<[L::Ntt; D]>,
}

impl<L: LweParams> BfvCiphertext<L> {
    pub fn into_raw(self) -> (Poly<L>, Poly<L>) {
        (self.0, self.1)
    }

    pub fn as_raw(&self) -> (&Poly<L>, &Poly<L>) {
        (&self.0, &self.1)
    }

    pub fn as_raw_mut(&mut self) -> (&mut Poly<L>, &mut Poly<L>) {
        (&mut self.0, &mut self.1)
    }

    pub fn from_raw(c0: Poly<L>, c1: Poly<L>) -> Self {
        Self(c0, c1)
    }

    pub fn add_assign_scalar(&mut self, scalar: Field<L>) {
        self.1.add_assign_scalar(scalar);
    }

    pub fn rotate(&mut self, count: isize) {
        self.0.rotate(count);
        self.1.rotate(count);
    }

    pub fn switch_modulus<LPrime>(&self) -> BfvCiphertext<LPrime>
    where
        LPrime: LweParams,
        LPrime::Storage: TryFrom<L::OpStorage>,
        L::OpStorage: TryFrom<LPrime::Storage>,
    {
        BfvCiphertext(self.0.switch_modulus(), self.1.switch_modulus())
    }
}

impl<L: LweParams> BfvCiphertextNtt<L> {
    pub fn into_raw(self) -> (L::Ntt, L::Ntt) {
        (self.0, self.1)
    }

    pub fn as_raw(&self) -> (&L::Ntt, &L::Ntt) {
        (&self.0, &self.1)
    }

    pub fn as_raw_mut(&mut self) -> (&mut L::Ntt, &mut L::Ntt) {
        (&mut self.0, &mut self.1)
    }

    pub fn from_raw(c0: L::Ntt, c1: L::Ntt) -> Self {
        Self(c0, c1)
    }
}

impl<L: LweParams> From<BfvCiphertext<L>> for BfvCiphertextNtt<L> {
    fn from(value: BfvCiphertext<L>) -> Self {
        Self(value.0.into(), value.1.into())
    }
}

// Note: Only correct if the input ciphertext has normalized NTTs.
impl<L: LweParams> From<BfvCiphertextNtt<L>> for BfvCiphertext<L> {
    fn from(value: BfvCiphertextNtt<L>) -> Self {
        Self(value.0.into(), value.1.into())
    }
}

impl<L: LweParams> Add<Self> for BfvCiphertext<L> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl<L: LweParams> Sub<Self> for BfvCiphertext<L> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

/// BFV plaintext-ciphertext addition
impl<L: LweParams> AddAssign<Poly<L>> for BfvCiphertext<L> {
    fn add_assign(&mut self, rhs: Poly<L>) {
        self.0 += rhs;
    }
}

/// BFV plaintext-ciphertext multiplication
impl<L: LweParams> Mul<Poly<L>> for BfvCiphertext<L> {
    type Output = Self;

    fn mul(self, rhs: Poly<L>) -> Self::Output {
        Self(self.0 * rhs.clone(), self.1 * rhs)
    }
}

impl<L: LweParams, const D: usize> From<GswCiphertext<L, D>> for GswCiphertextNtt<L, D> {
    fn from(value: GswCiphertext<L, D>) -> Self {
        let GswCiphertext { c00, c01, c10, c11 } = value;
        Self {
            c00: map_boxed_array(c00, L::Ntt::from),
            c01: map_boxed_array(c01, L::Ntt::from),
            c10: map_boxed_array(c10, L::Ntt::from),
            c11: map_boxed_array(c11, L::Ntt::from),
        }
    }
}

impl<L: LweParams, const D: usize> From<GswCiphertextNtt<L, D>> for GswCiphertext<L, D> {
    fn from(value: GswCiphertextNtt<L, D>) -> Self {
        let GswCiphertextNtt { c00, c01, c10, c11 } = value;
        Self {
            c00: map_boxed_array(c00, Into::into),
            c01: map_boxed_array(c01, Into::into),
            c10: map_boxed_array(c10, Into::into),
            c11: map_boxed_array(c11, Into::into),
        }
    }
}

impl<L: LweParams, const D: usize> GswCiphertext<L, D> {
    pub fn from_raw(
        c00: Box<[Poly<L>; D]>,
        c01: Box<[Poly<L>; D]>,
        c10: Box<[Poly<L>; D]>,
        c11: Box<[Poly<L>; D]>,
    ) -> Self {
        Self { c00, c01, c10, c11 }
    }

    pub fn from_scalar(scalar: Field<L>) -> Self {
        let zero: [Poly<L>; D] = array::from_fn(|_| Default::default());

        let mut c00 = Box::new(zero.clone());
        let mut c11 = Box::new(zero.clone());

        let mu_g_vector = <Poly<L> as Decompose<D>>::basis().map(|b| scalar * b);
        for i in 0..D {
            c00[i].add_assign_scalar(mu_g_vector[i]);
            c11[i].add_assign_scalar(mu_g_vector[i]);
        }

        GswCiphertext {
            c00,
            c01: Box::new(zero.clone()),
            c10: Box::new(zero),
            c11,
        }
    }
}

impl<L: LweParams, const D: usize> GswCiphertextNtt<L, D> {
    pub fn from_scalar(scalar: Field<L>) -> Self {
        Self::from(GswCiphertext::from_scalar(scalar))
    }

    pub fn from_raw(
        c00: Box<[L::Ntt; D]>,
        c01: Box<[L::Ntt; D]>,
        c10: Box<[L::Ntt; D]>,
        c11: Box<[L::Ntt; D]>,
    ) -> Self {
        Self { c00, c01, c10, c11 }
    }
}

impl<L: LweParams, const D: usize> Add<Self> for GswCiphertext<L, D> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..D {
            self.c00[i] += &rhs.c00[i];
            self.c01[i] += &rhs.c01[i];
            self.c10[i] += &rhs.c10[i];
            self.c11[i] += &rhs.c11[i];
        }
        self
    }
}

impl<L: LweParams, const D: usize> Sub<Self> for GswCiphertext<L, D> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..D {
            self.c00[i] -= &rhs.c00[i];
            self.c01[i] -= &rhs.c01[i];
            self.c10[i] -= &rhs.c10[i];
            self.c11[i] -= &rhs.c11[i];
        }
        self
    }
}

impl<L: LweParams, const D: usize> Sub<Self> for GswCiphertextNtt<L, D> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..D {
            self.c00[i] -= &rhs.c00[i];
            self.c01[i] -= &rhs.c01[i];
            self.c10[i] -= &rhs.c10[i];
            self.c11[i] -= &rhs.c11[i];
        }
        self
    }
}

/// BFV-RGSW external product
impl<L: LweParams, const D: usize> Mul<&BfvCiphertext<L>> for GswCiphertext<L, D> {
    type Output = BfvCiphertext<L>;

    fn mul(self, rhs: &BfvCiphertext<L>) -> Self::Output {
        let BfvCiphertext(rhs_c0, rhs_c1) = rhs;
        let rhs_d0 = rhs_c0.decompose();
        let rhs_d1 = rhs_c1.decompose();

        let GswCiphertext { c00, c01, c10, c11 } = self;

        let c0a = Poly::inner_prod(&c00, &rhs_d0);
        let c0b = Poly::inner_prod(&c10, &rhs_d1);
        let c1a = Poly::inner_prod(&c01, &rhs_d0);
        let c1b = Poly::inner_prod(&c11, &rhs_d1);

        BfvCiphertext(c0a + c0b, c1a + c1b)
    }
}

/// BFV-RGSW external product
impl<L: LweParams, const D: usize> Mul<&BfvCiphertext<L>> for &GswCiphertextNtt<L, D> {
    type Output = BfvCiphertextNtt<L>;

    fn mul(self, rhs: &BfvCiphertext<L>) -> Self::Output {
        let BfvCiphertext(rhs_c0, rhs_c1) = rhs;
        let rhs_d0 = rhs_c0.decompose().map(L::Ntt::from);
        let rhs_d1 = rhs_c1.decompose().map(L::Ntt::from);

        let GswCiphertextNtt {
            ref c00,
            ref c01,
            ref c10,
            ref c11,
        } = self;

        let c0a = L::Ntt::inner_prod(c00, &rhs_d0);
        let c0b = L::Ntt::inner_prod(c10, &rhs_d1);
        let c1a = L::Ntt::inner_prod(c01, &rhs_d0);
        let c1b = L::Ntt::inner_prod(c11, &rhs_d1);

        BfvCiphertextNtt(c0a + c0b, c1a + c1b)
    }
}

/// BFV-RGSW external product
impl<L: LweParams, const D: usize> Mul<(&[L::Ntt; D], &[L::Ntt; D])> for &GswCiphertextNtt<L, D> {
    type Output = BfvCiphertextNtt<L>;

    fn mul(self, (rhs_d0, rhs_d1): (&[L::Ntt; D], &[L::Ntt; D])) -> Self::Output {
        let GswCiphertextNtt {
            ref c00,
            ref c01,
            ref c10,
            ref c11,
        } = self;

        let c0a = L::Ntt::inner_prod(c00, rhs_d0);
        let c0b = L::Ntt::inner_prod(c10, rhs_d1);
        let c1a = L::Ntt::inner_prod(c01, rhs_d0);
        let c1b = L::Ntt::inner_prod(c11, rhs_d1);

        BfvCiphertextNtt(c0a + c0b, c1a + c1b)
    }
}

/// BFV-RGSW external product
impl<L: LweParams, const D: usize> Mul<&BfvCiphertext<L>> for OneMinus<&GswCiphertextNtt<L, D>> {
    type Output = BfvCiphertextNtt<L>;

    fn mul(self, rhs: &BfvCiphertext<L>) -> Self::Output {
        let BfvCiphertext(rhs_c0, rhs_c1) = rhs;
        let rhs_d0: Box<[_; D]> = map_boxed_array(rhs_c0.decompose(), L::Ntt::from);
        let rhs_d1: Box<[_; D]> = map_boxed_array(rhs_c1.decompose(), L::Ntt::from);

        let g_vec = map_boxed_array(<Poly<L> as Decompose<D>>::basis(), L::Ntt::from_scalar);

        let OneMinus(GswCiphertextNtt {
            ref c00,
            ref c01,
            ref c10,
            ref c11,
        }) = self;

        let om00 = zip(g_vec.as_ref(), c00.as_ref())
            .map(|(g, v)| g.clone() - v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let om10 = c10
            .iter()
            .cloned()
            .map(Neg::neg)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let om01 = c01
            .iter()
            .cloned()
            .map(Neg::neg)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let om11 = zip(g_vec.as_ref(), c11.as_ref())
            .map(|(g, v)| g.clone() - v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let c0a = L::Ntt::inner_prod(&rhs_d0, &om00);
        let c0b = L::Ntt::inner_prod(&rhs_d1, &om10);
        let c1a = L::Ntt::inner_prod(&rhs_d0, &om01);
        let c1b = L::Ntt::inner_prod(&rhs_d1, &om11);

        BfvCiphertextNtt(c0a + c0b, c1a + c1b)
    }
}

/// BFV-RGSW external product. In this version the
impl<L: LweParams, const D: usize> Mul<(&[L::Ntt; D], &[L::Ntt; D])>
    for OneMinus<&GswCiphertextNtt<L, D>>
{
    type Output = BfvCiphertextNtt<L>;

    fn mul(self, (rhs_d0, rhs_d1): (&[L::Ntt; D], &[L::Ntt; D])) -> Self::Output {
        let g_vec = map_boxed_array(<Poly<L> as Decompose<D>>::basis(), L::Ntt::from_scalar);

        let OneMinus(GswCiphertextNtt {
            ref c00,
            ref c01,
            ref c10,
            ref c11,
        }) = self;

        let om00 = zip(g_vec.as_ref(), c00.as_ref())
            .map(|(g, v)| g.clone() - v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let om10 = c10
            .iter()
            .cloned()
            .map(Neg::neg)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let om01 = c01
            .iter()
            .cloned()
            .map(Neg::neg)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let om11 = zip(g_vec.as_ref(), c11.as_ref())
            .map(|(g, v)| g.clone() - v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let c0a = L::Ntt::inner_prod(rhs_d0, &om00);
        let c0b = L::Ntt::inner_prod(rhs_d1, &om10);
        let c1a = L::Ntt::inner_prod(rhs_d0, &om01);
        let c1b = L::Ntt::inner_prod(rhs_d1, &om11);

        BfvCiphertextNtt(c0a + c0b, c1a + c1b)
    }
}

// This generates a fully random ciphertext, so it won't decrypt to anything.
impl<L: LweParams> Distribution<BfvCiphertext<L>> for Standard
where
    Standard: Distribution<Poly<L>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BfvCiphertext<L> {
        BfvCiphertext(self.sample(rng), self.sample(rng))
    }
}

// This generates a fully random ciphertext, so it won't decrypt to anything.
impl<L: LweParams, const D: usize> Distribution<GswCiphertext<L, D>> for Standard
where
    Standard: Distribution<Poly<L>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GswCiphertext<L, D> {
        GswCiphertext {
            c00: Box::new(array::from_fn(|_| self.sample(rng))),
            c01: Box::new(array::from_fn(|_| self.sample(rng))),
            c10: Box::new(array::from_fn(|_| self.sample(rng))),
            c11: Box::new(array::from_fn(|_| self.sample(rng))),
        }
    }
}

// This generates a fully random ciphertext, so it won't decrypt to anything.
impl<L: LwePrivate, const D: usize> Distribution<GswCiphertextNtt<L, D>> for Standard
where
    Standard: Distribution<L::Ntt>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GswCiphertextNtt<L, D> {
        GswCiphertextNtt {
            c00: Box::new(array::from_fn(|_| self.sample(rng))),
            c01: Box::new(array::from_fn(|_| self.sample(rng))),
            c10: Box::new(array::from_fn(|_| self.sample(rng))),
            c11: Box::new(array::from_fn(|_| self.sample(rng))),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn bfv_encrypt_decrypt() {
        let mut rng = thread_rng();
        let plaintext = rng.gen_range(0..Lwe1024::P_MODULUS);
        let mut lwe = Lwe1024::gen_uniform(rng.clone());
        let ciphertext = lwe.encrypt_bfv(plaintext);
        let plaintext_out = lwe.decrypt_bfv(ciphertext);
        assert_eq!(plaintext_out.as_raw_storage()[0], plaintext);
    }

    #[test]
    fn bfv_pt_ct_mul() {
        let mut rng = thread_rng();
        let plaintext = rng.gen_range(0..4);
        let mut lwe = Lwe1024::gen_uniform(rng.clone());
        let ciphertext = lwe.encrypt_bfv(plaintext);

        let mut other_arr = [0u32; Lwe1024::DEGREE];
        let other_val = rng.gen_range(0..4);
        other_arr[0] = other_val;
        let other: Poly<Lwe1024> = other_arr.into();
        let multiplied = ciphertext * other.clone();

        println!("{plaintext} {other_val}");
        let plaintext_out = lwe.decrypt_bfv(multiplied);
        assert_eq!(plaintext_out.as_raw_storage()[0], plaintext * other_val);
    }

    fn gsw<L: LweParams, const D: usize>() {
        let mut rng = thread_rng();
        let gsw_plaintext_val = rng.gen_range(0..4);
        let gsw_plaintext = Poly::from_scalar(L::field(L::Storage::from(gsw_plaintext_val)));

        let mut lwe = L::gen_uniform(rng.clone());
        let gsw_ciphertext: GswCiphertext<L, D> = lwe.encrypt_gsw(gsw_plaintext);

        let bfv_plaintext = rng.gen_range(0..4);
        let bfv_ciphertext = lwe.encrypt_bfv(bfv_plaintext);

        let multiplied = gsw_ciphertext * &bfv_ciphertext;

        let plaintext_out = lwe.decrypt_bfv(multiplied);
        assert_eq!(
            plaintext_out.as_raw_storage()[0],
            L::Storage::from(gsw_plaintext_val * bfv_plaintext),
        );
    }

    #[test]
    fn gsw_q31_p4_l4() {
        gsw::<Lwe1024, 4>();
    }

    #[test]
    fn gsw_q31_p4_l11() {
        gsw::<Lwe1024, 11>();
    }

    #[test]
    fn gsw_q31_p4_l16() {
        gsw::<Lwe1024, 16>();
    }

    #[test]
    fn gsw_q32_p8_l4() {
        gsw::<Lwe1024Q32P8, 4>();
    }

    #[test]
    fn gsw_q56_p8_l4() {
        gsw::<Lwe2048Q56P8, 4>();
    }

    #[test]
    fn switch_modulus() {
        let mut rng = thread_rng();
        let plaintext = Lwe1024::array_from_fn(|_| rng.gen_range(0..Lwe1024::P_MODULUS));
        let mut lwe = Lwe1024::gen_noise(rng.clone());
        let ciphertext =
            lwe.encrypt_bfv_poly(plaintext.map(|x| x * Lwe1024::floor_q_div_p()).into());
        let switched: BfvCiphertext<Lwe1024Q14P4> = ciphertext.switch_modulus();
        let plaintext_out = lwe.decrypt_bfv(switched.clone());
        println!(
            "log₂|ε| = {:.1}",
            (<_ as Into<u64>>::into(
                (lwe.raw_decrypt_bfv(switched)
                    - Poly::<Lwe1024Q14P4>::from(
                        plaintext.map(|x| x * Lwe1024Q14P4::floor_q_div_p())
                    ))
                .inf_norm()
            ) as f32)
                .log2(),
        );
        assert_eq!(plaintext_out, plaintext.into());
    }

    #[test]
    fn gsw_switch_modulus() {
        // This is a more interesting test than `switch_modulus`, because there is more
        // noise in the ciphertext prior to modulus switching.
        let mut rng = thread_rng();
        let gsw_plaintext_val = rng.gen_range(0..4);
        let gsw_plaintext = Poly::from_scalar(Lwe1024::field(gsw_plaintext_val));

        let mut lwe = Lwe1024::gen_noise(rng.clone());
        let gsw_ciphertext: GswCiphertext<Lwe1024, 4> = lwe.encrypt_gsw(gsw_plaintext);

        let bfv_plaintext = rng.gen_range(0..4);
        let bfv_ciphertext = lwe.encrypt_bfv(bfv_plaintext);

        let multiplied = gsw_ciphertext * &bfv_ciphertext;

        let plaintext_out = lwe.decrypt_bfv(multiplied.clone());
        println!(
            "log₂|ε| = {:.1}",
            (<_ as Into<u64>>::into(
                (lwe.raw_decrypt_bfv(multiplied.clone())
                    - Poly::from_scalar(Lwe1024::field(
                        gsw_plaintext_val * bfv_plaintext * Lwe1024::floor_q_div_p()
                    )))
                .inf_norm()
            ) as f32)
                .log2(),
        );
        assert_eq!(
            plaintext_out.as_raw_storage()[0],
            gsw_plaintext_val * bfv_plaintext,
        );
        let switched: BfvCiphertext<Lwe1024Q14P4> = multiplied.switch_modulus();
        let switched_plaintext_out = lwe.decrypt_bfv(switched.clone());
        println!(
            "log₂|ε| = {:.1}",
            (<_ as Into<u64>>::into(
                (lwe.raw_decrypt_bfv(switched)
                    - Poly::from_scalar(Lwe1024Q14P4::field(
                        gsw_plaintext_val * bfv_plaintext * Lwe1024Q14P4::floor_q_div_p()
                    )))
                .inf_norm()
            ) as f32)
                .log2(),
        );
        assert_eq!(
            switched_plaintext_out.as_raw_storage()[0],
            gsw_plaintext_val * bfv_plaintext,
        );
    }
}
