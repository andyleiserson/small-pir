pub(crate) mod residue_ntt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use concrete_ntt::{prime32::Plan as Prime32Plan, prime64::Plan as Prime64Plan};
use delegate::delegate;
use funty::Integral;
use rand::{distributions::Standard, prelude::Distribution, Rng};
pub use residue_ntt::ResidueNtt;

use crate::{field::Field, lwe::LwePrivate, pir::PirBackend, Decompose, InnerProduct};

pub trait Plan: 'static {
    type Storage;

    fn fwd(&self, buf: &mut [Self::Storage]);
    fn inv(&self, buf: &mut [Self::Storage]);

    fn mul_assign_normalize(&self, lhs: &mut [Self::Storage], rhs: &[Self::Storage]);
    fn mul_accumulate(
        &self,
        acc: &mut [Self::Storage],
        lhs: &[Self::Storage],
        rhs: &[Self::Storage],
    );
    fn normalize(&self, values: &mut [Self::Storage]);
}

// We have to write the methods twice (as opposed to write them once in a declarative
// macro, and call that macro from the two places the methods appear here), because proc
// macros are expanded before declarative macros.
macro_rules! plan_trait_impl {
    ($plan:ty, $integer:ty) => {
        impl Plan for $plan {
            type Storage = $integer;

            delegate! {
                #[through($plan)]
                to self {
                    fn fwd(&self, buf: &mut [Self::Storage]);
                    fn inv(&self, buf: &mut [Self::Storage]);

                    fn mul_assign_normalize(
                        &self,
                        lhs: &mut [Self::Storage],
                        rhs: &[Self::Storage],
                    );
                    fn mul_accumulate(
                        &self,
                        acc: &mut [Self::Storage],
                        lhs: &[Self::Storage],
                        rhs: &[Self::Storage],
                    );
                    fn normalize(&self, values: &mut [Self::Storage]);
                }
            }
        }
    };
}

plan_trait_impl!(Prime32Plan, u32);
plan_trait_impl!(Prime64Plan, u64);

#[derive(Clone, Debug)]
#[repr(align(64))]
pub struct Poly<L: LwePrivate>(L::Array);

impl<L: LwePrivate> Default for Poly<L> {
    fn default() -> Self {
        Self(L::array_zero())
    }
}

// It is tempting to write these generically as `impl<L> From<L::Array> for Poly<L>`,
// but that is not allowed because it would overlap with `impl From<T> for T` if
// `L::Array` is the same as `L`.
impl<L> From<[u32; 1024]> for Poly<L>
where
    L: LwePrivate<Array = [u32; 1024]>,
{
    fn from(value: [u32; 1024]) -> Self {
        Self(value)
    }
}

impl<L> From<[u64; 2048]> for Poly<L>
where
    L: LwePrivate<Array = [u64; 2048]>,
{
    fn from(value: [u64; 2048]) -> Self {
        Self(value)
    }
}

impl<L: LwePrivate> PartialEq for Poly<L> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<L: LwePrivate> Eq for Poly<L> {}

impl<L: LwePrivate> Distribution<Poly<L>> for Standard
where
    Standard: Distribution<Field<L>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly<L> {
        L::array_from_fn(|_| self.sample(rng).to_raw()).into()
    }
}

impl<L: LwePrivate> Add<&Self> for Poly<L> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let lhs_slice = self.0.as_ref();
        let rhs_slice = rhs.0.as_ref();
        Self(L::array_from_fn(|i| {
            (L::field(lhs_slice[i]) + L::field(rhs_slice[i])).to_raw()
        }))
    }
}

impl<L: LwePrivate> Add<Self> for Poly<L> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<L: LwePrivate> Add<&Self> for Ntt<L> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let lhs_slice = self.0.as_ref();
        let rhs_slice = rhs.0.as_ref();
        Self(L::array_from_fn(|i| {
            (L::field(lhs_slice[i]) + L::field(rhs_slice[i])).to_raw()
        }))
    }
}

impl<L: LwePrivate> Add<Self> for Ntt<L> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<L: LwePrivate> AddAssign<&Self> for Poly<L> {
    fn add_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.0.as_mut();
        let rhs_slice = rhs.0.as_ref();
        for i in 0..L::DEGREE {
            lhs_slice[i] = (L::field(lhs_slice[i]) + L::field(rhs_slice[i])).to_raw();
        }
    }
}

impl<L: LwePrivate> AddAssign<&Self> for Ntt<L> {
    fn add_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.0.as_mut();
        let rhs_slice = rhs.0.as_ref();
        for i in 0..L::DEGREE {
            lhs_slice[i] = (L::field(lhs_slice[i]) + L::field(rhs_slice[i])).to_raw();
        }
    }
}

impl<L: LwePrivate> AddAssign<Self> for Poly<L> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<L: LwePrivate> AddAssign<Self> for Ntt<L> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<L: LwePrivate> Sub<&Self> for Poly<L> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let lhs_slice = self.0.as_ref();
        let rhs_slice = rhs.0.as_ref();
        Self(L::array_from_fn(|i| {
            (L::field(lhs_slice[i]) - L::field(rhs_slice[i])).to_raw()
        }))
    }
}

impl<L: LwePrivate> Sub<Self> for Poly<L> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        <Poly<L> as Sub<&Self>>::sub(self, &rhs)
    }
}

impl<L: LwePrivate> SubAssign<&Self> for Poly<L> {
    fn sub_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.0.as_mut();
        let rhs_slice = rhs.0.as_ref();
        for i in 0..L::DEGREE {
            lhs_slice[i] = (L::field(lhs_slice[i]) - L::field(rhs_slice[i])).to_raw();
        }
    }
}

impl<L: LwePrivate> Sub<&Ntt<L>> for &Ntt<L> {
    type Output = Ntt<L>;

    fn sub(self, rhs: &Ntt<L>) -> Self::Output {
        let lhs_slice = self.0.as_ref();
        let rhs_slice = rhs.0.as_ref();
        Ntt(L::array_from_fn(|i| {
            (L::field(lhs_slice[i]) - L::field(rhs_slice[i])).to_raw()
        }))
    }
}

impl<L: LwePrivate> Sub<&Self> for Ntt<L> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        <Ntt<L> as SubAssign<&Ntt<L>>>::sub_assign(&mut self, rhs);
        self
    }
}

impl<L: LwePrivate> Sub<Self> for Ntt<L> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        <Ntt<L> as SubAssign<&Ntt<L>>>::sub_assign(&mut self, &rhs);
        self
    }
}

impl<L: LwePrivate> SubAssign<&Self> for Ntt<L> {
    fn sub_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.0.as_mut();
        let rhs_slice = rhs.0.as_ref();
        for i in 0..L::DEGREE {
            lhs_slice[i] = (L::field(lhs_slice[i]) - L::field(rhs_slice[i])).to_raw();
        }
    }
}

impl<L: LwePrivate> Neg for &Poly<L> {
    type Output = Poly<L>;

    fn neg(self) -> Self::Output {
        L::array_from_fn(|i| (-L::field(self.0.as_ref()[i])).to_raw()).into()
    }
}

impl<L: LwePrivate> Neg for &Ntt<L> {
    type Output = Ntt<L>;

    fn neg(self) -> Self::Output {
        Ntt(L::array_from_fn(|i| {
            (-L::field(self.0.as_ref()[i])).to_raw()
        }))
    }
}

impl<L: LwePrivate> Neg for Ntt<L> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        <&Ntt<L> as Neg>::neg(&self)
    }
}

impl<L: LwePrivate> Mul<Field<L>> for Poly<L> {
    type Output = Poly<L>;

    fn mul(self, rhs: Field<L>) -> Self {
        let lhs = self.0.as_ref();
        Self(L::array_from_fn(|i| (L::field(lhs[i]) * rhs).to_raw()))
    }
}

impl<L: LwePrivate> Poly<L> {
    pub fn from_raw(raw: L::Array) -> Self {
        Self(raw)
    }

    pub fn as_raw_storage(&self) -> &[L::Storage] {
        self.0.as_ref()
    }

    pub fn as_raw_storage_mut(&mut self) -> &mut [L::Storage] {
        self.0.as_mut()
    }

    pub fn from_scalar(scalar: Field<L>) -> Self {
        let mut result = Self::default();
        result.0.as_mut()[0] = scalar.to_raw();
        result
    }

    pub fn add_assign_scalar(&mut self, scalar: Field<L>) {
        self.0.as_mut()[0] = (L::field(self.0.as_ref()[0]) + scalar).to_raw();
    }

    pub fn round(self, round_to: L::Storage) -> Self {
        L::array_from_fn(|i| {
            let val = self.0.as_ref()[i];
            if val < round_to >> 1 {
                L::Storage::ZERO
            } else {
                let val2 = (val - (round_to >> 1)) / round_to;
                if val2 == L::P_MODULUS - L::Storage::ONE {
                    L::Storage::ZERO
                } else {
                    val2 + L::Storage::ONE
                }
            }
        })
        .into()
    }

    /// Rotate the polynomial, mapping $x^j \mapsto x^(j + k)$.
    pub fn rotate(&mut self, count: isize) {
        match usize::try_from(count) {
            Ok(count) => {
                self.0.as_mut().rotate_right(count);
                for x in self.0.as_mut()[0..count].iter_mut() {
                    *x = (-Field::<L>::new(*x)).to_raw();
                }
            }
            Err(_) => {
                let count = usize::try_from(-count).unwrap();
                self.0.as_mut().rotate_left(count);
                for x in self.0.as_mut()[L::DEGREE - count..L::DEGREE].iter_mut() {
                    *x = (-Field::<L>::new(*x)).to_raw();
                }
            }
        }
    }

    pub fn inf_norm(&self) -> L::Storage {
        self.0
            .as_ref()
            .iter()
            .fold(L::Storage::default(), |norm, v| {
                if *v < L::Q_MODULUS >> 1 {
                    norm.max(*v)
                } else {
                    norm.max(L::Q_MODULUS - *v)
                }
            })
    }

    pub fn switch_modulus<LPrime>(&self) -> Poly<LPrime>
    where
        LPrime: LwePrivate,
        LPrime::Storage: TryFrom<L::OpStorage>,
        L::OpStorage: TryFrom<LPrime::Storage>,
    {
        assert!(LPrime::Q_BITS <= L::Q_BITS);
        let l_modulus = L::OpStorage::from(L::Q_MODULUS);
        let half_l_modulus = l_modulus / L::OpStorage::from(2);
        let l_prime_modulus = L::OpStorage::try_from(LPrime::Q_MODULUS).ok().unwrap();
        LPrime::array_from_fn(|i| {
            let product = L::OpStorage::from(self.0.as_ref()[i]) * l_prime_modulus;
            let quotient = product / l_modulus;
            let remainder = product % l_modulus;
            let rounded = if remainder > half_l_modulus {
                if quotient == l_prime_modulus - L::OpStorage::from(1) {
                    L::OpStorage::from(0)
                } else {
                    quotient + L::OpStorage::from(1)
                }
            } else {
                quotient
            };
            LPrime::Storage::try_from(rounded).ok().unwrap()
        })
        .into()
    }
}

#[derive(Clone, Debug)]
#[repr(align(64))]
pub struct Ntt<L: LwePrivate>(L::Array);

impl<L: LwePrivate> Default for Ntt<L> {
    fn default() -> Self {
        Self(L::array_zero())
    }
}

pub trait NttTrait<L: LwePrivate> {
    fn from_scalar(scalar: Field<L>) -> Self;
}

impl<L: LwePrivate> Ntt<L> {
    pub fn into_raw(self) -> L::Array {
        self.0
    }

    pub fn as_raw_storage(&self) -> &[L::Storage] {
        self.0.as_ref()
    }

    pub fn as_raw_storage_mut(&mut self) -> &mut [L::Storage] {
        self.0.as_mut()
    }
}

impl<L: LwePrivate> NttTrait<L> for Ntt<L> {
    fn from_scalar(scalar: Field<L>) -> Self {
        Self(L::array_from_fn(|_| scalar.to_raw()))
    }
}

impl<L: LwePrivate> Distribution<Ntt<L>> for Standard
where
    Standard: Distribution<Field<L>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Ntt<L> {
        Ntt(L::array_from_fn(|_| self.sample(rng).to_raw()))
    }
}

impl<L: LwePrivate> Mul<Self> for Poly<L> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs = L::Ntt::from(self);
        let rhs = L::Ntt::from(rhs);
        let result = lhs * rhs;
        result.into()
    }
}

impl<L: LwePrivate> From<Poly<L>> for Ntt<L> {
    fn from(mut value: Poly<L>) -> Self {
        L::plan().fwd(value.0.as_mut());
        Self(value.0)
    }
}

// TODO normalization hygiene: this is only correct if the source `Ntt` has been
// normalized. A round trip using `Poly::from(Ntt::from(poly))` is not correct, which
// is bad.
impl<L: LwePrivate> From<Ntt<L>> for Poly<L> {
    fn from(mut value: Ntt<L>) -> Self {
        L::plan().inv(value.0.as_mut());
        Self(value.0)
    }
}

impl<L: LwePrivate> Mul<Self> for Ntt<L> {
    type Output = Self;

    fn mul(self, mut rhs: Self) -> Self::Output {
        L::plan().mul_assign_normalize(rhs.0.as_mut(), self.0.as_ref());
        Self(rhs.0)
    }
}

impl<L: LwePrivate> Mul<Poly<L>> for Ntt<L> {
    type Output = Self;

    fn mul(self, mut rhs: Poly<L>) -> Self::Output {
        L::plan().fwd(rhs.0.as_mut());
        L::plan().mul_assign_normalize(rhs.0.as_mut(), self.0.as_ref());
        Self(rhs.0)
    }
}

impl<L: LwePrivate, const D: usize> Decompose<D> for Poly<L> {
    type Basis = <Field<L> as Decompose<D>>::Basis;

    #[inline(never)]
    fn decompose(&self) -> Box<[Self; D]> {
        let log_beta = L::Q_BITS.div_ceil(D);
        let beta_mask = (L::Storage::ONE << log_beta) - L::Storage::ONE;
        let mut result: Box<[Self; D]> = vec![Default::default(); D].try_into().unwrap();
        for i in 0..L::DEGREE {
            let mut val = self.0.as_ref()[i];
            for k in 0..D {
                result[k].as_raw_storage_mut()[i] = val & beta_mask;
                val >>= log_beta;
            }
        }
        result
    }

    fn basis() -> <Field<L> as Decompose<D>>::Basis {
        Field::<L>::basis()
    }
}

impl<L: LwePrivate> InnerProduct for Poly<L> {
    fn inner_prod<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        // TODO: it would be faster to do the NTTs all at once and delegate to the
        // NTT inner product impl.
        let lhs: Box<[_; N]> = lhs
            .iter()
            .map(|poly| L::Ntt::from(poly.clone()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let rhs: Box<[_; N]> = rhs
            .iter()
            .map(|poly| L::Ntt::from(poly.clone()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let result = L::Ntt::inner_prod(&lhs, &rhs);
        result.into()
    }
}

impl<L: LwePrivate<Ntt = Ntt<L>>> InnerProduct for Ntt<L> {
    fn inner_prod<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        let mut result = Default::default();
        L::PirBackend::inner_prod(lhs, rhs, &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use proptest::{prelude::*, proptest};

    use super::*;
    use crate::lwe::Lwe1024;

    fn to_poly<L: LwePrivate>(src: Vec<Field<L>>) -> Poly<L> {
        L::array_from_fn(|i| src[i].to_raw()).into()
    }

    impl<L: LwePrivate> Arbitrary for Poly<L>
    where
        Field<L>: Arbitrary,
    {
        type Parameters = ();
        type Strategy = prop::strategy::Map<
            prop::collection::VecStrategy<<Field<L> as Arbitrary>::Strategy>,
            fn(Vec<Field<L>>) -> Poly<L>,
        >;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            prop::collection::vec(any::<Field<L>>(), L::DEGREE).prop_map(to_poly)
        }
    }

    fn test_decomposition<const D: usize>(input: Poly<Lwe1024>) {
        assert_eq!(
            Poly::<Lwe1024>::inner_prod(
                &<Poly<Lwe1024> as Decompose<D>>::decompose(&input),
                &<Poly<Lwe1024> as Decompose<D>>::basis().map(Poly::from_scalar)
            ),
            input,
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]
        #[test]
        fn decompose(
            poly in any::<Poly<Lwe1024>>(),
            degree in prop::sample::select(vec![2, 3, 4, 6, 8, 11, 16]),
        ) {
            match degree {
                2 => test_decomposition::<2>(poly),
                3 => test_decomposition::<3>(poly),
                4 => test_decomposition::<4>(poly),
                6 => test_decomposition::<6>(poly),
                8 => test_decomposition::<8>(poly),
                11 => test_decomposition::<11>(poly),
                16 => test_decomposition::<16>(poly),
                _ => unreachable!()
            }
        }
    }
}
