use std::{
    array,
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub, SubAssign},
    sync::OnceLock,
};

use concrete_ntt::prime32::Plan as Prime32Plan;
use rand::Rng;
use rand_distr::{Distribution, Standard};

use crate::{
    field::Field,
    lwe::LwePrivate,
    pir::PirBackend,
    poly::{NttTrait, Plan, Poly},
    InnerProduct,
};

pub trait Basis {
    const Q0: u32;
    const Q1: u32;
    const B0: i128;
    const B1: i128;
    fn plan0() -> &'static Prime32Plan;
    fn plan1() -> &'static Prime32Plan;
}

macro_rules! basis_impl {
    ($ty:ident, $q_modulus_a:expr, $q_modulus_b:expr, $b_coeff_a:expr, $b_coeff_b:expr $(,)?) => {
        #[derive(Clone, Debug)]
        pub struct $ty;

        impl Basis for $ty {
            const Q0: u32 = $q_modulus_a;
            const Q1: u32 = $q_modulus_b;
            const B0: i128 = $b_coeff_a;
            const B1: i128 = $b_coeff_b;
            fn plan0() -> &'static Prime32Plan {
                static PLAN: OnceLock<Prime32Plan> = OnceLock::new();
                PLAN.get_or_init(|| <Prime32Plan>::try_new(2048, $q_modulus_a).unwrap())
            }
            fn plan1() -> &'static Prime32Plan {
                static PLAN: OnceLock<Prime32Plan> = OnceLock::new();
                PLAN.get_or_init(|| <Prime32Plan>::try_new(2048, $q_modulus_b).unwrap())
            }
        }
    };
}

// Moduli with 32 < log q <= 62 are represented using the CRT.  F0 and F1 are the two
// prime factors. B0 and B1 are the Bezout coefficients used to convert back from
// residue form. They can be computed from the primes using the `xgcd` function in Sage.

pub const Q56_F0: u32 = 0xfff_0001;
pub const Q56_F1: u32 = 0xffe_e001;

pub const Q56_B0: i128 = -32759;
pub const Q56_B1: i128 = 32760;

pub const Q62_F0: u32 = 0x7ffe_9001;
pub const Q62_F1: u32 = 0x7ffe_6001;

pub const Q62_B0: i128 = -174754;
pub const Q62_B1: i128 = 174755;

basis_impl!(Q56Basis, Q56_F0, Q56_F1, Q56_B0, Q56_B1);
basis_impl!(Q62Basis, Q62_F0, Q62_F1, Q62_B0, Q62_B1);

pub trait ResidueLwe<N>: LwePrivate<Storage = u64, Array = [u64; 2048], Ntt = N> {}

impl<L: LwePrivate<Storage = u64, Array = [u64; 2048]>> ResidueLwe<L::Ntt> for L {}

#[derive(Clone, Debug)]
#[repr(align(64))]
pub struct ResidueNtt<L: ResidueLwe<Self>, B: Basis>([u32; 2048], [u32; 2048], PhantomData<(L, B)>);

impl<L: ResidueLwe<Self>, B: Basis> Default for ResidueNtt<L, B> {
    fn default() -> Self {
        Self([0; 2048], [0; 2048], PhantomData)
    }
}

impl<L: ResidueLwe<Self>, B: Basis> ResidueNtt<L, B> {
    pub fn as_raw_storage(&self) -> [&[u32]; 2] {
        [self.0.as_ref(), self.1.as_ref()]
    }

    pub fn as_raw_storage_mut(&mut self) -> [&mut [u32]; 2] {
        [self.0.as_mut(), self.1.as_mut()]
    }

    /*
    pub fn from_scalar(scalar: Field<L>) -> Self {
        Self(L::array_from_fn(|_| scalar.to_raw()))
    }
    */
}

impl<L: ResidueLwe<Self>, B: Basis> NttTrait<L> for ResidueNtt<L, B> {
    fn from_scalar(scalar: Field<L>) -> Self {
        let mod_q0 = (scalar.to_raw() % u64::from(B::Q0))
            .try_into()
            .ok()
            .unwrap();
        let mod_q1 = (scalar.to_raw() % u64::from(B::Q1))
            .try_into()
            .ok()
            .unwrap();
        Self(
            array::from_fn(|_| mod_q0),
            array::from_fn(|_| mod_q1),
            PhantomData,
        )
    }
}

impl<L: ResidueLwe<Self>, B: Basis> From<Poly<L>> for ResidueNtt<L, B> {
    fn from(value: Poly<L>) -> Self {
        let mut mod_q0 = array::from_fn(|i| {
            (value.as_raw_storage()[i] % u64::from(B::Q0))
                .try_into()
                .ok()
                .unwrap()
        });
        let mut mod_q1 = array::from_fn(|i| {
            (value.as_raw_storage()[i] % u64::from(B::Q1))
                .try_into()
                .ok()
                .unwrap()
        });
        B::plan0().fwd(&mut mod_q0);
        B::plan1().fwd(&mut mod_q1);
        Self(mod_q0, mod_q1, PhantomData)
    }
}

impl<L: ResidueLwe<ResidueNtt<L, B>>, B: Basis> From<ResidueNtt<L, B>> for Poly<L> {
    fn from(mut value: ResidueNtt<L, B>) -> Self {
        B::plan0().normalize(&mut value.0);
        B::plan1().normalize(&mut value.1);
        B::plan0().inv(&mut value.0);
        B::plan1().inv(&mut value.1);
        Self(L::array_from_fn(|i| {
            (i128::from(value.0[i]) * i128::from(B::Q1) * B::B1
                + i128::from(value.1[i]) * i128::from(B::Q0) * B::B0)
                .rem_euclid(i128::from(B::Q0) * i128::from(B::Q1))
                .try_into()
                .unwrap()
        }))
    }
}

fn mod_add(a: u32, b: u32, q: u32) -> u32 {
    let tmp = a + b;
    if tmp >= q {
        tmp - q
    } else {
        tmp
    }
}

impl<L: ResidueLwe<Self>, B: Basis> Add<&Self> for ResidueNtt<L, B> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let lhs0_slice = self.0.as_ref();
        let lhs1_slice = self.1.as_ref();
        let rhs0_slice = rhs.0.as_ref();
        let rhs1_slice = rhs.1.as_ref();
        Self(
            array::from_fn(|i| mod_add(lhs0_slice[i], rhs0_slice[i], B::Q0)),
            array::from_fn(|i| mod_add(lhs1_slice[i], rhs1_slice[i], B::Q1)),
            PhantomData,
        )
    }
}

impl<L: ResidueLwe<Self>, B: Basis> Add<Self> for ResidueNtt<L, B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

fn mod_sub(a: u32, b: u32, q: u32) -> u32 {
    let tmp = q + a - b;
    if tmp >= q {
        tmp - q
    } else {
        tmp
    }
}

impl<L: ResidueLwe<Self>, B: Basis> SubAssign<&Self> for ResidueNtt<L, B> {
    fn sub_assign(&mut self, rhs: &Self) {
        let lhs0_slice = self.0.as_mut();
        let lhs1_slice = self.1.as_mut();
        let rhs0_slice = rhs.0.as_ref();
        let rhs1_slice = rhs.1.as_ref();
        for i in 0..L::DEGREE {
            lhs0_slice[i] = mod_sub(lhs0_slice[i], rhs0_slice[i], B::Q0);
            lhs1_slice[i] = mod_sub(lhs1_slice[i], rhs1_slice[i], B::Q1);
        }
    }
}

impl<L: ResidueLwe<Self>, B: Basis> Sub<&Self> for ResidueNtt<L, B> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        <ResidueNtt<L, B> as SubAssign<&ResidueNtt<L, B>>>::sub_assign(&mut self, rhs);
        self
    }
}

impl<L: ResidueLwe<Self>, B: Basis> Sub<Self> for ResidueNtt<L, B> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        <ResidueNtt<L, B> as SubAssign<&ResidueNtt<L, B>>>::sub_assign(&mut self, &rhs);
        self
    }
}

fn mod_neg(v: u32, q: u32) -> u32 {
    if v != 0 {
        q - v
    } else {
        0
    }
}

impl<L: ResidueLwe<ResidueNtt<L, B>>, B: Basis> Neg for &ResidueNtt<L, B> {
    type Output = ResidueNtt<L, B>;

    fn neg(self) -> Self::Output {
        ResidueNtt(
            array::from_fn(|i| mod_neg(self.0[i], B::Q0)),
            array::from_fn(|i| mod_neg(self.1[i], B::Q1)),
            PhantomData,
        )
    }
}

impl<L: ResidueLwe<Self>, B: Basis> Neg for ResidueNtt<L, B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        <&ResidueNtt<L, B> as Neg>::neg(&self)
    }
}

impl<L: ResidueLwe<Self>, B: Basis> Mul<Self> for ResidueNtt<L, B> {
    type Output = Self;

    fn mul(self, mut rhs: Self) -> Self::Output {
        B::plan0().mul_assign(&mut rhs.0, &self.0, &mut [0; 2048]);
        B::plan1().mul_assign(&mut rhs.1, &self.1, &mut [0; 2048]);
        rhs
    }
}

impl<L: ResidueLwe<Self>, B: Basis> InnerProduct for ResidueNtt<L, B> {
    fn inner_prod<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        let mut result: Self = Default::default();
        L::PirBackend::inner_prod(lhs, rhs, &mut result);
        result
    }
}

impl<L: ResidueLwe<ResidueNtt<L, B>>, B: Basis> Distribution<ResidueNtt<L, B>> for Standard
where
    Standard: Distribution<Field<L>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ResidueNtt<L, B> {
        ResidueNtt(
            array::from_fn(|_| rng.sample(rand::distributions::Uniform::from(0..B::Q0))),
            array::from_fn(|_| rng.sample(rand::distributions::Uniform::from(0..B::Q1))),
            PhantomData,
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::{prelude::*, proptest};
    use rand::thread_rng;

    use super::{Basis, Q56Basis};
    use crate::{
        field::Field,
        lwe::{Lwe2048Q56P8, LwePrivate},
        poly::{NttTrait, Poly},
        Decompose, InnerProduct,
    };

    type Lwe = Lwe2048Q56P8;
    type ResidueNtt = super::ResidueNtt<Lwe, Q56Basis>;

    #[test]
    fn ntt() {
        let one_ntt = ResidueNtt::from_scalar(Field::ONE);
        assert_eq!(one_ntt.as_raw_storage()[0], &[1; 2048]);
        assert_eq!(one_ntt.as_raw_storage()[1], &[1; 2048]);

        let one_poly = Poly::from(one_ntt);
        assert_eq!(one_poly.as_raw_storage()[0], 1);
        assert_eq!(&one_poly.as_raw_storage()[1..], &[0; 2047]);

        let mut rng = thread_rng();
        let rand_scalar = rng.gen_range(0..Lwe::Q_MODULUS);
        let rand_scalar_ntt = ResidueNtt::from_scalar(Field::new(rand_scalar));
        assert_eq!(
            rand_scalar_ntt.as_raw_storage()[0],
            &[(rand_scalar % Q56Basis::Q0 as u64) as u32; 2048]
        );
        assert_eq!(
            rand_scalar_ntt.as_raw_storage()[1],
            &[(rand_scalar % Q56Basis::Q1 as u64) as u32; 2048]
        );

        let rand_scalar_poly = Poly::from(rand_scalar_ntt);
        assert_eq!(
            rand_scalar_poly.as_raw_storage()[0],
            Field::<Lwe>::new(rand_scalar).to_raw()
        );
        assert_eq!(&rand_scalar_poly.as_raw_storage()[1..], &[0; 2047]);

        let mut rng = thread_rng();
        let rand: Poly<Lwe> = rng.gen();
        let rand_ntt = ResidueNtt::from(rand.clone());
        let rand_poly = Poly::from(rand_ntt);
        assert_eq!(
            rand_poly.as_raw_storage()[0],
            rand.clone().as_raw_storage()[0]
        );
        assert_eq!(rand_poly.as_raw_storage(), rand.as_raw_storage());
    }

    fn test_decomposition<const D: usize>(input: Poly<Lwe>) {
        let dec = <Poly<Lwe> as Decompose<D>>::decompose(&input);
        assert_eq!(
            Poly::<Lwe>::inner_prod(
                &dec,
                &<Poly<Lwe> as Decompose<D>>::basis().map(Poly::from_scalar)
            ),
            input,
        );
    }

    #[test]
    fn decompose_simple() {
        assert_eq!(
            Poly::<Lwe>::default().decompose().as_ref(),
            &[Poly::<Lwe>::default(), Poly::<Lwe>::default()]
        );
        assert_eq!(
            Poly::<Lwe>::from_scalar(Field::ONE).decompose().as_ref(),
            &[Poly::<Lwe>::from_scalar(Field::ONE), Poly::<Lwe>::default(),]
        );
        test_decomposition::<2>(Poly::<Lwe>::from_scalar(Field::ONE));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]
        #[test]
        fn decompose(
            poly in any::<Poly<Lwe>>(),
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
