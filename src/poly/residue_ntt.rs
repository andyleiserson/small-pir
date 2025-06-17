use std::{
    array,
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub, SubAssign},
    sync::LazyLock,
};

use concrete_ntt::prime32::Plan as Prime32Plan;
use rand::Rng;
use rand_distr::{Distribution, Standard};

use crate::{
    field::Field,
    lwe::{LwePrivate, Q56_F0, Q56_F1},
    pir::PirBackend,
    poly::{NttTrait, Poly},
    InnerProduct,
};

const Q0: u32 = Q56_F0;
const Q1: u32 = Q56_F1;

// Bezout coefficients for Q0 and Q1
const B0: i128 = -32759;
const B1: i128 = 32760;

pub(crate) static PLAN0: LazyLock<Prime32Plan> =
    LazyLock::new(|| Prime32Plan::try_new(2048, Q0).unwrap());
pub(crate) static PLAN1: LazyLock<Prime32Plan> =
    LazyLock::new(|| Prime32Plan::try_new(2048, Q1).unwrap());

pub trait ResidueLwe:
    LwePrivate<Storage = u64, Array = [u64; 2048], Ntt = ResidueNtt<Self>>
{
}

impl<L: LwePrivate<Storage = u64, Array = [u64; 2048], Ntt = ResidueNtt<Self>>> ResidueLwe for L {}

#[derive(Clone, Debug)]
#[repr(align(64))]
pub struct ResidueNtt<L: ResidueLwe>([u32; 2048], [u32; 2048], PhantomData<L>);

impl<L: ResidueLwe> Default for ResidueNtt<L> {
    fn default() -> Self {
        Self([0; 2048], [0; 2048], PhantomData)
    }
}

impl<L: ResidueLwe> ResidueNtt<L> {
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

impl<L: ResidueLwe> NttTrait<L> for ResidueNtt<L> {
    fn from_scalar(scalar: Field<L>) -> Self {
        let mod_q0 = (scalar.to_raw() % u64::from(Q0)).try_into().ok().unwrap();
        let mod_q1 = (scalar.to_raw() % u64::from(Q1)).try_into().ok().unwrap();
        Self(
            array::from_fn(|_| mod_q0),
            array::from_fn(|_| mod_q1),
            PhantomData,
        )
    }
}

impl<L: ResidueLwe> From<Poly<L>> for ResidueNtt<L> {
    fn from(value: Poly<L>) -> Self {
        let mut mod_q0 = array::from_fn(|i| {
            (value.as_raw_storage()[i] % u64::from(Q0))
                .try_into()
                .ok()
                .unwrap()
        });
        let mut mod_q1 = array::from_fn(|i| {
            (value.as_raw_storage()[i] % u64::from(Q1))
                .try_into()
                .ok()
                .unwrap()
        });
        PLAN0.fwd(&mut mod_q0);
        PLAN1.fwd(&mut mod_q1);
        Self(mod_q0, mod_q1, PhantomData)
    }
}

// TODO normalization hygiene: this is only correct if the source `Ntt` has been
// normalized. A round trip using `Poly::from(Ntt::from(poly))` is not correct, which
// is bad.
impl<L: ResidueLwe> From<ResidueNtt<L>> for Poly<L> {
    fn from(mut value: ResidueNtt<L>) -> Self {
        PLAN0.inv(&mut value.0);
        PLAN1.inv(&mut value.1);
        Self(L::array_from_fn(|i| {
            (i128::from(value.0[i]) * i128::from(Q1) * B1
                + i128::from(value.1[i]) * i128::from(Q0) * B0)
                .rem_euclid(i128::from(Q0) * i128::from(Q1))
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

impl<L: ResidueLwe> Add<&Self> for ResidueNtt<L> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let lhs0_slice = self.0.as_ref();
        let lhs1_slice = self.1.as_ref();
        let rhs0_slice = rhs.0.as_ref();
        let rhs1_slice = rhs.1.as_ref();
        Self(
            array::from_fn(|i| mod_add(lhs0_slice[i], rhs0_slice[i], Q0)),
            array::from_fn(|i| mod_add(lhs1_slice[i], rhs1_slice[i], Q1)),
            PhantomData,
        )
    }
}

impl<L: ResidueLwe> Add<Self> for ResidueNtt<L> {
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

impl<L: ResidueLwe> SubAssign<&Self> for ResidueNtt<L> {
    fn sub_assign(&mut self, rhs: &Self) {
        let lhs0_slice = self.0.as_mut();
        let lhs1_slice = self.1.as_mut();
        let rhs0_slice = rhs.0.as_ref();
        let rhs1_slice = rhs.1.as_ref();
        for i in 0..L::DEGREE {
            lhs0_slice[i] = mod_sub(lhs0_slice[i], rhs0_slice[i], Q0);
            lhs1_slice[i] = mod_sub(lhs1_slice[i], rhs1_slice[i], Q1);
        }
    }
}

impl<L: ResidueLwe> Sub<&Self> for ResidueNtt<L> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        <ResidueNtt<L> as SubAssign<&ResidueNtt<L>>>::sub_assign(&mut self, rhs);
        self
    }
}

impl<L: ResidueLwe> Sub<Self> for ResidueNtt<L> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        <ResidueNtt<L> as SubAssign<&ResidueNtt<L>>>::sub_assign(&mut self, &rhs);
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

impl<L: ResidueLwe> Neg for &ResidueNtt<L> {
    type Output = ResidueNtt<L>;

    fn neg(self) -> Self::Output {
        ResidueNtt(
            array::from_fn(|i| mod_neg(self.0[i], Q0)),
            array::from_fn(|i| mod_neg(self.1[i], Q1)),
            PhantomData,
        )
    }
}

impl<L: ResidueLwe> Neg for ResidueNtt<L> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        <&ResidueNtt<L> as Neg>::neg(&self)
    }
}

impl<L: ResidueLwe> Mul<Self> for ResidueNtt<L> {
    type Output = Self;

    fn mul(self, mut rhs: Self) -> Self::Output {
        PLAN0.mul_assign_normalize(rhs.0.as_mut(), self.0.as_ref());
        PLAN1.mul_assign_normalize(rhs.1.as_mut(), self.1.as_ref());
        rhs
    }
}

impl<L: ResidueLwe> InnerProduct for ResidueNtt<L> {
    fn inner_prod<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        let mut result: Self = Default::default();
        L::PirBackend::inner_prod(lhs, rhs, &mut result);
        result
    }
}

impl<L: ResidueLwe> Distribution<ResidueNtt<L>> for Standard
where
    Standard: Distribution<Field<L>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ResidueNtt<L> {
        ResidueNtt(
            array::from_fn(|_| rng.sample(rand::distributions::Uniform::from(0..Q0))),
            array::from_fn(|_| rng.sample(rand::distributions::Uniform::from(0..Q1))),
            PhantomData,
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::{prelude::*, proptest};
    use rand::thread_rng;

    use super::*;
    use crate::{lwe::Lwe2048, Decompose};

    #[test]
    fn ntt() {
        let one_ntt = ResidueNtt::<Lwe2048>::from_scalar(Field::ONE);
        assert_eq!(one_ntt.as_raw_storage()[0], &[1; 2048]);
        assert_eq!(one_ntt.as_raw_storage()[1], &[1; 2048]);

        let one_poly = Poly::from(one_ntt);
        assert_eq!(one_poly.as_raw_storage()[0], 2048); // TODO: normalization hygiene
        assert_eq!(&one_poly.as_raw_storage()[1..], &[0; 2047]);

        let mut rng = thread_rng();
        let rand_scalar = rng.gen_range(0..Lwe2048::Q_MODULUS);
        let rand_scalar_ntt = ResidueNtt::<Lwe2048>::from_scalar(Field::new(rand_scalar));
        assert_eq!(
            rand_scalar_ntt.as_raw_storage()[0],
            &[(rand_scalar % Q0 as u64) as u32; 2048]
        );
        assert_eq!(
            rand_scalar_ntt.as_raw_storage()[1],
            &[(rand_scalar % Q1 as u64) as u32; 2048]
        );

        let rand_scalar_poly = Poly::from(rand_scalar_ntt);
        assert_eq!(
            rand_scalar_poly.as_raw_storage()[0],
            (Field::<Lwe2048>::new(rand_scalar) * Field::new(2048)).to_raw()
        ); // TODO: normalization hygiene
        assert_eq!(&rand_scalar_poly.as_raw_storage()[1..], &[0; 2047]);

        let mut rng = thread_rng();
        let rand: Poly<Lwe2048> = rng.gen();
        let rand_ntt = ResidueNtt::from(rand.clone());
        let rand_poly = Poly::from(rand_ntt);
        assert_eq!(
            rand_poly.as_raw_storage()[0],
            (rand.clone() * Field::new(2048)).as_raw_storage()[0]
        ); // TODO: normalization hygiene
        assert_eq!(
            rand_poly.as_raw_storage(),
            (rand * Field::new(2048)).as_raw_storage()
        ); // TODO: normalization hygiene
    }

    fn test_decomposition<const D: usize>(input: Poly<Lwe2048>) {
        let dec = <Poly<Lwe2048> as Decompose<D>>::decompose(&input);
        assert_eq!(
            Poly::<Lwe2048>::inner_prod(
                &dec,
                &<Poly<Lwe2048> as Decompose<D>>::basis().map(Poly::from_scalar)
            ),
            input,
        );
    }

    #[test]
    fn decompose_simple() {
        assert_eq!(
            Poly::<Lwe2048>::default().decompose().as_ref(),
            &[Poly::<Lwe2048>::default(), Poly::<Lwe2048>::default()]
        );
        assert_eq!(
            Poly::<Lwe2048>::from_scalar(Field::ONE)
                .decompose()
                .as_ref(),
            &[
                Poly::<Lwe2048>::from_scalar(Field::ONE),
                Poly::<Lwe2048>::default(),
            ]
        );
        test_decomposition::<2>(Poly::<Lwe2048>::from_scalar(Field::ONE));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]
        #[test]
        fn decompose(
            poly in any::<Poly<Lwe2048>>(),
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
