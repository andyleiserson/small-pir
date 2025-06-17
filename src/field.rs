use std::{
    array,
    ops::{Add, Mul, Neg, Sub},
};

use funty::Integral;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{lwe::LwePrivate, Decompose, InnerProduct};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Field<L: LwePrivate>(L::Storage);

impl<L: LwePrivate> Field<L> {
    pub const ZERO: Self = Self(L::Storage::ZERO);
    pub const ONE: Self = Self(L::Storage::ONE);
}

impl<L: LwePrivate> Field<L> {
    pub fn new(value: L::Storage) -> Self {
        Self(value)
    }

    pub fn to_raw(&self) -> L::Storage {
        self.0
    }
}

impl<L: LwePrivate> Add<Self> for Field<L> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(L::reduce(
            L::OpStorage::from(self.0) + L::OpStorage::from(rhs.0),
        ))
    }
}

impl<L: LwePrivate> Sub<Self> for Field<L> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(L::reduce(
            L::OpStorage::from(self.0) + L::OpStorage::from(L::Q_MODULUS)
                - L::OpStorage::from(rhs.0),
        ))
    }
}

impl<L: LwePrivate> Neg for Field<L> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.0 != L::Storage::from(0) {
            Self(L::Q_MODULUS - self.0)
        } else {
            Self(L::Storage::from(0))
        }
    }
}

impl<L: LwePrivate> Mul<Self> for Field<L> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(L::reduce(
            L::OpStorage::from(self.0) * L::OpStorage::from(rhs.0),
        ))
    }
}

impl<L: LwePrivate> Distribution<Field<L>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Field<L> {
        Field(rng.sample(rand::distributions::Uniform::from(
            L::Storage::ZERO..L::Q_MODULUS,
        )))
    }
}

impl<L: LwePrivate, const D: usize> Decompose<D> for Field<L> {
    type Basis = Box<[Self; D]>;

    fn decompose(&self) -> Box<[Self; D]> {
        let log_beta = L::Q_BITS.div_ceil(D);
        let beta_mask = (L::Storage::ONE << log_beta) - L::Storage::ONE;
        let mut result: Box<[Self; D]> = vec![Default::default(); D].try_into().unwrap();
        let mut val = self.0;
        for i in 0..D {
            result[i] = L::field(val & beta_mask);
            val >>= log_beta;
        }
        result
    }

    fn basis() -> Box<[Self; D]> {
        let log_beta =
            (L::Storage::BITS - L::Q_MODULUS.leading_zeros()).div_ceil(u32::try_from(D).unwrap());
        Box::new(array::from_fn(|i| {
            let i = u32::try_from(i).unwrap();
            Self(L::Storage::from(1) << (i * log_beta))
        }))
    }
}

impl<L: LwePrivate> InnerProduct for Field<L> {
    fn inner_prod<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        let mut result = Default::default();
        for i in 0..N {
            result = result + lhs[i] * rhs[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use proptest::{prelude::*, proptest};

    use super::*;
    use crate::lwe::Lwe1024;

    impl<L: LwePrivate> Arbitrary for Field<L>
    where
        Range<L::Storage>: Strategy<Value = L::Storage>,
    {
        type Parameters = ();
        type Strategy = prop::strategy::Map<Range<L::Storage>, fn(L::Storage) -> Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (L::Storage::ZERO..L::Q_MODULUS).prop_map(L::field)
        }
    }

    #[test]
    fn decompose_simple() {
        assert_eq!(
            Field::<Lwe1024>::ZERO.decompose().as_ref(),
            &[0; 2].map(Field)
        );
        assert_eq!(
            Field::<Lwe1024>::ZERO.decompose().as_ref(),
            &[0; 3].map(Field)
        );
        assert_eq!(
            Field::<Lwe1024>::ZERO.decompose().as_ref(),
            &[0; 4].map(Field)
        );

        if Lwe1024::Q_MODULUS == 0xffffd801 {
            assert_eq!(
                (-Field::<Lwe1024>::ONE).decompose().as_ref(),
                &[0xd800, 0xffff].map(Field)
            );
            assert_eq!(
                (-Field::<Lwe1024>::ONE).decompose().as_ref(),
                &[0, 0x7fb, 0x3ff].map(Field)
            );
            assert_eq!(
                (-Field::<Lwe1024>::ONE).decompose().as_ref(),
                &[0, 0xd8, 0xff, 0xff].map(Field)
            );
        }
    }

    fn test_decomposition<const D: usize>(input: Field<Lwe1024>) {
        assert_eq!(
            Field::<Lwe1024>::inner_prod(
                &<Field<Lwe1024> as Decompose<D>>::decompose(&input),
                &<Field<Lwe1024> as Decompose<D>>::basis()
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
        fn decompose_proptest(
            inputs in prop::collection::vec(any::<Field<Lwe1024>>(), 32),
            degree in prop::sample::select(vec![2, 3, 4, 6, 8, 11, 16]),
        ) {
            for input in inputs {
                match degree {
                    2 => test_decomposition::<2>(input),
                    3 => test_decomposition::<3>(input),
                    4 => test_decomposition::<4>(input),
                    6 => test_decomposition::<6>(input),
                    8 => test_decomposition::<8>(input),
                    11 => test_decomposition::<11>(input),
                    16 => test_decomposition::<16>(input),
                    _ => unreachable!()
                }
            }
        }
    }
}
