use std::iter::zip;

use num_integer::gcd;
use rand_distr::{Distribution, Standard};

use crate::{
    lwe::{BfvCiphertext, Compressor, Decompress, LwePrivate},
    map_boxed_array,
    poly::Poly,
    Decompose, InnerProduct,
};

#[derive(Clone)]
pub struct Morph<L: LwePrivate> {
    table: L::Morph,
}

impl<L: LwePrivate> Morph<L> {
    pub fn new(power: usize) -> Option<Self> {
        if gcd(power, 2 * L::DEGREE) != 1 {
            return None;
        }

        let table = L::morph_from_fn(|i| {
            let dest = i * power % (2 * L::DEGREE);
            if dest >= L::DEGREE {
                -i16::try_from(dest - L::DEGREE).unwrap()
            } else {
                i16::try_from(dest).unwrap()
            }
        });

        Some(Self { table })
    }

    pub fn apply(&self, value: &Poly<L>) -> Poly<L> {
        let mut result = L::array_zero();
        for i in 0..L::DEGREE {
            let dest = self.table.as_ref()[i];
            if dest < 0 {
                result.as_mut()[usize::try_from(-dest).unwrap()] =
                    (-L::field(value.as_raw_storage()[i])).to_raw();
            } else {
                result.as_mut()[usize::try_from(dest).unwrap()] = value.as_raw_storage()[i];
            }
        }
        result.into()
    }
}

#[allow(clippy::type_complexity)]
pub struct EncryptedMorph<L: LwePrivate, const D: usize> {
    pub key: (Box<[L::Ntt; D]>, Box<[L::Ntt; D]>),
    pub morph: Morph<L>,
}

pub struct EncryptedMorphCompressed<L: LwePrivate, const D: usize> {
    pub key: ([u64; D], Box<[L::Ntt; D]>),
    pub morph: Morph<L>,
}

impl<L: LwePrivate, const D: usize> Decompress for EncryptedMorphCompressed<L, D>
where
    Standard: Distribution<L::Ntt>,
{
    type Decompressed = EncryptedMorph<L, D>;

    fn decompress(&self, compressor: &Compressor) -> Self::Decompressed {
        let key0 = Box::new(self.key.0.map(|i| compressor.get(i)));
        EncryptedMorph {
            key: (key0, self.key.1.clone()),
            morph: self.morph.clone(),
        }
    }
}

impl<L: LwePrivate, const D: usize> EncryptedMorph<L, D> {
    pub fn apply(&self, value: &BfvCiphertext<L>) -> BfvCiphertext<L> {
        let (in0, in1) = value.as_raw();

        let in0m = self.morph.apply(in0);
        let in1m = self.morph.apply(in1);

        let in0md: Box<[_; D]> =
            map_boxed_array(<_ as Decompose<D>>::decompose(&in0m), L::Ntt::from);

        let out0: Poly<_> = L::Ntt::inner_prod(&self.key.0, &in0md).into();
        let out1: Poly<_> = L::Ntt::inner_prod(&self.key.1, &in0md).into() + in1m;

        BfvCiphertext::from_raw(out0, out1)
    }

    pub fn size(&self) -> usize {
        2 * D * L::Q_BITS * L::DEGREE / 8
    }
}

impl<L: LwePrivate, const D: usize> EncryptedMorphCompressed<L, D> {
    pub fn size(&self) -> usize {
        D * L::Q_BITS * L::DEGREE / 8
    }
}

pub fn unpack_one<L: LwePrivate, const D: usize>(
    morph: &EncryptedMorph<L, D>,
    pack_dim: usize,
    ciphertext: &BfvCiphertext<L>,
    index: usize,
) -> BfvCiphertext<L> {
    unpack_one_ext(morph, pack_dim, ciphertext, index, 1, 1)
}

/// Extended unpack primitive that can be used to implement nested unpackings.
///
/// `scale1` is the number of times to iterate `enc_morph`.
/// `scale2` is the scaling for pack_dim and index. It is used for subsequent unpack
/// passes, which are effectively operating in a subgroup.
pub fn unpack_one_ext<L: LwePrivate, const D: usize>(
    morph: &EncryptedMorph<L, D>,
    pack_dim: usize,
    ciphertext: &BfvCiphertext<L>,
    index: usize,
    scale1: usize,
    scale2: usize,
) -> BfvCiphertext<L> {
    let scale2 = scale2 as isize;
    let mut y = ciphertext.clone();
    y.rotate(-(scale2 * (pack_dim + index - 1) as isize));

    for _ in 0..pack_dim - 1 {
        let mut z = y.clone();
        for _ in 0..scale1 {
            z = morph.apply(&z);
        }
        y = y - z;
        y.rotate(scale2);
    }

    y
}

pub fn unpack_all<L: LwePrivate, const D: usize>(
    morph: &EncryptedMorph<L, D>,
    pack_dim: usize,
    ciphertext: &BfvCiphertext<L>,
) -> Vec<BfvCiphertext<L>> {
    let mut unpacked = Vec::with_capacity(pack_dim);
    unpacked.push(ciphertext.clone());

    let mut steps = pack_dim / 2;
    for j in 0..pack_dim.ilog2() {
        /*
        for i in 0..unpacked.len() {
            let norm = decryptor(unpacked[i].clone()).inf_norm();
            println!("unp[{j}][{i}]: {}", (<_ as TryInto<u64>>::try_into(norm).ok().unwrap() as f32).log2());
        }
        */
        let mut unpacked_next = Vec::with_capacity(2 * unpacked.len());
        let z = unpacked
            .iter()
            .map(|y| {
                let mut z = y.clone();
                for _ in 0..steps {
                    z = morph.apply(&z);
                    /*
                    if super::PRINT_NOISE {
                        let norm = decryptor(z.clone()).inf_norm();
                        println!("step {k} z: {}", (<_ as TryInto<u64>>::try_into(norm).ok().unwrap() as f32).log2());
                    }
                    */
                }
                z
            })
            .collect::<Vec<_>>();
        unpacked_next.extend(zip(&unpacked, &z).map(|(y, z)| y.clone() + z.clone()));
        unpacked_next.extend(zip(unpacked, z).map(|(y, z)| {
            let mut x = y - z;
            x.rotate(-1isize << j);
            x
        }));
        steps >>= 1;
        unpacked = unpacked_next;
    }

    /*
    for i in 0..unpacked.len() {
        let norm = decryptor(unpacked[i].clone()).inf_norm();
        println!("unp[{depth}][{i}]: {}", (<_ as TryInto<u64>>::try_into(norm).ok().unwrap() as f32).log2());
    }
    */

    unpacked
}

#[cfg(test)]
mod tests {
    use std::{array, iter::zip};

    use proptest::{prelude::*, proptest};
    use rand::thread_rng;

    use super::*;
    use crate::lwe::{Lwe1024, Lwe2048, LweParams};

    #[test]
    fn morph() {
        let morph = Morph::<Lwe1024>::new(3).unwrap();
        let morph_inv = Morph::<Lwe1024>::new(683).unwrap();

        assert_eq!(
            morph.apply(&Poly::from_scalar(Lwe1024::field(1))),
            Poly::from_scalar(Lwe1024::field(1))
        );

        let x = Lwe1024::array_from_fn(|i| if i == 1 { 1 } else { 0 }).into();
        let y = Lwe1024::array_from_fn(|i| if i == 3 { 1 } else { 0 }).into();
        assert_eq!(morph.apply(&x), y);

        assert_eq!(morph_inv.apply(&morph.apply(&x)), x);
    }

    #[test]
    fn morph_invalid() {
        assert!(Morph::<Lwe1024>::new(2).is_none());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]
        #[test]
        fn morph_proptest(
            poly in any::<Poly<Lwe1024>>(),
        ) {
            let morph = Morph::<Lwe1024>::new(3).unwrap();
            let morph_inv = Morph::<Lwe1024>::new(683).unwrap();

            assert_eq!(morph_inv.apply(&morph.apply(&poly)), poly);
        }
    }

    #[test]
    fn encrypted_morph() {
        let mut rng = thread_rng();
        let plaintext = rng.gen_range(0..4);
        let mut lwe = Lwe1024::gen_uniform(rng.clone());
        let ciphertext = lwe.encrypt_bfv(plaintext);

        let morph = Morph::<Lwe1024>::new(3).unwrap();
        let enc_morph = lwe.morph::<8>(3).unwrap();

        let ct_morphed = enc_morph.apply(&ciphertext);

        let pt_morphed = lwe.decrypt_bfv(ct_morphed);

        assert_eq!(
            morph.apply(&Poly::from_scalar(Lwe1024::field(plaintext))),
            pt_morphed
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]
        #[test]
        fn encrypted_morph_proptest(
            plaintext in 0u32..4,
        ) {
            let rng = thread_rng();
            let mut lwe = Lwe1024::gen_uniform(rng.clone());
            let ciphertext = lwe.encrypt_bfv(plaintext);

            let morph = Morph::<Lwe1024>::new(3).unwrap();
            let enc_morph = lwe.morph::<8>(3).unwrap();

            let ct_morphed = enc_morph.apply(&ciphertext);

            let pt_morphed = lwe.decrypt_bfv(ct_morphed);

            assert_eq!(morph.apply(&Poly::from_scalar(Lwe1024::field(plaintext))), pt_morphed);
        }
    }

    #[test]
    fn unpack2() {
        let mut rng = thread_rng();
        let mut lwe = Lwe1024::gen_uniform(rng.clone());
        let i0 = rng.gen_range(1..4);
        let i1 = rng.gen_range(1..4);
        let mut plaintext = Lwe1024::array_zero();
        plaintext[0] = i0 * Lwe1024::floor_q_div_p();
        plaintext[1] = i1 * Lwe1024::floor_q_div_p();
        let ciphertext = lwe.encrypt_bfv_poly(Poly::from(plaintext));

        let enc_morph = lwe.morph::<8>(1025).unwrap();

        let mut x0 = ciphertext.clone();
        x0.rotate(-1);
        let mut x1 = ciphertext;
        x1.rotate(-2);

        let extract2 = |mut ct: BfvCiphertext<Lwe1024>| {
            ct = ct.clone() - enc_morph.apply(&ct);
            ct.rotate(1);
            ct
        };

        assert_eq!(lwe.decrypt_bfv(extract2(x0)).as_raw_storage()[0], 2 * i0);
        assert_eq!(lwe.decrypt_bfv(extract2(x1)).as_raw_storage()[0], 2 * i1);
    }

    #[test]
    fn unpack4() {
        let mut rng = thread_rng();
        let mut lwe = Lwe1024::gen_uniform(rng.clone());
        let i0 = rng.gen_range(1..4);
        let i1 = rng.gen_range(1..4);
        let i2 = rng.gen_range(1..4);
        let i3 = rng.gen_range(1..4);
        let mut plaintext = Lwe1024::array_zero();
        plaintext[0] = i0 * Lwe1024::floor_q_div_p();
        plaintext[1] = i1 * Lwe1024::floor_q_div_p();
        plaintext[2] = i2 * Lwe1024::floor_q_div_p();
        plaintext[3] = i3 * Lwe1024::floor_q_div_p();
        let ciphertext = lwe.encrypt_bfv_poly(Poly::from(plaintext));

        let enc_morph = lwe.morph::<8>(513).unwrap();

        let mut x0 = ciphertext.clone();
        x0.rotate(-3);
        let mut x1 = ciphertext.clone();
        x1.rotate(-4);
        let mut x2 = ciphertext.clone();
        x2.rotate(-5);
        let mut x3 = ciphertext;
        x3.rotate(-6);

        let extract4 = |mut ct: BfvCiphertext<Lwe1024>| {
            ct = ct.clone() - enc_morph.apply(&ct);
            ct.rotate(1);
            ct = ct.clone() - enc_morph.apply(&ct);
            ct.rotate(1);
            ct = ct.clone() - enc_morph.apply(&ct);
            ct.rotate(1);
            ct
        };

        let y0 = extract4(x0);
        let y1 = extract4(x1);
        let y2 = extract4(x2);
        let y3 = extract4(x3);

        assert_eq!(lwe.decrypt_bfv(y0).as_raw_storage()[0], 4 * i0);
        assert_eq!(lwe.decrypt_bfv(y1).as_raw_storage()[0], 4 * i1);
        assert_eq!(lwe.decrypt_bfv(y2).as_raw_storage()[0], 4 * i2);
        assert_eq!(lwe.decrypt_bfv(y3).as_raw_storage()[0], 4 * i3);
    }

    fn unpack_test<L: LweParams, const ELL_KS: usize, const PACK: usize>() {
        let mut rng = thread_rng();
        let mut lwe = L::gen_uniform(rng.clone());
        let x_vec: [L::Storage; PACK] = array::from_fn(|_| rng.gen_range(1..4).into());
        let mut plaintext = L::array_zero();
        for (i, &x) in x_vec.iter().enumerate() {
            plaintext.as_mut()[i] = x * L::floor_q_div_p() / L::Storage::from(PACK as u32);
        }
        let ciphertext = lwe.encrypt_bfv_poly(plaintext.clone().into());
        println!(
            "packed: log₂|ε| = {:.1}",
            (<_ as TryInto<u64>>::try_into(
                (lwe.raw_decrypt_bfv(ciphertext.clone()) - plaintext.clone().into()).inf_norm()
            )
            .ok()
            .unwrap() as f64)
                .log2(),
        );

        let enc_morph = lwe.unpack_morph::<ELL_KS>(PACK).unwrap();

        for (i, &x) in x_vec.iter().enumerate() {
            let unp = unpack_one::<_, ELL_KS>(&enc_morph, PACK, &ciphertext, i);
            let dec = lwe.decrypt_bfv(unp.clone());
            assert_eq!(dec.as_raw_storage()[0], x);
            println!(
                "unpack[{i}]: log₂|ε| = {:.1}",
                (<_ as TryInto<u64>>::try_into(
                    (lwe.raw_decrypt_bfv(unp)
                        - Poly::from_scalar(L::field(x * L::floor_q_div_p())))
                    .inf_norm()
                )
                .ok()
                .unwrap() as f64)
                    .log2(),
            );
        }
    }

    fn unpack_all_test<L: LweParams, const ELL_KS: usize, const PACK: usize>() {
        let mut rng = thread_rng();
        let mut lwe = L::gen_uniform(rng.clone());
        let x_vec: [L::Storage; PACK] = array::from_fn(|_| rng.gen_range(1..4).into());
        let mut plaintext = L::array_zero();
        for (i, &x) in x_vec.iter().enumerate() {
            plaintext.as_mut()[i] = x * L::floor_q_div_p() / L::Storage::from(PACK as u32);
        }
        let ciphertext = lwe.encrypt_bfv_poly(plaintext.clone().into());
        println!(
            "packed: log₂|ε| = {:.1}",
            (<_ as TryInto<u64>>::try_into(
                (lwe.raw_decrypt_bfv(ciphertext.clone()) - plaintext.clone().into()).inf_norm()
            )
            .ok()
            .unwrap() as f64)
                .log2(),
        );

        let enc_morph = lwe.unpack_morph::<ELL_KS>(PACK).unwrap();
        let unpacked = unpack_all::<_, ELL_KS>(&enc_morph, PACK, &ciphertext);

        for (i, (&x, unp)) in zip(&x_vec, unpacked).enumerate() {
            let dec = lwe.decrypt_bfv(unp.clone());
            assert_eq!(dec.as_raw_storage()[0], x);
            println!(
                "unpack[{i}]: log₂|ε| = {:.1}",
                (<_ as TryInto<u64>>::try_into(
                    (lwe.raw_decrypt_bfv(unp)
                        - Poly::from_scalar(L::field(x * L::floor_q_div_p())))
                    .inf_norm()
                )
                .ok()
                .unwrap() as f64)
                    .log2(),
            );
        }
    }

    #[test]
    fn unpack4alt() {
        unpack_test::<Lwe1024, 8, 4>();
    }

    #[test]
    fn unpack8() {
        unpack_test::<Lwe1024, 8, 8>();
    }

    #[test]
    fn unpack8alt() {
        unpack_all_test::<Lwe1024, 8, 8>();
    }

    #[test]
    fn unpack16() {
        unpack_test::<Lwe1024, 8, 16>();
    }

    #[test]
    fn unpack32() {
        unpack_test::<Lwe1024, 8, 32>();
    }

    #[test]
    fn unpack8_2048() {
        unpack_all_test::<Lwe2048, 8, 8>();
    }

    #[test]
    fn unpack64_2048() {
        unpack_all_test::<Lwe2048, 8, 64>();
    }

    fn double_unpack4<L: LweParams, const ELL_KS: usize>() {
        // Unpack 16 items with nested 4x unpackings
        const PACK: usize = 4;
        let mut rng = thread_rng();
        let mut lwe = L::gen_uniform(rng.clone());
        let x_vec: [L::Storage; PACK * PACK] = array::from_fn(|_| rng.gen_range(1..4).into());
        let mut plaintext = L::array_zero();
        for (i, &x) in x_vec.iter().enumerate() {
            plaintext.as_mut()[i] = x * L::floor_q_div_p() / L::Storage::from((PACK * PACK) as u32);
        }
        let ciphertext = lwe.encrypt_bfv_poly(plaintext.clone().into());
        println!(
            "packed: log₂|ε| = {:.1}",
            (<_ as TryInto<u64>>::try_into(
                (lwe.raw_decrypt_bfv(ciphertext.clone()) - plaintext.clone().into()).inf_norm()
            )
            .ok()
            .unwrap() as f64)
                .log2(),
        );

        // Applying this x16 automorphism four times is equivalent to x4 automorphism.
        let enc_morph = lwe.unpack_morph::<ELL_KS>(PACK * PACK).unwrap();

        // Apply x4 unpacking to isolate the indices at a particular position mod 4.
        // This is just a regular unpack, except that we need to iterate enc_morph.
        // We do this for each position mod 4 since we are going to unpack and verify
        // all the indices.
        let unp1: [_; PACK] = array::from_fn(|i| {
            let unp = super::unpack_one_ext::<_, ELL_KS>(&enc_morph, PACK, &ciphertext, i, PACK, 1);
            let pt = L::array_from_fn(|j| {
                if j % PACK == 0 {
                    plaintext.as_ref()[j + i] * L::Storage::from(PACK as u32)
                } else {
                    L::Storage::default()
                }
            })
            .into();
            println!(
                "first-level unpack[{i}]: log₂|ε| = {:.1}",
                (<_ as TryInto<u64>>::try_into((lwe.raw_decrypt_bfv(unp.clone()) - pt).inf_norm())
                    .ok()
                    .unwrap() as f64)
                    .log2(),
            );
            unp
        });

        // Apply x4 nested unpacking. Once we have indices only in positions with i % 4 == 0,
        // we can use the x16 automorphism to do this. It is effectively acting as the x4
        // automorphism on the positions with i % 4 == 0.
        for (i, &x) in x_vec.iter().enumerate() {
            let unp = super::unpack_one_ext::<_, ELL_KS>(
                &enc_morph,
                PACK,
                &unp1[i % PACK],
                i / PACK,
                1,
                PACK,
            );
            let dec = lwe.decrypt_bfv(unp.clone());
            println!(
                "unpack[{i}]: log₂|ε| = {:.1}",
                (<_ as TryInto<u64>>::try_into(
                    (lwe.raw_decrypt_bfv(unp)
                        - Poly::from_scalar(L::field(x * L::floor_q_div_p())))
                    .inf_norm()
                )
                .ok()
                .unwrap() as f64)
                    .log2(),
            );
            assert_eq!(
                dec.as_raw_storage()[0],
                x,
                "decrypted {} != expected {}",
                dec.as_raw_storage()[0],
                x
            );
        }
    }

    #[test]
    fn test_double_unpack4() {
        double_unpack4::<Lwe1024, 8>();
    }

    fn triple_unpack4<L: LweParams, const ELL_KS: usize>() {
        let mut rng = thread_rng();
        let mut lwe = L::gen_uniform(rng.clone());
        let x_vec: [L::Storage; 64] = array::from_fn(|_| rng.gen_range(1..4).into());
        let mut plaintext = L::array_zero();
        for (i, &x) in x_vec.iter().enumerate() {
            plaintext.as_mut()[i] = x * L::floor_q_div_p() / L::Storage::from(64);
        }
        let ciphertext = lwe.encrypt_bfv_poly(plaintext.clone().into());
        println!(
            "packed: log₂|ε| = {:.1}",
            (<_ as TryInto<u64>>::try_into(
                (lwe.raw_decrypt_bfv(ciphertext.clone()) - plaintext.clone().into()).inf_norm()
            )
            .ok()
            .unwrap() as f64)
                .log2(),
        );

        let enc_morph33 = lwe.unpack_morph::<ELL_KS>(64).unwrap();
        let enc_morph129 = lwe.unpack_morph::<ELL_KS>(16).unwrap();

        // Apply x4 unpacking to isolate the indices at a particular position mod 4.
        // This is just a regular unpack, except that we need to iterate enc_morph.
        // We do this for each position mod 4 since we are going to unpack and verify
        // all the indices.
        let unp1: Box<[_; 4]> = (0..4)
            .map(|i| {
                let unp =
                    super::unpack_one_ext::<_, ELL_KS>(&enc_morph129, 4, &ciphertext, i, 4, 1);
                let pt = L::array_from_fn(|j| {
                    if j % 4 == 0 {
                        plaintext.as_ref()[j + i] * L::Storage::from(4)
                    } else {
                        L::Storage::default()
                    }
                })
                .into();
                println!(
                    "first-level unpack[{i}]: log₂|ε| = {:.1}",
                    (<_ as TryInto<u64>>::try_into(
                        (lwe.raw_decrypt_bfv(unp.clone()) - pt).inf_norm()
                    )
                    .ok()
                    .unwrap() as f64)
                        .log2(),
                );
                unp
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Apply second level nested unpacking.
        let unp2: Box<[_; 16]> = (0..16)
            .map(|i| {
                let unp =
                    super::unpack_one_ext::<_, ELL_KS>(&enc_morph129, 4, &unp1[i % 4], i / 4, 1, 4);
                let pt = L::array_from_fn(|j| {
                    if j % 16 == 0 {
                        plaintext.as_ref()[j + i] * L::Storage::from(4) * L::Storage::from(4)
                    } else {
                        L::Storage::default()
                    }
                })
                .into();
                println!(
                    "second-level unpack[{i}]: log₂|ε| = {:.1}",
                    (<_ as TryInto<u64>>::try_into(
                        (lwe.raw_decrypt_bfv(unp.clone()) - pt).inf_norm()
                    )
                    .ok()
                    .unwrap() as f64)
                        .log2(),
                );
                unp
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Apply third and final nested unpacking.
        for (i, &x) in x_vec.iter().enumerate() {
            let unp =
                super::unpack_one_ext::<_, ELL_KS>(&enc_morph33, 4, &unp2[i % 16], i / 16, 1, 16);
            let dec = lwe.decrypt_bfv(unp.clone());
            println!(
                "unpack[{i}]: log₂|ε| = {:.1}",
                (<_ as TryInto<u64>>::try_into(
                    (lwe.raw_decrypt_bfv(unp)
                        - Poly::from_scalar(L::field(x * L::floor_q_div_p())))
                    .inf_norm()
                )
                .ok()
                .unwrap() as f64)
                    .log2(),
            );
            assert_eq!(
                dec.as_raw_storage()[0],
                x,
                "decrypted {} != expected {}",
                dec.as_raw_storage()[0],
                x
            );
        }
    }

    #[test]
    fn test_triple_unpack4() {
        triple_unpack4::<Lwe2048, 8>();
    }

    /*
    fn quadruple_unpack4<L: LweParams, const ELL_KS: usize>() {
        let mut rng = thread_rng();
        let mut lwe = L::gen_uniform(rng.clone());
        let x_vec: [L::Storage; 256] = array::from_fn(|_| rng.gen_range(1..4).into());
        let mut plaintext = L::array_zero();
        for (i, &x) in x_vec.iter().enumerate() {
            plaintext.as_mut()[i] = x * L::floor_q_div_p() / L::Storage::from(256 as u32);
        }
        let ciphertext = lwe.encrypt_bfv_poly(&mut rng, plaintext.clone().into());
        println!(
            "packed: log₂|ε| = {:.1}",
            (
                <_ as TryInto<u64>>::try_into(
                    (lwe.raw_decrypt_bfv(ciphertext.clone()) - plaintext.clone().into())
                        .inf_norm()
                )
                .ok()
                .unwrap()
            as f64).log2(),
        );

        let enc_morph33 = lwe.unpack_morph::<ELL_KS>(64).unwrap();
        let enc_morph129 = lwe.unpack_morph::<ELL_KS>(16).unwrap();

        // Apply x4 unpacking to isolate the indices at a particular position mod 4.
        // This is just a regular unpack, except that we need to iterate enc_morph.
        // We do this for each position mod 4 since we are going to unpack and verify
        // all the indices.
        let unp1: Box<[_; 4]> = (0..4).map(|i| {
            let unp = super::unpack_ext::<_, ELL_KS>(&enc_morph129, 4, &ciphertext, i, 4, 1);
            let pt = L::array_from_fn(|j| {
                if j % 4 == 0 {
                    plaintext.as_ref()[j + i] * L::Storage::from(4)
                } else {
                    L::Storage::default()
                }
            }).into();
            println!(
                "first-level unpack[{i}]: log₂|ε| = {:.1}",
                (
                    <_ as TryInto<u64>>::try_into(
                        (lwe.raw_decrypt_bfv(unp.clone()) - pt).inf_norm()
                    )
                    .ok()
                    .unwrap()
                as f64).log2(),
            );
            unp
        }).collect::<Vec<_>>().try_into().unwrap();

        // Apply second level nested unpacking.
        let unp2: Box<[_; 16]> = (0..16).map(|i| {
            let unp = super::unpack_ext::<_, ELL_KS>(&enc_morph129, 4, &unp1[i % 4], i / 4, 1, 4);
            let pt = L::array_from_fn(|j| {
                if j % 16 == 0 {
                    plaintext.as_ref()[j + i] * L::Storage::from(4) * L::Storage::from(4)
                } else {
                    L::Storage::default()
                }
            }).into();
            println!(
                "second-level unpack[{i}]: log₂|ε| = {:.1}",
                (
                    <_ as TryInto<u64>>::try_into(
                        (lwe.raw_decrypt_bfv(unp.clone()) - pt).inf_norm()
                    )
                    .ok()
                    .unwrap()
                as f64).log2(),
            );
            unp
        }).collect::<Vec<_>>().try_into().unwrap();

        // Apply third and final nested unpacking.
        for (i, &x) in x_vec.iter().enumerate() {
            let unp = super::unpack_ext::<_, ELL_KS>(&enc_morph33, 4, &unp2[i % 16], i / 16, 1, 16);
            let dec = lwe.decrypt_bfv(unp.clone());
            println!(
                "unpack[{i}]: log₂|ε| = {:.1}",
                (
                    <_ as TryInto<u64>>::try_into(
                        (lwe.raw_decrypt_bfv(unp) - Poly::from_scalar(L::field(x * L::floor_q_div_p())))
                            .inf_norm()
                    )
                    .ok()
                    .unwrap()
                as f64).log2(),
            );
            assert_eq!(dec.as_raw_storage()[0], x, "decrypted {} != expected {}", dec.as_raw_storage()[0], x);
        }
    }

    #[test]
    fn test_triple_unpack4() {
        triple_unpack4::<Lwe2048, 8>();
    }
    */
}
