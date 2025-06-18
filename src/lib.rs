#![feature(array_chunks)]

pub mod field;
pub mod lwe;
pub mod math;
pub mod morph;
pub mod pir;
pub mod poly;
mod timer;

pub trait Decompose<const D: usize>: Sized {
    type Basis;

    fn decompose(&self) -> Box<[Self; D]>;

    fn basis() -> Self::Basis;
}

pub trait InnerProduct: Sized {
    fn inner_prod<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self;
}

pub struct OneMinus<T>(pub T);

pub fn map_boxed_array<T, const N: usize, F: FnMut(T) -> U, U>(
    boxed_array: Box<[T; N]>,
    f: F,
) -> Box<[U; N]> {
    // Calling into_iter on the boxed array directly seems to move it back to the stack.
    Vec::from(boxed_array as Box<[_]>)
        .into_iter()
        .map(f)
        .collect::<Vec<_>>()
        .try_into()
        .ok()
        .unwrap()
}
