#![cfg_attr(not(feature = "std"), no_std)]
#![deny(clippy::all)]
#![warn(
    // TODO activate this // missing_docs,
    clippy::missing_const_for_fn,
    clippy::pedantic,
    missing_copy_implementations,
    missing_debug_implementations,
    unused_qualifications
)]
#![allow(clippy::use_self)]
#![doc(test(attr(deny(warnings))))]

pub mod iter;

use core::{
    array,
    clone::Clone,
    iter::{Iterator, Sum},
    num::NonZeroUsize,
    ops::{Add, Sub},
};
use iter::{
    BoundedMooreNeighborhood, BoundedNeumannNeighborhood, CartesianProduct, MooreNeighborhood,
    NeumannNeighborhood,
};
use num_iter::range_inclusive;
use num_traits::{Bounded, CheckedAdd, CheckedSub, One, ToPrimitive, Zero};

/// An extension trait for working with fixed-length arrays as grid coordinates.
///
/// Use it with
/// ```
/// # #[allow(unused_imports)]
/// use array_cord::ArrayCord;
/// ```
pub trait ArrayCord<T, const DIM: usize>
where
    Self: Sized,
{
    /// Make an array from an iterator of values. [`None`] if not enough values.
    fn from_iter<I: Iterator>(mut it: I) -> Option<[I::Item; DIM]> {
        let a = array::from_fn(|_| it.next());
        if a.iter().any(Option::is_none) {
            None
        } else {
            Some(a.map(|x| x.expect("Checked not None")))
        }
    }

    /// Find the cordinate that coresponds to a given offset where maximum width of each axis is given.
    /// Lower axis idx increment before higher axis idx.
    /// ```
    /// # use array_cord::ArrayCord;
    /// # use core::num::NonZeroUsize;
    /// // x x
    /// // x x
    /// // x x
    /// let widths = [2, 3].map(|x| NonZeroUsize::new(x).unwrap());
    /// assert_eq!(<[usize; 2]>::from_offset(0, widths), [0, 0]);
    /// assert_eq!(<[usize; 2]>::from_offset(1, widths), [1, 0]);
    /// assert_eq!(<[usize; 2]>::from_offset(2, widths), [0, 1]);
    /// assert_eq!(<[usize; 2]>::from_offset(3, widths), [1, 1]);
    /// assert_eq!(<[usize; 2]>::from_offset(4, widths), [0, 2]);
    /// ```
    fn from_offset(mut offset: usize, widths: [NonZeroUsize; DIM]) -> [T; DIM]
    where
        T: From<usize>,
    {
        let mut out = [0; DIM];
        for axis in (0..DIM).rev() {
            if let Some(next_lowest_axis_width) = axis.checked_sub(1) {
                out[axis] = offset / widths[next_lowest_axis_width];
                offset -= out[axis] * usize::from(widths[axis - 1]);
            }
        }
        if let Some(x) = out.first_mut() {
            *x = offset;
        }
        out.map(Into::into)
    }

    /// Elementwise application of a function on two arrays
    fn apply<O>(self, other: Self, func: impl FnMut(T, T) -> O) -> [O; DIM];

    /// The number of orthogonal moves to reach `other` from `self`.
    fn manhattan_distance(self, other: Self) -> T
    where
        T: Sum + Sub<Output = T> + PartialOrd;

    /// [`MooreNeighborhood`] centered on self.
    ///
    /// e.g. with radius `1`:
    /// ```txt
    /// xxx
    /// x x
    /// xxx
    /// ```
    ///
    /// e.g. with radius `2`:
    /// ```txt
    /// xxxxx
    /// xxxxx
    /// xx xx
    /// xxxxx
    /// xxxxx
    /// ```
    /// # Panics
    /// If `T` can't represent all the neighbors of `Self` (e.g. overflow/underflow) then this will panic with overflow checks enabled.
    fn moore_neighborhood(self, radius: T) -> MooreNeighborhood<T, DIM>
    where
        T: Sub<Output = T> + Ord + Clone + ToPrimitive + Zero + One;

    /// Same as [`Self::moore_neighborhood()`] but bounded to within `T`'s range instead of panicking.
    /// ```
    /// # use array_cord::ArrayCord;
    /// assert_eq!([0u8].moore_neighborhood_bounded(1).collect::<Vec<_>>(), vec![[1]]);
    /// assert_eq!([0i8].moore_neighborhood_bounded(1).collect::<Vec<_>>(), vec![[-1], [1]]);
    /// ```
    fn moore_neighborhood_bounded(self, radius: T) -> BoundedMooreNeighborhood<T, DIM>
    where
        T: Ord + Clone + ToPrimitive + Zero + One + Bounded + CheckedSub + CheckedAdd;

    /// [`NeumannNeighborhood`] centered on self.
    ///
    /// e.g. with radius `1`:
    /// ```txt
    ///  x
    /// x x
    ///  x
    /// ```
    ///
    /// e.g. with radius `2`:
    /// ```txt
    ///   x
    ///  xxx
    /// xx xx
    ///  xxx
    ///   x
    /// ```
    /// # Panics
    /// If `T` can't represent all the neighbors of `Self` (e.g. overflow/underflow) then this will panic with overflow checks enabled.
    fn neumann_neighborhood(self, radius: T) -> NeumannNeighborhood<T, DIM>
    where
        T: Sub<Output = T> + Ord + Clone + ToPrimitive + Zero + One;

    /// Same as [`Self::neumann_neighborhood()`] but bounded to within `T`'s range instead of panicking.
    /// ```
    /// # use array_cord::ArrayCord;
    /// assert_eq!([0i8].neumann_neighborhood_bounded(1).collect::<Vec<_>>(), vec![[-1], [1]]);
    /// assert_eq!([0u8].neumann_neighborhood_bounded(1).collect::<Vec<_>>(), vec![[1]]);
    /// ```
    fn neumann_neighborhood_bounded(self, radius: T) -> BoundedNeumannNeighborhood<T, DIM>
    where
        T: Ord + Clone + ToPrimitive + Zero + One + Bounded + CheckedSub + CheckedAdd;

    /// Return an iterator over all points (inclusive) between `self` and `other`.
    fn interpolate(&self, other: &Self) -> CartesianProduct<num_iter::RangeInclusive<T>, DIM>
    where
        T: Add<Output = T> + Ord + Clone + One + ToPrimitive;

    /// Finds the smallest value in each dimension and largest value in each dimension as the pair `(min, max)`.
    fn extents(&self, other: &Self) -> (Self, Self)
    where
        T: Ord + Clone;

    /// Finds the overall extents `(min_per_axis, max_per_axis)` for many cord using [`Self::extents`]. Handles empty iterator with [`None`].
    fn extents_from_iter(it: impl Iterator<Item = Self>) -> Option<(Self, Self)>
    where
        T: Ord + Clone;
}
impl<T, const DIM: usize> ArrayCord<T, DIM> for [T; DIM] {
    fn apply<O>(self, other: Self, mut func: impl FnMut(T, T) -> O) -> [O; DIM] {
        Self::from_iter(self.into_iter().zip(other).map(|(x, y)| func(x, y))).expect("DIM len")
    }

    fn extents(&self, other: &Self) -> (Self, Self)
    where
        T: Ord + Clone,
    {
        let smallest = array::from_fn(|axis| self[axis].clone().min(other[axis].clone()));
        let largest = array::from_fn(|axis| self[axis].clone().max(other[axis].clone()));
        (smallest, largest)
    }

    fn extents_from_iter(mut it: impl Iterator<Item = Self>) -> Option<(Self, Self)>
    where
        T: Ord + Clone,
    {
        let first = it.next()?;
        Some(it.fold((first.clone(), first), |(min, max), x| {
            (x.extents(&min).0, x.extents(&max).1)
        }))
    }

    fn interpolate(&self, other: &Self) -> CartesianProduct<num_iter::RangeInclusive<T>, DIM>
    where
        T: Add<Output = T> + Ord + Clone + One + ToPrimitive,
    {
        let extents = self.extents(other);

        let ranges = Self::from_iter(
            core::iter::zip(extents.0, extents.1).map(|x| range_inclusive(x.0, x.1)),
        )
        .expect("DIM len");
        CartesianProduct::new(ranges)
    }

    fn manhattan_distance(self, other: Self) -> T
    where
        T: Sum + Sub<Output = T> + PartialOrd,
    {
        fn abs_diff<T: Sub<Output = T> + PartialOrd>(x: T, y: T) -> T {
            if x >= y {
                x - y
            } else {
                y - x
            }
        }

        let diff_per_axis = self.apply(other, abs_diff::<T>);
        diff_per_axis.into_iter().sum()
    }

    fn moore_neighborhood(self, radius: T) -> MooreNeighborhood<T, DIM>
    where
        T: Add<Output = T> + Sub<Output = T> + Ord + Clone + ToPrimitive + Zero + One,
    {
        MooreNeighborhood::new(self, radius)
    }

    fn moore_neighborhood_bounded(self, radius: T) -> BoundedMooreNeighborhood<T, DIM>
    where
        T: Ord + Clone + ToPrimitive + Zero + One + Bounded + CheckedSub + CheckedAdd,
    {
        BoundedMooreNeighborhood::new(self, radius)
    }

    fn neumann_neighborhood(self, radius: T) -> NeumannNeighborhood<T, DIM>
    where
        T: Sub<Output = T> + Ord + Clone + ToPrimitive + Zero + One,
    {
        NeumannNeighborhood::new(self, radius)
    }

    fn neumann_neighborhood_bounded(self, radius: T) -> BoundedNeumannNeighborhood<T, DIM>
    where
        T: Ord + Clone + ToPrimitive + Zero + One + Bounded + CheckedSub + CheckedAdd,
    {
        BoundedNeumannNeighborhood::new(self, radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manhattan_distance() {
        let cord1 = [-2isize, 4];
        let cord2 = [498, 6];
        let out = cord1.manhattan_distance(cord2);
        assert_eq!(out, 500 + 2);
    }

    #[test]
    fn moore_neighborhood() {
        let cord = [0];
        let out = cord.moore_neighborhood(1);
        let out_iter_len = out.len();
        let out_size = out.size_hint();
        let out_vec = out.collect::<Vec<_>>();
        assert_eq!(out_vec, vec![[-1], [1]]);
        assert_eq!(out_vec.len(), out_size.0);
        assert_eq!(out_vec.len(), out_size.1.unwrap());
        assert_eq!(out_vec.len(), out_iter_len);

        let cord = [-8, 4];
        let out = cord.moore_neighborhood(1);
        let out_iter_len = out.len();
        let out_size = out.size_hint();
        let out_vec = out.collect::<Vec<_>>();
        #[rustfmt::skip]
        assert_eq!(
            out_vec,
            vec![
                [-9, 3], [-9, 4], [-9, 5],
                [-8, 3],          [-8, 5],
                [-7, 3], [-7, 4], [-7, 5]
            ]
        );
        assert_eq!(out_vec.len(), out_size.0);
        assert_eq!(out_vec.len(), out_size.1.unwrap());
        assert_eq!(out_vec.len(), out_iter_len);
        let out = cord.moore_neighborhood(2);
        let out_iter_len = out.len();
        let out_size = out.size_hint();
        let out_vec = out.collect::<Vec<_>>();
        assert_eq!(out_vec.len(), out_size.0);
        assert_eq!(out_vec.len(), out_size.1.unwrap());
        assert_eq!(out_vec.len(), out_iter_len);
        #[rustfmt::skip]
        assert_eq!(
            out_vec,
            vec![
                [-10, 2],[-10, 3],[-10, 4],[-10, 5],[-10, 6],
                [-9, 2], [-9, 3], [-9, 4], [-9, 5], [-9, 6],
                [-8, 2], [-8, 3],          [-8, 5], [-8, 6],
                [-7, 2], [-7, 3], [-7, 4], [-7, 5], [-7, 6],
                [-6, 2], [-6, 3], [-6, 4], [-6, 5], [-6, 6]
            ]
        );

        let cord = [0, 0];
        let out = cord.moore_neighborhood(3);
        let out_iter_len = out.len();
        let out_size = out.size_hint();
        let out_vec = out.collect::<Vec<_>>();
        assert_eq!(out_vec.len(), out_size.0);
        assert_eq!(out_vec.len(), out_size.1.unwrap());
        assert_eq!(out_vec.len(), out_iter_len);
        #[rustfmt::skip]
        assert_eq!(
            out_vec,
            vec![
                [-3, -3],[-3, -2],[-3, -1],[-3, 0],[-3, 1],[-3, 2],[-3, 3],
                [-2, -3],[-2, -2],[-2, -1],[-2, 0],[-2, 1],[-2, 2],[-2, 3],
                [-1, -3],[-1, -2],[-1, -1],[-1, 0],[-1, 1],[-1, 2],[-1, 3],
                [0, -3], [0, -2], [0, -1],         [0, 1], [0, 2], [0, 3],
                [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3],
                [2, -3], [2, -2], [2, -1], [2, 0], [2, 1], [2, 2], [2, 3],
                [3, -3], [3, -2], [3, -1], [3, 0], [3, 1], [3, 2], [3, 3]
            ]
        );
    }

    #[test]
    fn neumann_neighborhood() {
        let cord = [-8, 4];
        let out = cord.neumann_neighborhood(1);
        #[rustfmt::skip]
        assert_eq!(
            out.collect::<Vec<_>>(),
            vec![
                [-9, 4],
        [-8, 3],         [-8, 5],
                [-7, 4]
            ]
        );

        let out: Vec<_> = cord.neumann_neighborhood(2).collect();
        #[rustfmt::skip]
        assert_eq!(
            out,
            vec![
                                  [-10, 4],
                         [-9, 3], [-9, 4], [-9, 5],
                [-8, 2], [-8, 3],          [-8, 5], [-8, 6],
                         [-7, 3], [-7, 4], [-7, 5],
                                  [-6, 4]
            ]
        );

        let cord = [0, 0];
        let out: Vec<_> = cord.neumann_neighborhood(3).collect();
        #[rustfmt::skip]
        assert_eq!(
            out,
            vec![
                                           [-3, 0],
                                  [-2, -1],[-2, 0], [-2, 1],
                         [-1, -2],[-1, -1],[-1, 0], [-1, 1], [-1, 2],
                [0, -3], [0, -2], [0, -1],          [0, 1],  [0, 2], [0, 3],
                         [1, -2], [1, -1], [1, 0],  [1, 1],  [1, 2],
                                  [2, -1], [2, 0],  [2, 1],
                                           [3, 0]
            ]
        );
    }

    #[test]
    fn interpolate() {
        let cord1 = [498, 4];
        let cord2 = [498, 6];
        let out: Vec<_> = cord1.interpolate(&cord2).collect();
        assert_eq!(out, vec![[498, 4], [498, 5], [498, 6]]);

        let cord1 = [498, 6];
        let cord2 = [496, 6];
        let out: Vec<_> = cord1.interpolate(&cord2).collect();
        assert_eq!(out, vec![[496, 6], [497, 6], [498, 6]]);

        let cord1 = [498, 6];
        let cord2 = [496, 7];
        let out: Vec<_> = cord1.interpolate(&cord2).collect();
        #[rustfmt::skip]
        assert_eq!(
            out,
            vec![
                [496, 6], [496, 7],
                [497, 6], [497, 7],
                [498, 6], [498, 7]
            ]
        );
    }

    #[test]
    fn offset_to_cord() {
        {
            // x x
            // x x
            // x x
            let widths = [2, 3].map(|x| NonZeroUsize::new(x).unwrap());
            assert_eq!(<[usize; 2]>::from_offset(0, widths), [0, 0]);
            assert_eq!(<[usize; 2]>::from_offset(1, widths), [1, 0]);
            assert_eq!(<[usize; 2]>::from_offset(2, widths), [0, 1]);
            assert_eq!(<[usize; 2]>::from_offset(3, widths), [1, 1]);
            assert_eq!(<[usize; 2]>::from_offset(4, widths), [0, 2]);
        }
        {
            // z = 0
            // x
            // x
            // z = 1
            //  x
            //  x
            // z = 2
            //   x
            //   x
            let widths = [1, 2, 3].map(|x| NonZeroUsize::new(x).unwrap());
            assert_eq!(<[usize; 3]>::from_offset(0, widths), [0, 0, 0]);
            assert_eq!(<[usize; 3]>::from_offset(1, widths), [0, 1, 0]);
            assert_eq!(<[usize; 3]>::from_offset(2, widths), [0, 0, 1]);
            assert_eq!(<[usize; 3]>::from_offset(3, widths), [0, 1, 1]);
        }
    }

    // overflow_checks cfg unstable
    // see https://github.com/rust-lang/rust/issues/111466, https://users.rust-lang.org/t/detecting-overflow-checks/67698
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn moore_neighborhood_panic_out_of_bounds() {
        [0u8, 0].moore_neighborhood(1).for_each(drop);
    }

    // overflow_checks cfg unstable
    // see https://github.com/rust-lang/rust/issues/111466, https://users.rust-lang.org/t/detecting-overflow-checks/67698
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn neumann_neighborhood_panic_out_of_bounds() {
        [0u8, 0].neumann_neighborhood(1).for_each(drop);
    }

    #[test]
    fn moore_neighborhood_bounded_no_panic_out_of_bounds() {
        assert_eq!([0u8, 0].moore_neighborhood_bounded(1).count(), 3);
        assert_eq!([0u8, 1].moore_neighborhood_bounded(1).count(), 5);
        assert_eq!([0u8, 2].moore_neighborhood_bounded(1).count(), 5);
        assert_eq!([0u8, 0].moore_neighborhood_bounded(2).count(), 8);
        assert_eq!([0u8, 1].moore_neighborhood_bounded(2).count(), 11);
        assert_eq!([0u8, 2].moore_neighborhood_bounded(2).count(), 14);
        assert_eq!([u8::MAX, 0].moore_neighborhood_bounded(1).count(), 3);
        assert_eq!([u8::MAX, 1].moore_neighborhood_bounded(1).count(), 5);
        assert_eq!([0, u8::MAX].moore_neighborhood_bounded(1).count(), 3);
        assert_eq!([1, u8::MAX].moore_neighborhood_bounded(1).count(), 5);
    }

    #[test]
    fn moore_neighborhood_bounded_vs_unbounded() {
        // Some random numbers that make test fast but cover many cases
        for x in 150..200u8 {
            for y in 150..200 {
                for r in 0..5 {
                    assert_eq!(
                        [x, y].moore_neighborhood(r).collect::<Vec<_>>(),
                        [x, y].moore_neighborhood_bounded(r).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn neumann_neighborhood_bounded_no_panic_out_of_bounds() {
        assert_eq!([0u8, 0].neumann_neighborhood_bounded(1).count(), 2);
        assert_eq!([0u8, 1].neumann_neighborhood_bounded(1).count(), 3);
        assert_eq!([0u8, 2].neumann_neighborhood_bounded(1).count(), 3);
        assert_eq!([0u8, 0].neumann_neighborhood_bounded(2).count(), 5);
        assert_eq!([0u8, 1].neumann_neighborhood_bounded(2).count(), 7);
        assert_eq!([0u8, 2].neumann_neighborhood_bounded(2).count(), 8);
        assert_eq!([u8::MAX, 0].neumann_neighborhood_bounded(1).count(), 2);
        assert_eq!([u8::MAX, 1].neumann_neighborhood_bounded(1).count(), 3);
        assert_eq!([0, u8::MAX].neumann_neighborhood_bounded(1).count(), 2);
        assert_eq!([1, u8::MAX].neumann_neighborhood_bounded(1).count(), 3);
    }

    #[test]
    fn neumann_neighborhood_bounded_vs_unbounded() {
        // Some random numbers that make test fast but cover many cases
        for x in 150..200u8 {
            for y in 150..200 {
                for r in 0..5 {
                    assert_eq!(
                        [x, y].neumann_neighborhood(r).collect::<Vec<_>>(),
                        [x, y].neumann_neighborhood_bounded(r).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
