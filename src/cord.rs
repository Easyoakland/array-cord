use crate::iter::{CartesianProduct, MooreNeighborhoodIter, NeumannNeighborhoodIter};
use core::{
    array,
    clone::Clone,
    iter::{Iterator, Sum},
    num::NonZeroUsize,
    ops::{Add, Sub},
};
use num_iter::range_inclusive;
use num_traits::{One, ToPrimitive, Zero};

pub trait ArrayExt<T, const DIM: usize>
where
    Self: Sized,
{
    fn from_iter<I: Iterator>(mut it: I) -> [I::Item; DIM] {
        array::from_fn(|_| {
            it.next()
                .expect("iterator length should match array length")
        })
    }

    /// Elementwise application of a function on two arrays
    fn apply<O>(self, other: Self, func: impl FnMut(T, T) -> O) -> [O; DIM];

    /// The number of orthogonal moves to reach `other` from `self`.
    fn manhattan_distance(self, other: Self) -> T
    where
        T: Sum + Sub<Output = T> + PartialOrd + Clone;

    /// The square formed by the extents of the `neumann_neighborhood` not including the center.
    ///
    /// e.g. with radius `1`:
    /// ```txt
    /// x x x
    /// x c x
    /// x x x
    /// ```
    fn moore_neighborhood(&self, radius: T) -> MooreNeighborhoodIter<T, DIM>
    where
        T: Sub<Output = T> + PartialOrd + Clone + ToPrimitive + Zero + One;

    /// All cord with a manhattan distance <= `radius` from the center or less not including the center.
    ///
    /// e.g. with radius `1`:
    /// ```txt
    ///   x
    /// x c x
    ///   x
    /// ```
    fn neumann_neighborhood<'a>(&'a self, radius: T) -> NeumannNeighborhoodIter<T, DIM>
    where
        T: Sub<Output = T> + Sum + PartialOrd + Clone + ToPrimitive + Zero + One;

    /// Return an iterator over all points (inclusive) between `self` and `other`. Order is lexicographical.
    fn interpolate(&self, other: &Self) -> CartesianProduct<num_iter::RangeInclusive<T>, DIM>
    where
        T: Add<Output = T> + Ord + Clone + One + ToPrimitive;

    /// Finds the largest value in each dimension and smallest value in each dimension as the pair `(min, max)`.
    fn extents(&self, other: &Self) -> (Self, Self)
    where
        T: Ord + Clone;

    /// Finds the overall extents for many cord using [`extents`]. Handles empty iterator with [`None`].
    /// # Return
    /// `(min_per_axis, max_per_axis)`
    fn extents_iter(it: impl Iterator<Item = Self>) -> Option<(Self, Self)>
    where
        T: Ord + Clone;

    /// Find the cordinate that coresponds to a given offset where maximum width of each axis is given.
    /// Lower axis idx increment before higher axis idx.
    /// ```
    /// # use ndcord::cord::ArrayExt;
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
    fn from_offset(offset: usize, widths: [NonZeroUsize; DIM]) -> [T; DIM]
    where
        T: From<usize>;
}
impl<T, const DIM: usize> ArrayExt<T, DIM> for [T; DIM] {
    fn apply<O>(self, other: Self, mut func: impl FnMut(T, T) -> O) -> [O; DIM] {
        Self::from_iter(self.into_iter().zip(other).map(|(x, y)| func(x, y)))
    }

    fn manhattan_distance(self, other: Self) -> T
    where
        T: Sum + Sub<Output = T> + PartialOrd + Clone,
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

    fn moore_neighborhood(&self, radius: T) -> MooreNeighborhoodIter<T, DIM>
    where
        T: PartialOrd + Clone + ToPrimitive + Zero + One,
    {
        let dim_max = radius.clone() + radius.clone();

        let iterator = CartesianProduct::new(array::from_fn(|_| {
            range_inclusive(Zero::zero(), dim_max.clone())
        }));

        MooreNeighborhoodIter {
            iterator,
            cord: self.clone(),
            radius,
        }
    }

    fn neumann_neighborhood(&self, radius: T) -> NeumannNeighborhoodIter<T, DIM>
    where
        T: Sub<Output = T> + Sum + PartialOrd + Clone + ToPrimitive + Zero + One,
    {
        NeumannNeighborhoodIter {
            it: self.moore_neighborhood(radius),
        }
    }

    fn interpolate(&self, other: &Self) -> CartesianProduct<num_iter::RangeInclusive<T>, DIM>
    where
        T: Add<Output = T> + Ord + Clone + One + ToPrimitive,
    {
        // Use min and max so range doesn't silently emit no values (high..low is length 0 range)
        let ranges = array::from_fn(|i| {
            range_inclusive(
                self[i].clone().min(other[i].clone()),
                self[i].clone().max(other[i].clone()),
            )
        });
        CartesianProduct::new(ranges)
    }

    fn extents(&self, other: &Self) -> (Self, Self)
    where
        T: Ord + Clone,
    {
        let smallest = array::from_fn(|axis| self[axis].clone().min(other[axis].clone()));
        let largest = array::from_fn(|axis| self[axis].clone().max(other[axis].clone()));
        (smallest.into(), largest.into())
    }

    fn extents_iter(mut it: impl Iterator<Item = Self>) -> Option<(Self, Self)>
    where
        T: Ord + Clone,
    {
        let first = it.next()?;
        Some(it.fold((first.clone(), first), |(min, max), x| {
            (x.extents(&min).0, x.extents(&max).1)
        }))
    }

    fn from_offset(mut offset: usize, widths: [NonZeroUsize; DIM]) -> [T; DIM]
    where
        T: From<usize>,
    {
        let mut out = [0; DIM];
        for axis in (0..DIM).rev() {
            let next_lowest_axis_width = axis.checked_sub(1);
            out[axis] = match next_lowest_axis_width {
                Some(x) => offset / widths[x],
                None => offset,
            };
            if next_lowest_axis_width.is_some() {
                offset -= out[axis] * <usize as From<_>>::from(widths[axis - 1]);
            }
        }
        out.map(Into::into).into()
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
        assert_eq!(
            out.collect::<Vec<_>>(),
            vec![[-9, 4], [-8, 3], [-8, 5], [-7, 4]]
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
}
