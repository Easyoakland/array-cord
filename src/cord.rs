use crate::iter::NDCartesianProduct;
use num_iter::range_inclusive;
use num_traits::{cast, NumCast, One, ToPrimitive, Zero};
use std::{
    array,
    clone::Clone,
    cmp::PartialEq,
    fmt::{Debug, Display},
    iter::{Iterator, Sum},
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

pub fn abs_diff<T: Sub<Output = T> + PartialOrd>(x: T, y: T) -> T {
    if x >= y {
        x - y
    } else {
        y - x
    }
}

/// Newtype on n dimensional arrays representing coordinates in a grid-like space.
///
/// Capable of things like neighborhood calculation, cordinate addition, interpolation, etc...
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cord<T, const DIM: usize>(pub [T; DIM]);

impl<T: Display, const DIM: usize> Display for Cord<T, DIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        if self.0.len() > 0 {
            for e in &self.0[..self.0.len() - 1] {
                write!(f, "{}, ", e)?;
            }
            {
                write!(f, "{}", self.0.last().unwrap())?;
            }
        }
        write!(f, "]")
    }
}

impl<T: Default + Copy, const DIM: usize> Default for Cord<T, DIM> {
    fn default() -> Self {
        Self([Default::default(); DIM])
    }
}

impl<T, const DIM: usize> Add for Cord<T, DIM>
where
    T: Add<Output = T>,
{
    type Output = Cord<T, DIM>;

    fn add(self, rhs: Self) -> Self::Output {
        self.apply(rhs, T::add)
    }
}

impl<T, const DIM: usize> AddAssign for Cord<T, DIM>
where
    T: Add<Output = T> + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<T, const DIM: usize> Sub for Cord<T, DIM>
where
    T: Sub<Output = T>,
{
    type Output = Cord<T, DIM>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.apply(rhs, T::sub)
    }
}

impl<T, const DIM: usize> SubAssign for Cord<T, DIM>
where
    T: Sub<Output = T> + Clone,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs
    }
}

impl<T, const DIM: usize> Mul<T> for Cord<T, DIM>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        array::from_fn(|i| self.0[i].clone() * rhs.clone()).into()
    }
}

impl<T, const DIM: usize> MulAssign<T> for Cord<T, DIM>
where
    T: Mul<Output = T> + Clone,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs
    }
}

impl<T, const DIM: usize> Div<T> for Cord<T, DIM>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        array::from_fn(|i| self.0[i].clone() / rhs.clone()).into()
    }
}

impl<T, const DIM: usize> DivAssign<T> for Cord<T, DIM>
where
    T: Div<Output = T> + Clone,
{
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone() / rhs
    }
}

/// Iterator over the moore neighborhood centered at some cord.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct MooreNeighborhoodIterator<I, T, const DIM: usize> {
    iterator: I,
    cord: Cord<T, DIM>,
    radius: usize,
}

impl<I, T, const DIM: usize> Iterator for MooreNeighborhoodIterator<I, T, DIM>
where
    I: Iterator<Item = Cord<T, DIM>>,
    T: Add<Output = T> + Sub<Output = T> + PartialEq + Clone + NumCast,
{
    type Item = Cord<T, DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        // Each radius increases number of cells in each dimension by 2 (each extent direction by 1) starting with 1 cell at radius = 1.
        while let Some(cord_offset) = self.iterator.next() {
            let smallest_neighbor = Cord(self.cord.0.clone().map(|x| {
                x - cast(self.radius).expect("Can't cast the radius to cord's datatype.")
            }));
            let new_cord = smallest_neighbor + cord_offset;

            // Don't add self to neighbor list.
            if new_cord == self.cord {
                continue;
            }

            return Some(new_cord);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let sidelength = self.radius + 1 + self.radius;
        let volume = (0..DIM).map(|_| sidelength).product::<usize>();
        // Area or Volume minus the cell the neighborhood is for.
        (volume - 1, Some(volume - 1))
    }
}

impl<I, T, const DIM: usize> ExactSizeIterator for MooreNeighborhoodIterator<I, T, DIM>
where
    I: Iterator<Item = Cord<T, DIM>>,
    T: Add<Output = T> + Sub<Output = T> + PartialEq + Clone + NumCast,
{
}

impl<T, const DIM: usize> Cord<T, DIM> {
    pub fn apply<O>(self, other: Self, func: impl Fn(T, T) -> O) -> Cord<O, DIM> {
        let mut other = other.0.into_iter();
        Cord(
            self.0
                .map(|x| func(x, other.next().expect("Same length arrays"))),
        )
    }
    pub fn manhattan_distance(self, other: &Self) -> T
    where
        T: Sum + Sub<Output = T> + PartialOrd + Clone,
    {
        let diff_per_axis = self.apply(other.clone(), abs_diff::<T>);
        diff_per_axis.0.into_iter().sum()
    }

    /// Radius is manhattan distance from center to edge.
    /// Moore neighborhood is a square formed by the extents of the Neumann neighborhood.
    pub fn moore_neighborhood(
        &self,
        radius: usize,
    ) -> MooreNeighborhoodIterator<impl Iterator<Item = Cord<T, DIM>> + Clone, T, DIM>
    where
        T: Add<Output = T> + PartialOrd + Clone + NumCast + Zero + One,
    {
        let dim_max = cast::<usize, T>(radius + radius)
            .expect("Can't convert radius + radius into cord's datatype.");

        let iterator = NDCartesianProduct::new(array::from_fn(|_| {
            range_inclusive(Zero::zero(), dim_max.clone())
        }))
        .map(Cord);

        MooreNeighborhoodIterator {
            iterator,
            cord: self.clone(),
            radius,
        }
    }

    /// Radius is manhattan distance of furthest neighbors.
    /// Neumann neighborhood is all cells a manhattan distance of the radius or smaller.
    pub fn neumann_neighborhood(&self, radius: usize) -> impl Iterator<Item = Cord<T, DIM>> + '_
    where
        T: Sub<Output = T> + Sum + PartialOrd + Clone + NumCast + Zero + One,
    {
        let neighbors = self.moore_neighborhood(radius);
        neighbors.filter(move |x| {
            x.clone().manhattan_distance(self)
                <= cast(radius).expect("Can't convert radius to cord's datatype.")
        })
    }

    /// Return an iterator over all points (inclusive) between `self` and `other`. Order is lexicographical.
    pub fn interpolate(&self, other: &Self) -> impl Iterator<Item = Cord<T, DIM>>
    where
        T: Add<Output = T> + Ord + Clone + One + ToPrimitive,
    {
        // Use min and max so range doesn't silently emit no values (high..low is length 0 range)
        let ranges = array::from_fn(|i| {
            range_inclusive(
                self.0[i].clone().min(other.0[i].clone()),
                self.0[i].clone().max(other.0[i].clone()),
            )
        });
        NDCartesianProduct::new(ranges).map(Cord)
    }

    /// Finds the largest value in each dimension and smallest value in each dimension as the pair `(min, max)`.
    pub fn extents(&self, other: &Self) -> (Self, Self)
    where
        T: Ord + Clone,
    {
        let smallest = array::from_fn(|axis| self.0[axis].clone().min(other.0[axis].clone()));
        let largest = array::from_fn(|axis| self.0[axis].clone().max(other.0[axis].clone()));
        (smallest.into(), largest.into())
    }

    /// Finds the overall extents for many [`NDCord`] with [`NDCord::extents`]. Handles empty iterator with [`None`].
    /// # Return
    /// `(min_per_axis, max_per_axis)`
    pub fn extents_iter(mut it: impl Iterator<Item = Self>) -> Option<(Self, Self)>
    where
        T: Ord + Clone,
    {
        let first = it.next()?;
        Some(it.fold((first.clone(), first), |(min, max), x| {
            (x.extents(&min).0, x.extents(&max).1)
        }))
    }

    /// Find the cordinate that coresponds to a given offset where maximum width of each axis is given.
    /// Fills values of coordinates from greatest index to least.
    /// ```
    /// # use ndcord::cord::Cord;
    /// # use core::num::NonZeroUsize;
    /// // x x
    /// // x x
    /// // x x
    /// let widths = [2, 3].map(|x| NonZeroUsize::new(x).unwrap());
    /// assert_eq!(Cord::from_offset(0, widths), Cord([0, 0]));
    /// assert_eq!(Cord::from_offset(1, widths), Cord([1, 0]));
    /// assert_eq!(Cord::from_offset(2, widths), Cord([0, 1]));
    /// assert_eq!(Cord::from_offset(3, widths), Cord([1, 1]));
    /// assert_eq!(Cord::from_offset(4, widths), Cord([0, 2]));
    /// ```
    pub fn from_offset(mut offset: usize, widths: [NonZeroUsize; DIM]) -> Cord<T, DIM>
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

impl<T, const DIM: usize> From<[T; DIM]> for Cord<T, DIM> {
    fn from(value: [T; DIM]) -> Self {
        Cord(value)
    }
}

impl<T, const DIM: usize> Cord<T, DIM> {
    /// Same as [`From`] impl but const.
    pub const fn new(value: [T; DIM]) -> Self {
        Cord(value)
    }
}

impl<T, const DIM: usize> From<Cord<T, DIM>> for [T; DIM] {
    fn from(value: Cord<T, DIM>) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manhattan_distance() {
        let cord1 = Cord([-2isize, 4]);
        let cord2 = Cord([498, 6]);
        let out = cord1.manhattan_distance(&cord2);
        assert_eq!(out, 500 + 2);
    }

    #[test]
    fn moore_neighborhood() {
        let cord = Cord([0]);
        let out = cord.moore_neighborhood(1);
        let out_iter_len = out.len();
        let out_size = out.size_hint();
        let out_vec = out.collect::<Vec<_>>();
        assert_eq!(out_vec, vec![Cord([-1]), Cord([1])]);
        assert_eq!(out_vec.len(), out_size.0);
        assert_eq!(out_vec.len(), out_size.1.unwrap());
        assert_eq!(out_vec.len(), out_iter_len);

        let cord = Cord([-8, 4]);
        let out = cord.moore_neighborhood(1);
        let out_iter_len = out.len();
        let out_size = out.size_hint();
        let out_vec = out.collect::<Vec<_>>();
        assert_eq!(
            out_vec,
            vec![
                Cord([-9, 3]),
                Cord([-9, 4]),
                Cord([-9, 5]),
                Cord([-8, 3]),
                Cord([-8, 5]),
                Cord([-7, 3]),
                Cord([-7, 4]),
                Cord([-7, 5])
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
                Cord([-10, 2]),Cord([-10, 3]),Cord([-10, 4]),Cord([-10, 5]),Cord([-10, 6]),
                Cord([-9, 2]), Cord([-9, 3]), Cord([-9, 4]), Cord([-9, 5]), Cord([-9, 6]),
                Cord([-8, 2]), Cord([-8, 3]),                  Cord([-8, 5]), Cord([-8, 6]),
                Cord([-7, 2]), Cord([-7, 3]), Cord([-7, 4]), Cord([-7, 5]), Cord([-7, 6]),
                Cord([-6, 2]), Cord([-6, 3]), Cord([-6, 4]), Cord([-6, 5]), Cord([-6, 6])
            ]
        );

        let cord = Cord([0, 0]);
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
                Cord([-3, -3]),Cord([-3, -2]),Cord([-3, -1]),Cord([-3, 0]),Cord([-3, 1]),Cord([-3, 2]),Cord([-3, 3]),
                Cord([-2, -3]),Cord([-2, -2]),Cord([-2, -1]),Cord([-2, 0]),Cord([-2, 1]),Cord([-2, 2]),Cord([-2, 3]),
                Cord([-1, -3]),Cord([-1, -2]),Cord([-1, -1]),Cord([-1, 0]),Cord([-1, 1]),Cord([-1, 2]),Cord([-1, 3]),
                Cord([0, -3]), Cord([0, -2]), Cord([0, -1]),                 Cord([0, 1]), Cord([0, 2]), Cord([0, 3]),
                Cord([1, -3]), Cord([1, -2]), Cord([1, -1]), Cord([1, 0]), Cord([1, 1]), Cord([1, 2]), Cord([1, 3]),
                Cord([2, -3]), Cord([2, -2]), Cord([2, -1]), Cord([2, 0]), Cord([2, 1]), Cord([2, 2]), Cord([2, 3]),
                Cord([3, -3]), Cord([3, -2]), Cord([3, -1]), Cord([3, 0]), Cord([3, 1]), Cord([3, 2]), Cord([3, 3])
            ]
        );
    }

    #[test]
    fn neumann_neighborhood() {
        let cord = Cord([-8, 4]);
        let out = cord.neumann_neighborhood(1);
        assert_eq!(
            out.collect::<Vec<_>>(),
            vec![Cord([-9, 4]), Cord([-8, 3]), Cord([-8, 5]), Cord([-7, 4])]
        );

        let out: Vec<_> = cord.neumann_neighborhood(2).collect();
        #[rustfmt::skip]
        assert_eq!(
            out,
            vec![
                                                  Cord([-10, 4]),
                                 Cord([-9, 3]), Cord([-9, 4]), Cord([-9, 5]),
                Cord([-8, 2]), Cord([-8, 3]),                  Cord([-8, 5]), Cord([-8, 6]),
                                 Cord([-7, 3]), Cord([-7, 4]), Cord([-7, 5]),
                                                  Cord([-6, 4])
            ]
        );

        let cord = Cord([0, 0]);
        let out: Vec<_> = cord.neumann_neighborhood(3).collect();
        #[rustfmt::skip]
        assert_eq!(
            out,
            vec![
                                                                   Cord([-3, 0]),
                                                  Cord([-2, -1]),Cord([-2, 0]), Cord([-2, 1]),
                                 Cord([-1, -2]),Cord([-1, -1]),Cord([-1, 0]), Cord([-1, 1]), Cord([-1, 2]),
                Cord([0, -3]), Cord([0, -2]), Cord([0, -1]),                  Cord([0, 1]),  Cord([0, 2]), Cord([0, 3]),
                                 Cord([1, -2]), Cord([1, -1]), Cord([1, 0]),  Cord([1, 1]),  Cord([1, 2]),
                                                  Cord([2, -1]), Cord([2, 0]),  Cord([2, 1]),
                                                                   Cord([3, 0])
            ]
        );
    }

    #[test]
    fn interpolate() {
        let cord1 = Cord([498, 4]);
        let cord2 = Cord([498, 6]);
        let out: Vec<_> = cord1.interpolate(&cord2).collect();
        assert_eq!(out, vec![Cord([498, 4]), Cord([498, 5]), Cord([498, 6])]);

        let cord1 = Cord([498, 6]);
        let cord2 = Cord([496, 6]);
        let out: Vec<_> = cord1.interpolate(&cord2).collect();
        assert_eq!(out, vec![Cord([496, 6]), Cord([497, 6]), Cord([498, 6])]);

        let cord1 = Cord([498, 6]);
        let cord2 = Cord([496, 7]);
        let out: Vec<_> = cord1.interpolate(&cord2).collect();
        assert_eq!(
            out,
            vec![
                Cord([496, 6]),
                Cord([496, 7]),
                Cord([497, 6]),
                Cord([497, 7]),
                Cord([498, 6]),
                Cord([498, 7])
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
            assert_eq!(Cord::from_offset(0, widths), Cord([0, 0]));
            assert_eq!(Cord::from_offset(1, widths), Cord([1, 0]));
            assert_eq!(Cord::from_offset(2, widths), Cord([0, 1]));
            assert_eq!(Cord::from_offset(3, widths), Cord([1, 1]));
            assert_eq!(Cord::from_offset(4, widths), Cord([0, 2]));
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
            assert_eq!(Cord::from_offset(0, widths), Cord([0, 0, 0]));
            assert_eq!(Cord::from_offset(1, widths), Cord([0, 1, 0]));
            assert_eq!(Cord::from_offset(2, widths), Cord([0, 0, 1]));
            assert_eq!(Cord::from_offset(3, widths), Cord([0, 1, 1]));
        }
    }
}
