use crate::cord::Cord;
use core::{
    array,
    ops::{Add, Sub},
};
use num_traits::ToPrimitive;

/// Determines next value of products in lexicographic order.
fn next_product_iter<T, const N: usize, I>(
    mut current: [T; N],
    next_val_per_idx: &mut [I; N],
    reset_per_idx: &[I; N],
) -> Option<[T; N]>
where
    I: Iterator<Item = T> + Clone,
{
    // Start at least significant digit first.
    for i in (0..N).rev() {
        // If still new values for idx get next and return.
        if let Some(next) = next_val_per_idx[i].next() {
            current[i] = next;
            return Some(current);
        }
        // If still more to check reset it and try next
        else if i > 0 {
            next_val_per_idx[i] = reset_per_idx[i].clone();
            current[i] = next_val_per_idx[i].next().expect("Already reset iterator");
        }
    }
    // If no more to check and all are at max then there is no more.
    None
}

/// Cartesian product in lexicographical order over `N` iterators.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct CartesianProduct<I, const N: usize>
where
    I: Iterator,
{
    original_iters: [I; N],
    next_val_iters: [I; N],
    current: [I::Item; N],
}

impl<I, const N: usize> CartesianProduct<I, N>
where
    I: Iterator + Clone,
{
    /// # Panics
    /// - If an axis has 0 valid values
    pub fn new(mut values_per_axis: [I; N]) -> Self {
        let original_iters = values_per_axis.clone();
        // The length of current is N and so is values per axis. This unwrap should thus never fail unless an empty iterator is used.
        // The values_per_axis are purposefully stepped here so that the lower bound is not repeated.
        let current = array::from_fn(|i| {
            values_per_axis[i]
                .next()
                .expect("All values per axis should have at least 1 valid value.")
        });

        // Reset the least significant idx (0) so the first element is not skipped
        if let (Some(x), Some(y)) = (values_per_axis.last_mut(), original_iters.last()) {
            *x = y.clone();
        }

        Self {
            original_iters,
            next_val_iters: values_per_axis,
            current,
        }
    }
}

impl<I, const N: usize> Iterator for CartesianProduct<I, N>
where
    I: Iterator + Clone,
    I::Item: Clone,
{
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        self.current = next_product_iter(
            self.current.clone(),
            &mut self.next_val_iters,
            &self.original_iters,
        )?;
        Some(self.current.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        fn op2(
            sh1: (usize, Option<usize>),
            sh2: (usize, Option<usize>),
            mut op: impl FnMut(usize, usize) -> usize,
        ) -> (usize, Option<usize>) {
            (op(sh1.0, sh2.0), sh1.1.zip(sh2.1).map(|x| op(x.0, x.1)))
        }
        fn iter_to_array<I: Iterator, const N: usize>(mut it: I) -> [I::Item; N] {
            array::from_fn(|_| it.next().expect("len match"))
        }

        if N == 0 {
            return (0, Some(0));
        }

        let original_size_hints: [_; N] =
            iter_to_array(self.original_iters.iter().map(Iterator::size_hint));
        let next_size_hints: [_; N] =
            iter_to_array(self.next_val_iters.iter().map(Iterator::size_hint));
        let weights: [_; N] = {
            let mut weights: [_; N] = array::from_fn(|_| (0, None));
            weights[N - 1] = (1, Some(1));
            for i in (0..(N - 1)).rev() {
                weights[i] = op2(
                    weights[i + 1],
                    original_size_hints[i + 1],
                    usize::saturating_mul,
                );
            }
            weights
        };
        next_size_hints
            .into_iter()
            .zip(weights)
            .map(|(val, weight)| op2(val, weight, usize::saturating_mul))
            .reduce(|sh1, sh2| op2(sh1, sh2, usize::saturating_add))
            .expect("nonempty")
    }
}

/// Iterator over the moore neighborhood centered at some [`Cord`].
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct MooreNeighborhoodIterator<I, T, const DIM: usize> {
    /// Iterator of cord offsets from the center
    iterator: I,
    /// The center cordinate
    cord: Cord<T, DIM>,
    /// The radius used for size_hints
    radius: T,
}

impl<I, T, const DIM: usize> MooreNeighborhoodIterator<I, T, DIM> {
    pub const fn new(iterator: I, cord: Cord<T, DIM>, radius: T) -> Self {
        Self {
            iterator,
            cord,
            radius,
        }
    }
}

impl<I, T, const DIM: usize> Iterator for MooreNeighborhoodIterator<I, T, DIM>
where
    I: Iterator<Item = Cord<T, DIM>>,
    T: Add<Output = T> + Sub<Output = T> + PartialEq + Clone + ToPrimitive,
{
    type Item = Cord<T, DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        // Each radius increases number of cells in each dimension by 2 (each extent direction by 1) starting with 1 cell at radius = 1.
        while let Some(cord_offset) = self.iterator.next() {
            let smallest_neighbor = Cord(self.cord.0.clone().map(|x| x - self.radius.clone()));
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
        let radius = self.radius.to_usize().expect("Can't cast radius to usize");
        let sidelength = radius + 1 + radius;
        let volume = (0..DIM).map(|_| sidelength).product::<usize>();
        // Area or Volume minus the cell the neighborhood is for.
        (volume - 1, Some(volume - 1))
    }
}

impl<I, T, const DIM: usize> ExactSizeIterator for MooreNeighborhoodIterator<I, T, DIM>
where
    I: Iterator<Item = Cord<T, DIM>>,
    T: Add<Output = T> + Sub<Output = T> + PartialEq + Clone + ToPrimitive,
{
}

#[cfg(test)]
mod tests {
    use super::CartesianProduct;
    use core::array;

    #[test]
    fn cartesian_product_cord_size_hint() {
        fn test_it(mut it: impl Iterator + Clone) {
            for _ in 0..=u8::MAX {
                let count = it.clone().count();
                assert!(it.size_hint().0 <= count);
                assert!(it.size_hint().1.unwrap() >= count);
                it.next();
            }
        }
        test_it(CartesianProduct::new([0..2, 0..3, 0..4]));
        let filter = |x: &i32| *x % 2 == 0;
        test_it(CartesianProduct::new([
            (0..2).filter(filter),
            (0..3).filter(filter),
            (0..4).filter(filter),
        ]));
        let map = |_| "";
        test_it(CartesianProduct::new([
            (0..2).map(map),
            (0..3).map(map),
            (0..4).map(map),
        ]));
    }

    #[test]
    fn cartesian_product_size_hint_inf() {
        let its: [_; 1] = array::from_fn(|_| core::iter::repeat(0));
        let nd = CartesianProduct::new(its);
        assert_eq!(nd.size_hint().0, usize::MAX);
        assert_eq!(nd.size_hint().1, None);

        let its: [_; 0] = array::from_fn(|_| core::iter::repeat(0));
        let nd = CartesianProduct::new(its);
        assert_eq!(nd.size_hint().0, 0);
        assert_eq!(nd.size_hint().1, Some(0));
    }
}
