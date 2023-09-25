use crate::cord::Cord;
use num_traits::ToPrimitive;
use std::{
    array,
    ops::{Add, Sub},
};

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

/// Cartesian product in lexicographical order over N iterators.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct NDCartesianProduct<I, const N: usize>
where
    I: Iterator,
{
    original_iters: [I; N],
    next_val_iters: [I; N],
    current: [I::Item; N],
}

impl<I, const N: usize> NDCartesianProduct<I, N>
where
    I: Iterator + Clone,
{
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

        NDCartesianProduct {
            original_iters,
            next_val_iters: values_per_axis,
            current,
        }
    }
}

impl<I: Iterator + Clone, const N: usize> Iterator for NDCartesianProduct<I, N>
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
}

/// Iterator over the moore neighborhood centered at some cord.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct MooreNeighborhoodIterator<I, T, const DIM: usize> {
    iterator: I,
    cord: Cord<T, DIM>,
    radius: T,
}

impl<I, T, const DIM: usize> MooreNeighborhoodIterator<I, T, DIM> {
    pub fn new(iterator: I, cord: Cord<T, DIM>, radius: T) -> Self {
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
