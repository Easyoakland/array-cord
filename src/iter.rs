use std::array;

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
