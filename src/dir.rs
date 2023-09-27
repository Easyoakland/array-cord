use crate::cord::Cord;
use core::{
    cmp::PartialEq,
    ops::{MulAssign, Neg},
};
use num_traits::{One, Zero};

pub type Velocity<T> = Cord<T, 2>;

/// Direction where the discriminant represents the number of clockwise turns from the right.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, enum_iterator::Sequence)]
pub enum Dir {
    Right = 0,
    Down = 1,
    Left = 2,
    Up = 3,
}

impl Dir {
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // doesn't panic
    pub fn rotate(self, rotation: Rotation) -> Self {
        match rotation {
            Rotation::Right => enum_iterator::next_cycle(&self).expect(">0 variants"),
            Rotation::Left => enum_iterator::previous_cycle(&self).expect(">0 variants"),
        }
    }

    /// Converts the direction to velocity with the assumption that Right is increasing x and Down is increasing y (and vice versa).
    #[must_use]
    pub fn to_velocity<T>(self) -> Velocity<T>
    where
        T: Zero + One + Neg<Output = T>,
    {
        match self {
            Dir::Right => [T::one(), T::zero()],
            Dir::Down => [T::zero(), T::one()],
            Dir::Left => [-T::one(), T::zero()],
            Dir::Up => [T::zero(), -T::one()],
        }
        .into()
    }

    /// Converts from a velocity to direction with the assumption that Right is increasing x and Down is increasing y (and vice versa).
    /// Additionally assumes that velocity has a magnitude of `1` or `0` in each dimension.
    ///
    /// # Panics
    /// - velocity is not `[1,0]` rotated by some integer multiple of `pi/2` amount.
    pub fn from_velocity<T: Zero + One + Neg<Output = T> + PartialEq>(
        velocity: Velocity<T>,
    ) -> Self {
        match velocity.0 {
            x if x == [T::one(), T::zero()] => Self::Right,
            x if x == [T::zero(), T::one()] => Self::Down,
            x if x == [-T::one(), T::zero()] => Self::Left,
            x if x == [T::zero(), -T::one()] => Self::Up,
            _ => panic!("Invalid velocity"),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Rotation {
    /// Counter-clockwise
    Left,
    /// Clockwise
    Right,
}

pub fn rotate<T: MulAssign + One + Neg<Output = T>>(dir: &mut Velocity<T>, rotation: &Rotation) {
    dir.0.swap(0, 1);
    match rotation {
        // 1,0 -> 0,1 -> -1,0 -> 0,-1
        Rotation::Right => dir[0] *= -T::one(),
        // 1,0 -> 0,-1 -> -1,0 -> 0,1
        Rotation::Left => dir[1] *= -T::one(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Dir, Rotation};

    #[test]
    fn direction_rotation() {
        assert_eq!(Dir::Right.rotate(Rotation::Right), Dir::Down);
        assert_eq!(Dir::Down.rotate(Rotation::Right), Dir::Left);
        assert_eq!(Dir::Left.rotate(Rotation::Right), Dir::Up);
        assert_eq!(Dir::Up.rotate(Rotation::Right), Dir::Right);

        assert_eq!(Dir::Right.rotate(Rotation::Left), Dir::Up);
        assert_eq!(Dir::Up.rotate(Rotation::Left), Dir::Left);
        assert_eq!(Dir::Left.rotate(Rotation::Left), Dir::Down);
        assert_eq!(Dir::Down.rotate(Rotation::Left), Dir::Right);
    }
}
