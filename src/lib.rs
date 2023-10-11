// #![cfg_attr(not(feature = "std"), no_std)] // TODO re-add after removing Box
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

pub mod cord;
pub mod iter;
