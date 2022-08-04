//! Fast Transposes of flattened Arrays
//!
//! # ToDo
//!
//! - Parallel Transposes
pub mod inplace;
pub mod outofplace;
pub use inplace::ip_transpose;
pub use outofplace::oop_transpose;
