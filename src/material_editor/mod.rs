//! High-level helpers for tooling that edits material databases.
//!
//! This module is intentionally separate from the low-level parsing and runtime
//! systems so that GUI/front-end tooling can evolve independently while still
//! sharing serialization logic with the rest of the crate.

pub mod io;
pub mod preview;
pub mod project;
pub mod ui;
