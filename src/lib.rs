//! # slam-core
//!
//! Simultaneous Localization and Mapping (SLAM): occupancy grid mapping,
//! particle filter localization, landmark detection. Pure Rust, no external dependencies.

pub mod occupancy;
pub mod particle;
pub mod landmark;
pub mod map;
pub mod pose;
