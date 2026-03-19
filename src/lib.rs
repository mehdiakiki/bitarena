//! # bitarena
//!
//! A bitset-accelerated generational arena optimized for sparse iteration.
//!
//! ## Architecture
//!
//! This crate implements a generational arena using Struct-of-Arrays (SoA)
//! layout with a bitset occupancy tracker. This is the same pattern used
//! internally by ECS frameworks like Bevy, but packaged as a standalone,
//! dependency-free data structure.
//!
//! ## Why this exists
//!
//! `thunderdome` and `generational-arena` use Array-of-Structs (AoS):
//!
//! ```text
//! Vec<Entry<T>>  where  Entry<T> = Occupied(gen, T) | Empty(gen, next)
//! ```
//!
//! This means:
//! 1. Every slot pays the cost of an enum discriminant (tag byte + padding)
//! 2. Iterating loads every entry (including empty ones) into cache
//! 3. For large `T`, empty slots waste enormous cache bandwidth
//! 4. Branch prediction suffers on sparse arenas (random occupied/empty pattern)
//!
//! ## Design (Struct-of-Arrays + Bitset)
//!
//! ```text
//! occupancy:   Vec<u64>            — 1 bit per slot, 64 slots per word
//! generations: Vec<u32>            — one generation counter per slot
//! values:      Vec<MaybeUninit<T>> — raw storage, only valid when bit is set
//! free_list:   Vec<u32>            — stack of free slot indices
//! ```
//!
//! Benefits:
//! 1. Iteration scans a tiny bitset (10k slots = 157 bytes) instead of
//!    loading 10k × `size_of::<Entry<T>>()` bytes
//! 2. Uses hardware `tzcnt`/`blsr` instructions for branchless bit scanning
//! 3. Cache lines during iteration contain only occupancy data (no value pollution)
//! 4. Values array has zero overhead per empty slot (just uninitialized memory)
//! 5. Trivially parallelizable (each `u64` word is independent — rayon)
//!
//! ## Quick start
//!
//! ```
//! use bitarena::{Arena, Index};
//!
//! let mut arena = Arena::new();
//! let idx: Index = arena.insert("hello");
//! assert_eq!(arena.get(idx), Some(&"hello"));
//! assert_eq!(arena.remove(idx), Some("hello"));
//! assert_eq!(arena.get(idx), None);
//! ```
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std`   | yes     | Enables `std::error::Error` impls |
//! | `rayon` | no      | Parallel iteration via rayon |
//! | `serde` | no      | Serialize/deserialize support |
//!
//! For `no_std`, disable default features:
//!
//! ```toml
//! [dependencies]
//! bitarena = { version = "0.1", default-features = false }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

// ──────────────────────────────────────────────────────────────────────
// no_std SUPPORT NOTES
// ──────────────────────────────────────────────────────────────────────
// We need `alloc` for Vec and Box. In no_std environments, the user must
// provide a global allocator. This is standard practice for embedded Rust.
//
// We NEVER use: std::collections, std::io, std::fs, std::thread, println!
// We ONLY use: alloc::vec::Vec, alloc::boxed::Box, core::*
//
// The `std` feature exists solely to implement std-specific traits
// (like std::error::Error) if the user wants them.
// ──────────────────────────────────────────────────────────────────────

extern crate alloc;

mod arena;
mod bitset;
mod free_list;
mod index;
mod iter;

#[cfg(feature = "serde")]
mod serde_impl;

pub use arena::Arena;
pub use index::Index;
pub use iter::{Drain, IntoIter, Iter, IterMut, Keys, Values, ValuesMut};

// ──────────────────────────────────────────────────────────────────────
// SERDE WIRE FORMAT (feature = "serde")
// ──────────────────────────────────────────────────────────────────────
// Arena<T> serializes as a JSON array (or equivalent) of entry objects:
//
//   [
//     { "slot": 0, "generation": 1, "value": <T> },
//     { "slot": 3, "generation": 2, "value": <T> },
//     ...
//   ]
//
// Only occupied entries are emitted (sparse arenas serialize small).
// Slot order in the sequence matches arena iteration order (ascending slot).
//
// Index serializes as a single u64 with the layout:
//   bits 63..32 — generation (NonZeroU32, upper half)
//   bits 31..0  — slot (u32, lower half)
// This matches the `Index::to_bits()` / `Index::from_bits()` encoding and
// is stable across all semver-compatible versions.
// ──────────────────────────────────────────────────────────────────────
