// ══════════════════════════════════════════════════════════════════════
// bitarena — Bitset-Accelerated Generational Arena
// ══════════════════════════════════════════════════════════════════════
//
// ARCHITECTURE OVERVIEW
// ─────────────────────
// This crate implements a generational arena using Struct-of-Arrays (SoA)
// layout with a bitset occupancy tracker. This is the same pattern used
// internally by ECS frameworks like Bevy, but packaged as a standalone,
// dependency-free data structure.
//
// WHY THIS EXISTS (The Problem with Existing Arenas)
// ──────────────────────────────────────────────────
// thunderdome and generational-arena use Array-of-Structs (AoS):
//
//   Vec<Entry<T>>  where  Entry<T> = Occupied(gen, T) | Empty(gen, next)
//
// This means:
//   1. Every slot pays the cost of an enum discriminant (tag byte + padding)
//   2. Iterating loads EVERY entry (including empty ones) into cache
//   3. For large T, empty slots waste enormous cache bandwidth
//   4. Branch prediction suffers on sparse arenas (random occupied/empty pattern)
//
// OUR DESIGN (Struct-of-Arrays + Bitset)
// ──────────────────────────────────────
//   occupancy:   Vec<u64>            — 1 bit per slot, 64 slots per word
//   generations: Vec<u32>            — one generation counter per slot
//   values:      Vec<MaybeUninit<T>> — raw storage, only valid when bit is set
//   free_list:   Vec<u32>            — stack of free slot indices
//
// Benefits:
//   1. Iteration scans a tiny bitset (10k slots = 157 bytes) instead of
//      loading 10k × sizeof(Entry<T>) bytes
//   2. Uses hardware tzcnt/blsr instructions for branchless bit scanning
//   3. Cache lines during iteration contain ONLY occupancy data (no value pollution)
//   4. Values array has zero overhead per empty slot (just uninitialized memory)
//   5. Trivially parallelizable (each u64 word is independent → rayon)
//
// SAFETY PHILOSOPHY
// ─────────────────
// This crate uses `unsafe` in a small number of carefully isolated locations.
// The central safety invariant is:
//
//   ┌──────────────────────────────────────────────────────────────────┐
//   │ values[i] is initialized if and only if the occupancy bit for   │
//   │ slot i is set. Every method that reads a value MUST check the   │
//   │ bit first. Every method that sets a bit MUST initialize the     │
//   │ value first. Every method that clears a bit MUST drop/read the  │
//   │ value first.                                                    │
//   └──────────────────────────────────────────────────────────────────┘
//
// All unsafe code is:
//   - Annotated with // SAFETY: comments explaining the invariant
//   - Tested under miri (cargo +nightly miri test)
//   - Property-tested against a safe HashMap-based oracle
//
// ══════════════════════════════════════════════════════════════════════

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
