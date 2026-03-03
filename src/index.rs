// ══════════════════════════════════════════════════════════════════════
// Index — The Arena Key Type
// ══════════════════════════════════════════════════════════════════════
//
// DESIGN DECISIONS
// ────────────────
//
// 1. SIZE: 8 bytes (u32 slot + u32 generation), same as thunderdome.
//    This is critical for API compatibility. Users store millions of
//    indices in their own data structures — every byte matters.
//
// 2. OPTION NICHE: Option<Index> must also be 8 bytes (not 12 or 16).
//    thunderdome achieves this by using NonZeroU32 for generation.
//    We do the same. Generation starts at 1, wraps from u32::MAX to 1.
//    The zero bit pattern means "None" for free.
//
//    This matters because game engines store indices in components:
//      struct Enemy { target: Option<Index> }
//    If Option<Index> bloats to 16 bytes, nobody will use this crate.
//
// 3. SLOT vs GENERATION SPLIT:
//    - Slot (u32): physical position in the arena arrays. Max 2^32 slots.
//    - Generation (NonZeroU32): increments each time a slot is reused.
//      Prevents the ABA problem: if entity A is removed and entity B
//      takes slot 5, the old Index{slot:5, gen:1} won't match gen:2.
//
// 4. GENERATION OVERFLOW: When generation hits u32::MAX, it wraps to 1
//    (not 0, because 0 is reserved for the Option niche). In theory this
//    means after 4 billion reuses of the same slot, a stale index could
//    collide. In practice, this never happens.
//
// 5. BITS ROUND-TRIP: to_bits() / from_bits() encode the full Index as
//    a u64. This is important for FFI (passing to C, storing in databases,
//    sending over network). Thunderdome has this; we must too.
//
// 6. TRAIT IMPLS:
//    - Copy + Clone: Indices are just two integers, always copyable
//    - Eq + Hash: For use as HashMap/HashSet keys
//    - Ord: For use in BTreeMap or sorted collections
//    - Debug: For error messages (but NOT Display — indices aren't user-facing)
//
// ══════════════════════════════════════════════════════════════════════

use core::fmt;
use core::num::NonZeroU32;

/// A handle to an element in an [`Arena`](crate::Arena).
///
/// An `Index` consists of a slot (physical position) and a generation
/// (version counter). The generation prevents stale indices from
/// accessing reallocated slots — the [ABA problem](https://en.wikipedia.org/wiki/ABA_problem).
///
/// # Size
///
/// `Index` is 8 bytes. `Option<Index>` is also 8 bytes thanks to
/// niche optimization on the generation field.
///
/// ```
/// # use bitarena::Index;
/// assert_eq!(std::mem::size_of::<Index>(), 8);
/// assert_eq!(std::mem::size_of::<Option<Index>>(), 8);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Index {
    // ──────────────────────────────────────────────────────────────────
    // FIELD ORDERING NOTE
    // ──────────────────────────────────────────────────────────────────
    // slot comes first so that Index sorts by slot first, then generation.
    // This matches the physical layout in the arena arrays and gives
    // better cache behavior when sorting a collection of indices.
    // ──────────────────────────────────────────────────────────────────
    pub(crate) slot: u32,
    pub(crate) generation: NonZeroU32,
}

impl Index {
    /// A sentinel index that is extremely unlikely to be valid in any arena.
    /// Useful for two-phase initialization patterns where you need a placeholder.
    ///
    /// **Prefer `Option<Index>` over using this sentinel.** This exists only
    /// for compatibility with code that cannot use Option (e.g., certain FFI
    /// patterns or ECS frameworks that require a concrete type).
    ///
    /// # Note on Safety
    /// Using DANGLING to access an arena will always return None — it is
    /// never UB, just always invalid. The slot is u32::MAX and the generation
    /// is u32::MAX, which would require an arena with 4 billion slots AND
    /// 4 billion reuses of the last slot to accidentally collide.
    pub const DANGLING: Self = Self {
        slot: u32::MAX,
        // SAFETY: u32::MAX is not zero.
        generation: unsafe { NonZeroU32::new_unchecked(u32::MAX) },
    };

    // ──────────────────────────────────────────────────────────────────
    // BITS ENCODING
    // ──────────────────────────────────────────────────────────────────
    // Layout: [generation: upper 32 bits] [slot: lower 32 bits]
    //
    // This matches thunderdome's encoding for maximum interoperability.
    // If someone migrates from thunderdome, their serialized indices
    // will still decode correctly.
    //
    // IMPORTANT: This encoding must be stable across semver-compatible
    // versions. Once we ship 0.1.0, we cannot change this layout
    // without a major version bump.
    // ──────────────────────────────────────────────────────────────────

    /// Encode this index as a `u64` for storage, FFI, or serialization.
    ///
    /// The encoding is stable across all semver-compatible versions.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        ((self.generation.get() as u64) << 32) | (self.slot as u64)
    }

    /// Decode an index from bits created by [`to_bits`](Index::to_bits).
    ///
    /// Returns `None` if the generation portion is zero (invalid encoding).
    #[inline]
    pub const fn from_bits(bits: u64) -> Option<Self> {
        let gen_raw = (bits >> 32) as u32;
        match NonZeroU32::new(gen_raw) {
            Some(generation) => Some(Self {
                slot: bits as u32,
                generation,
            }),
            None => None,
        }
    }

    /// The slot (physical position) of this index.
    #[inline]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// The generation (version) of this index.
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation.get()
    }
}

// ──────────────────────────────────────────────────────────────────────
// GENERATION HELPER
// ──────────────────────────────────────────────────────────────────────
// These are crate-internal helpers. We don't expose generation
// manipulation to users — they only see the Index.
// ──────────────────────────────────────────────────────────────────────

/// Increment a generation, wrapping from MAX to 1 (skipping 0).
///
/// Why skip 0? Because NonZeroU32 uses 0 as the niche for Option<Index>.
/// If we allowed generation 0, Option<Index> would grow from 8 to 12 bytes.
#[cfg(test)]
#[inline]
pub(crate) fn next_generation(gen: NonZeroU32) -> NonZeroU32 {
    let next = gen.get().wrapping_add(1);
    // If wrapping_add produced 0, use 1 instead.
    // SAFETY: The max of (next, 1) is always >= 1, which is not zero.
    unsafe { NonZeroU32::new_unchecked(if next == 0 { 1 } else { next }) }
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ──────────────────────────────────────────────────────────────
        // FORMAT NOTE: We show slot:generation for easy debugging.
        // Example: Index(slot: 42, gen: 3)
        //
        // We intentionally do NOT implement Display. Indices are internal
        // handles, not user-facing strings. Using Debug forces users to
        // use {:?} which signals "this is a programmer-facing value."
        // ──────────────────────────────────────────────────────────────
        f.debug_struct("Index")
            .field("slot", &self.slot)
            .field("gen", &self.generation.get())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    // ──────────────────────────────────────────────────────────────────
    // SIZE TESTS — These are compile-time contracts.
    // If any of these fail, the crate is broken for every user.
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn index_is_8_bytes() {
        assert_eq!(size_of::<Index>(), 8);
    }

    #[test]
    fn option_index_is_8_bytes() {
        // This is THE critical niche optimization test.
        // If this fails, we broke the NonZeroU32 niche and every user
        // storing Option<Index> in their structs will have bloated layouts.
        assert_eq!(size_of::<Option<Index>>(), 8);
    }

    #[test]
    fn bits_roundtrip() {
        let idx = Index {
            slot: 0xDEAD_BEEF,
            generation: NonZeroU32::new(0x1BAD_CAFE).unwrap(),
        };
        let bits = idx.to_bits();
        let decoded = Index::from_bits(bits).unwrap();
        assert_eq!(idx, decoded);
    }

    #[test]
    fn bits_zero_generation_returns_none() {
        // Lower 32 bits = valid slot, upper 32 bits = 0 (invalid generation)
        assert!(Index::from_bits(0x0000_0000_DEAD_BEEF).is_none());
    }

    #[test]
    fn generation_wraps_from_max_to_1() {
        let max = NonZeroU32::new(u32::MAX).unwrap();
        let next = next_generation(max);
        assert_eq!(next.get(), 1); // Wraps to 1, not 0
    }

    #[test]
    fn generation_increments_normally() {
        let gen = NonZeroU32::new(5).unwrap();
        assert_eq!(next_generation(gen).get(), 6);
    }

    #[test]
    fn dangling_is_valid_index_struct() {
        // DANGLING should be a valid Index (not None when in an Option)
        let opt: Option<Index> = Some(Index::DANGLING);
        assert!(opt.is_some());
    }
}
