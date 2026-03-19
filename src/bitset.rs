// ══════════════════════════════════════════════════════════════════════
// Bitset — Occupancy Tracking via Packed Bits
// ══════════════════════════════════════════════════════════════════════
//
// This is the ENGINE of the entire crate. Every performance advantage
// we have over thunderdome flows from this module.
//
// WHAT THIS DOES
// ──────────────
// Tracks which slots in the arena are occupied (1) or empty (0) using
// a packed array of u64 words. Each u64 tracks 64 slots.
//
//   Slot index:   0  1  2  3  4  5  6  7  ...  63 | 64 65 66 ...
//   Bit position: b0 b1 b2 b3 b4 b5 b6 b7 ... b63 | b0 b1 b2 ...
//   Word index:   ──────── word[0] ────────────────|── word[1] ──
//
// WHY u64 (not u32 or u128)
// ─────────────────────────
// - u64 matches the native register width on x86_64 and aarch64.
//   `trailing_zeros()` compiles to a SINGLE instruction: tzcnt (x86)
//   or rbit+clz (ARM). No function call, no loop.
// - u128 would cover more slots per word, but Rust's u128::trailing_zeros()
//   compiles to two u64 operations on most platforms. No hardware u128 tzcnt.
// - u32 works but processes half the slots per instruction. On 64-bit CPUs,
//   the u64 version is strictly faster.
//
// HOW ITERATION WORKS (The Core Algorithm)
// ────────────────────────────────────────
// To iterate all occupied slots:
//
//   for (word_idx, &word) in self.words.iter().enumerate() {
//       let mut bits = word;
//       while bits != 0 {
//           let bit = bits.trailing_zeros();        // tzcnt — 1 cycle
//           let slot = word_idx * 64 + bit as usize;
//           yield slot;
//           bits &= bits - 1;                       // blsr — 1 cycle
//       }
//   }
//
// This is Kernighan's bit-clearing trick: `bits &= bits - 1` clears the
// lowest set bit. On x86 with BMI1, this compiles to the `blsr` instruction.
// Two instructions per occupied slot. Zero instructions per empty slot.
//
// Compare to thunderdome's iterator which does:
//   - Load Entry<T> (potentially huge if T is large)
//   - Check enum discriminant (branch)
//   - If empty: wasted load + possible branch misprediction
//
// CACHE BEHAVIOR
// ──────────────
// A cache line is 64 bytes on x86/ARM. One cache line holds 8 u64 words
// = 512 slots of occupancy data.
//
// For a 10,000-slot arena:
//   - Occupancy bitset: 10,000 / 8 = 1,250 bytes = ~20 cache lines
//   - thunderdome entries (T=u64): 10,000 × 24 bytes = 240,000 bytes = ~3,750 cache lines
//
// We use 188x fewer cache lines just to find occupied slots.
//
// SIMD POTENTIAL (Future Optimization)
// ────────────────────────────────────
// The bitset is already SIMD-friendly because it's a contiguous array of
// u64s. Future versions could use:
//   - SSE2: _mm_cmpeq_epi64 to find non-zero words (2 words at a time)
//   - AVX2: _mm256_testz_si256 to skip 4 zero words at once
//   - AVX-512: VPCOMPRESSB for direct index extraction
//
// For now, the scalar trailing_zeros() approach is excellent and portable.
// The SIMD path is only worth it for very large arenas (100k+ slots) where
// scanning zero words dominates. Profile before optimizing.
//
// RAYON PARALLELISM
// ─────────────────
// The words array is trivially splittable:
//   - Thread 0 scans words[0..N/4]   → slots 0..N*16
//   - Thread 1 scans words[N/4..N/2] → slots N*16..N*32
//   - etc.
// No synchronization needed for read-only iteration.
// For par_iter_mut, we need to ensure non-overlapping value access,
// which is guaranteed because each thread owns disjoint slot ranges.
//
// ══════════════════════════════════════════════════════════════════════

use alloc::vec::Vec;

/// Packed bitset for tracking slot occupancy.
///
/// Invariants:
/// - `words.len() == (capacity + 63) / 64`  (ceiling division)
/// - Bits beyond `capacity` are always 0 (no phantom "occupied" slots)
/// - `count` always equals the total number of set bits (popcount)
#[derive(Clone)]
pub(crate) struct Bitset {
    words: Vec<u64>,
    /// Cached popcount — avoids scanning the whole bitset for len().
    ///
    /// INVARIANT: count == self.words.iter().map(|w| w.count_ones()).sum()
    /// This must be updated on EVERY set/clear operation.
    count: u32,
}

impl Bitset {
    // ──────────────────────────────────────────────────────────────────
    // CONSTRUCTION
    // ──────────────────────────────────────────────────────────────────

    /// Create a new bitset with all bits cleared.
    ///
    /// `capacity` is the number of SLOTS (not words).
    /// Internally allocates ceil(capacity / 64) words.
    pub(crate) const fn new() -> Self {
        Self {
            words: Vec::new(),
            count: 0,
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let num_words = capacity.div_ceil(64);
        Self {
            words: alloc::vec![0u64; num_words],
            count: 0,
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // BIT MANIPULATION
    // ──────────────────────────────────────────────────────────────────

    /// Set bit at `slot` to 1 (occupied).
    ///
    /// # Panics
    /// Panics if `slot` is out of bounds.
    ///
    /// # Implementation Notes
    /// ```text
    /// word_index = slot / 64     (or slot >> 6)
    /// bit_index  = slot % 64     (or slot & 63)
    /// words[word_index] |= 1u64 << bit_index
    /// ```
    ///
    /// IMPORTANT: Must also increment `self.count`.
    /// IMPORTANT: Only increment count if the bit was NOT already set.
    /// Use: `if !self.is_set(slot) { self.count += 1; }`
    /// Or: check the old bit value before ORing.
    #[inline]
    pub(crate) fn set(&mut self, slot: usize) {
        let word_idx = slot >> 6;
        let bit_idx = slot & 63;
        let mask = 1u64 << bit_idx;
        if self.words[word_idx] & mask == 0 {
            self.count += 1;
        }
        self.words[word_idx] |= mask;
    }

    /// Set bit at `slot` to 1 (occupied), WITHOUT checking if already set.
    ///
    /// # Safety Contract
    /// Caller MUST ensure:
    ///   - `slot / 64 < self.words.len()` (within bounds)
    ///   - The bit at `slot` is currently CLEAR (not already set)
    ///
    /// This is used by insert() where we know the slot is unoccupied
    /// (either from the free list or from a fresh slot).
    #[inline]
    pub(crate) fn set_unchecked(&mut self, slot: usize) {
        let word_idx = slot >> 6;
        let bit_idx = slot & 63;
        // SAFETY: caller guarantees word_idx is in bounds
        unsafe {
            *self.words.get_unchecked_mut(word_idx) |= 1u64 << bit_idx;
        }
        self.count += 1;
    }

    /// Clear bit at `slot` to 0 (empty).
    ///
    /// # Implementation Notes
    /// ```text
    /// words[word_index] &= !(1u64 << bit_index)
    /// ```
    ///
    /// IMPORTANT: Must also decrement `self.count`.
    /// Only decrement if the bit WAS set.
    #[inline]
    pub(crate) fn clear(&mut self, slot: usize) {
        let word_idx = slot >> 6;
        let bit_idx = slot & 63;
        let mask = 1u64 << bit_idx;
        // SAFETY: callers (remove, drain, into_iter) always ensure slot is in bounds
        let word = unsafe { self.words.get_unchecked_mut(word_idx) };
        if *word & mask != 0 {
            self.count -= 1;
        }
        *word &= !mask;
    }

    /// Check if bit at `slot` is set.
    ///
    /// This is on the HOT PATH for get() and get_mut().
    /// Must be as fast as possible — two operations:
    ///   1. Load the u64 word (likely already in L1 cache if nearby slots accessed recently)
    ///   2. Test one bit with AND + compare
    #[inline]
    pub(crate) fn is_set(&self, slot: usize) -> bool {
        let word_idx = slot >> 6;
        if word_idx >= self.words.len() {
            return false;
        }
        let bit_idx = slot & 63;
        // SAFETY: word_idx < self.words.len() checked above
        (unsafe { *self.words.get_unchecked(word_idx) } >> bit_idx) & 1 == 1
    }

    /// Number of set bits (occupied slots).
    ///
    /// O(1) because we cache the count.
    #[cfg(test)]
    #[inline]
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Raw access to the underlying word slice (for iterators).
    #[inline]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    /// Mutable raw access to the underlying word slice (for fold() bulk-clear).
    #[inline]
    pub(crate) fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    // ──────────────────────────────────────────────────────────────────
    // GROWTH
    // ──────────────────────────────────────────────────────────────────

    /// Ensure the bitset can hold at least `slot + 1` slots.
    ///
    /// If the bitset needs to grow, new words are initialized to 0
    /// (all empty). This mirrors Vec::resize behavior.
    ///
    /// # Implementation Notes
    /// Only grow if needed: `if slot / 64 >= self.words.len()`
    /// Use `self.words.resize(new_word_count, 0u64)` — the 0 ensures
    /// new slots start as unoccupied.
    pub(crate) fn grow_to_include(&mut self, slot: usize) {
        let required_words = (slot >> 6) + 1;
        if required_words > self.words.len() {
            self.words.resize(required_words, 0u64);
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // ITERATION — The Star of the Show
    // ──────────────────────────────────────────────────────────────────

    /// Iterate all set bit positions.
    ///
    /// This is the core iteration primitive. Arena::iter() calls this
    /// to find occupied slots, then reads values only at those positions.
    ///
    /// # Performance
    /// - Processes 64 slots per u64 word
    /// - Uses trailing_zeros() (tzcnt) + Kernighan's trick (blsr)
    /// - 2 CPU instructions per occupied slot found
    /// - ZERO cost for empty slots within a word (just one `word != 0` check per 64 slots)
    /// - ZERO cost for fully-empty words (the while loop body never executes)
    ///
    /// # Why This is an Iterator Over Words, Not Individual Bits
    /// We return an iterator that processes one u64 word at a time.
    /// The outer loop (over words) is branch-predictable (sequential access).
    /// The inner loop (extracting bits) uses hardware instructions.
    /// This two-level design matches how the CPU actually works.
    pub(crate) fn iter_set_bits(&self) -> SetBitsIter<'_> {
        // Find the first non-zero word to start in.
        let mut word_idx = 0;
        let mut current_word = 0u64;
        for (i, &w) in self.words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                current_word = w;
                break;
            }
        }
        SetBitsIter {
            words: &self.words,
            word_idx,
            current_word,
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // BULK OPERATIONS (for drain, clear, retain)
    // ──────────────────────────────────────────────────────────────────

    /// Clear all bits and reset count to 0.
    ///
    /// NOTE: This does NOT drop values. The caller (Arena::clear) must
    /// drop all occupied values BEFORE calling this.
    pub(crate) fn clear_all(&mut self) {
        self.words.fill(0);
        self.count = 0;
    }

    // ──────────────────────────────────────────────────────────────────
    // DEBUGGING & TESTING HELPERS
    // ──────────────────────────────────────────────────────────────────

    /// Verify the count invariant (for debug assertions and tests).
    ///
    /// Computes the true popcount by scanning all words and compares
    /// to the cached count. This is O(n) and should only be used in
    /// debug builds or tests.
    #[cfg(test)]
    pub(crate) fn verify_count(&self) -> bool {
        let real: u32 = self.words.iter().map(|w| w.count_ones()).sum();
        real == self.count
    }
}

// ──────────────────────────────────────────────────────────────────────
// SetBitsIter — Iterator over occupied slot indices
// ──────────────────────────────────────────────────────────────────────
//
// DESIGN NOTES ON ITERATOR STATE
// ──────────────────────────────
// We store the "current word being drained" separately from the words
// slice. This avoids re-reading from memory on every next() call —
// the current word lives in a register.
//
// This is a micro-optimization but it matters: the iterator's next()
// is called once per occupied slot, potentially millions of times.
// Keeping the hot state in registers is critical.
//
// ANATOMY OF ONE next() CALL (x86_64 assembly, approximately):
//   tzcnt  rax, rcx          ; trailing_zeros of current_word → bit position
//   blsr   rcx, rcx          ; clear lowest set bit of current_word
//   shl    rdx, 6            ; word_idx * 64
//   add    rax, rdx          ; slot = word_idx * 64 + bit
//   ret                      ; ~4 instructions, ~2-3 cycles
//
// Compare to thunderdome's next() which involves:
//   - Load 24+ bytes of Entry<T>
//   - Compare discriminant (branch)
//   - If empty: loop back (branch misprediction ~15 cycles)
//   - If occupied: construct Index, return
//
// ──────────────────────────────────────────────────────────────────────

pub(crate) struct SetBitsIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    current_word: u64,
}

impl<'a> Iterator for SetBitsIter<'a> {
    type Item = usize; // slot index

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1; // blsr: clear lowest set bit
                return Some(self.word_idx * 64 + bit);
            }
            // Advance to next non-zero word.
            self.word_idx += 1;
            loop {
                if self.word_idx >= self.words.len() {
                    return None;
                }
                // SAFETY: word_idx < words.len(), checked above
                let w = unsafe { *self.words.get_unchecked(self.word_idx) };
                if w != 0 {
                    self.current_word = w;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let lo = self.current_word.count_ones() as usize;
        let hi: usize = lo
            + self
                .words
                .get(self.word_idx + 1..)
                .map_or(0, |rest| rest.iter().map(|w| w.count_ones() as usize).sum());
        (lo, Some(hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ──────────────────────────────────────────────────────────────────
    // BITSET TESTS — These test the foundation of the entire crate.
    // Every arena operation depends on correct bitset behavior.
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn set_and_check() {
        let mut bs = Bitset::with_capacity(64);
        bs.set(42);
        assert!(bs.is_set(42));
        assert!(!bs.is_set(41));
        assert_eq!(bs.count(), 1);
    }

    #[test]
    fn clear_and_check() {
        let mut bs = Bitset::with_capacity(64);
        bs.set(10);
        assert!(bs.is_set(10));
        bs.clear(10);
        assert!(!bs.is_set(10));
        assert_eq!(bs.count(), 0);
    }

    #[test]
    fn count_tracks_correctly() {
        let mut bs = Bitset::with_capacity(128);
        for i in [0usize, 5, 10, 63, 64, 100] {
            bs.set(i);
        }
        assert_eq!(bs.count(), 6);
        bs.clear(5);
        bs.clear(100);
        assert_eq!(bs.count(), 4);
        assert!(bs.verify_count());
    }

    #[test]
    fn iter_set_bits_returns_all_set() {
        let mut bs = Bitset::with_capacity(1001);
        let positions = [0usize, 5, 63, 64, 127, 1000];
        for &p in &positions {
            bs.set(p);
        }
        let found: alloc::vec::Vec<usize> = bs.iter_set_bits().collect();
        assert_eq!(found, positions);
    }

    #[test]
    fn iter_empty_bitset_returns_nothing() {
        let bs = Bitset::new();
        assert!(bs.iter_set_bits().next().is_none());
    }

    #[test]
    fn iter_fully_dense_bitset() {
        let mut bs = Bitset::with_capacity(256);
        for i in 0..256 {
            bs.set(i);
        }
        let found: alloc::vec::Vec<usize> = bs.iter_set_bits().collect();
        let expected: alloc::vec::Vec<usize> = (0..256).collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn grow_preserves_existing_bits() {
        let mut bs = Bitset::with_capacity(64);
        bs.set(5);
        bs.grow_to_include(200);
        assert!(bs.is_set(5));
        assert!(!bs.is_set(200));
    }

    #[test]
    fn word_boundary_bits() {
        let mut bs = Bitset::with_capacity(256);
        for &slot in &[63usize, 64, 127, 128] {
            bs.set(slot);
            assert!(bs.is_set(slot));
            bs.clear(slot);
            assert!(!bs.is_set(slot));
        }
    }

    #[test]
    fn double_set_does_not_double_count() {
        let mut bs = Bitset::with_capacity(64);
        bs.set(10);
        bs.set(10);
        assert_eq!(bs.count(), 1);
    }

    #[test]
    fn clear_unset_bit_does_not_underflow() {
        let mut bs = Bitset::with_capacity(64);
        bs.clear(10); // bit was never set
        assert_eq!(bs.count(), 0);
    }
}
