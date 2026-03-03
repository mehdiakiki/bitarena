// ══════════════════════════════════════════════════════════════════════
// FreeList — Slot Recycling for O(1) Insert
// ══════════════════════════════════════════════════════════════════════
//
// DESIGN CHOICE: Stack (Vec<u32>) vs Linked List vs Bitset Scan
// ──────────────────────────────────────────────────────────────
//
// Option A: Linked list threaded through empty entries (thunderdome's approach)
//   Pros: Zero extra memory — free pointers live inside empty Entry slots
//   Cons: We don't HAVE Entry slots. Our values array is MaybeUninit<T>,
//         and we can't store a u32 pointer inside an arbitrary T-sized slot
//         without type-punning. Also, insert_at requires O(n) traversal
//         to unlink a specific slot from the middle of the list.
//   Verdict: NOT suitable for SoA layout.
//
// Option B: Bitset scan — find first 0 bit in occupancy
//   Pros: Zero extra memory — reuses the occupancy bitset
//   Cons: O(n/64) per insert in the worst case. For an arena with 100k
//         slots and 99k occupied, finding the one free slot requires
//         scanning ~1,562 words. Still fast in practice (~microseconds)
//         but not O(1).
//   Verdict: Good as a FALLBACK, not as the primary mechanism.
//
// Option C: Explicit stack of free slots (Vec<u32>) ← OUR CHOICE
//   Pros:
//     - O(1) push/pop (amortized) for insert/remove
//     - insert_at can be handled without traversal
//     - Simple, debuggable, no type-punning
//     - Vec<u32> is extremely cache-friendly (packed 4-byte integers)
//   Cons:
//     - Extra memory: 4 bytes per free slot
//     - In the worst case (arena with 100k slots, all removed),
//       the free list is 400KB. This is acceptable.
//   Verdict: BEST for our design. The simplicity and O(1) behavior
//            outweigh the memory cost.
//
// LIFO vs FIFO ORDER
// ──────────────────
// We use LIFO (stack) order: the most recently freed slot is reused first.
// Why? Because the most recently freed slot is likely still in L1/L2 cache.
// Reusing it means the subsequent write (to store the new value) hits warm
// cache instead of cold memory.
//
// FIFO would spread inserts across the arena more evenly, which could
// theoretically improve iteration order, but the cache benefit of LIFO
// is more impactful in practice.
//
// INTERACTION WITH BITSET
// ───────────────────────
// The free list and bitset must stay in sync:
//   - remove(slot): push slot onto free list AND clear occupancy bit
//   - insert(): pop slot from free list AND set occupancy bit
//
// If they ever disagree, we have UB (reading uninitialized memory from
// a slot the bitset says is occupied, or double-freeing a slot that
// appears twice in the free list).
//
// INVARIANT: Every slot index in the free list has its occupancy bit CLEAR.
//            No slot index appears more than once in the free list.
//
// ══════════════════════════════════════════════════════════════════════

use alloc::vec::Vec;

/// Stack-based free slot tracker.
///
/// Provides O(1) amortized push/pop for slot recycling.
///
/// # Invariants
/// - All slot indices in `slots` have their occupancy bit cleared in the bitset
/// - No duplicate indices in `slots`
/// - All indices are < arena capacity
#[derive(Clone)]
pub(crate) struct FreeList {
    /// Stack of free slot indices. Last element = top of stack = next to reuse.
    slots: Vec<u32>,
}

impl FreeList {
    pub(crate) const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Push a freed slot onto the stack.
    ///
    /// Called by Arena::remove().
    ///
    /// # Contract
    /// - Caller MUST have already cleared the occupancy bit
    /// - Caller MUST have already read/dropped the value at this slot
    /// - Slot MUST NOT already be in the free list (no double-free)
    #[inline]
    pub(crate) fn push(&mut self, slot: u32) {
        debug_assert!(!self.slots.contains(&slot), "double free of slot {}", slot);
        self.slots.push(slot);
    }

    /// Pop the most recently freed slot.
    ///
    /// Called by Arena::insert() to reuse a slot.
    /// Returns None if no free slots (arena must grow instead).
    ///
    /// # LIFO Cache Benefit
    /// The returned slot was the most recently freed, meaning it's likely
    /// still in L1/L2 cache. The caller will write a new value to this slot,
    /// which will be a cache hit instead of a cold write.
    #[inline]
    pub(crate) fn pop(&mut self) -> Option<u32> {
        self.slots.pop()
    }

    /// Number of free slots available for reuse.
    #[cfg(test)]
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Clear the free list entirely.
    ///
    /// Called by Arena::clear() after all values have been dropped.
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
    }

    /// Remove a specific slot from the free list (if present).
    ///
    /// Called by Arena::insert_at() when the target slot happens to be
    /// in the free list.
    ///
    /// # Performance
    /// This is O(n) in the free list length. However:
    /// - insert_at is a rare operation (most users use insert)
    /// - swap_remove maintains O(1) for subsequent push/pop
    /// - thunderdome also has an O(n) path for this (free list traversal)
    ///
    /// # Implementation
    /// ```text
    /// if let Some(pos) = self.slots.iter().position(|&s| s == slot) {
    ///     self.slots.swap_remove(pos);  // O(1) removal, order doesn't matter
    /// }
    /// ```
    pub(crate) fn remove_slot(&mut self, slot: u32) {
        if let Some(pos) = self.slots.iter().position(|&s| s == slot) {
            self.slots.swap_remove(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_lifo() {
        let mut fl = FreeList::new();
        fl.push(1);
        fl.push(2);
        fl.push(3);
        assert_eq!(fl.pop(), Some(3));
        assert_eq!(fl.pop(), Some(2));
        assert_eq!(fl.pop(), Some(1));
        assert_eq!(fl.pop(), None);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut fl = FreeList::new();
        assert_eq!(fl.pop(), None);
    }

    #[test]
    fn remove_slot_from_middle() {
        let mut fl = FreeList::new();
        fl.push(1);
        fl.push(2);
        fl.push(3);
        fl.remove_slot(2);
        assert_eq!(fl.len(), 2);
        let mut remaining = [fl.pop().unwrap(), fl.pop().unwrap()];
        remaining.sort();
        assert_eq!(remaining, [1, 3]);
    }

    #[test]
    fn remove_nonexistent_slot_is_noop() {
        let mut fl = FreeList::new();
        fl.push(1);
        fl.push(2);
        fl.remove_slot(99);
        assert_eq!(fl.len(), 2);
    }
}
