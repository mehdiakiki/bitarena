// ══════════════════════════════════════════════════════════════════════
// Arena<T> — The Main Public Interface
// ══════════════════════════════════════════════════════════════════════
//
// This is the struct users interact with. Its API matches thunderdome's
// as closely as possible so migration is trivial.
//
// MEMORY LAYOUT (The SoA Advantage)
// ──────────────────────────────────
// Three parallel arrays + a bitset + a free list:
//
//   occupancy:   [u64, u64, u64, ...]     ← 1 bit per slot, packed
//   generations: [u32, u32, u32, ...]     ← 1 generation per slot
//   values:      [MaybeUninit<T>, ...]    ← raw storage
//   free_list:   [u32, u32, ...]          ← stack of free slot indices
//
// All arrays have the same logical length: `capacity` slots.
// `len` tracks the number of occupied slots.
//
// Compare to thunderdome's single array:
//
//   storage: [Entry<T>, Entry<T>, ...]    ← enum per slot
//
// The SoA layout means:
//   - Checking occupancy touches ONLY the bitset (tiny, fits in L1 cache)
//   - Checking generation touches ONLY the generations array (small)
//   - Value data is ONLY loaded when you actually need the value
//   - Empty slots don't pollute the cache with value-sized garbage
//
// SAFETY INVARIANTS (Central Contract)
// ─────────────────────────────────────
//
//   I1: values[i] is initialized  ⟺  occupancy.is_set(i)
//   I2: generations[i] is always valid (initialized on slot creation)
//   I3: free_list contains ONLY slots where occupancy.is_set(i) == false
//   I4: No slot appears more than once in free_list
//   I5: len == occupancy.count() (always in sync)
//   I6: All arrays have the same logical capacity
//
// Every unsafe operation in this file must cite which invariant(s)
// guarantee its safety.
//
// GROWTH STRATEGY
// ───────────────
// When insert() finds no free slots AND capacity is full:
//   1. Double the capacity (standard amortized O(1) growth)
//   2. Grow all three arrays (occupancy bitset, generations, values)
//   3. New occupancy bits are 0 (empty), new generations are 0 (unused)
//   4. New value slots are uninitialized (MaybeUninit)
//   5. Do NOT add new slots to free list — just bump the "next fresh slot" counter
//
// Why not add new slots to free list? Because we can track the "next slot
// that was never used" with a simple counter (`next_fresh`). This avoids
// filling the free list with thousands of sequential indices on growth.
// The free list is only for RECYCLED slots.
//
// ══════════════════════════════════════════════════════════════════════

use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::num::NonZeroU32;
use core::ops;

use crate::bitset::Bitset;
use crate::free_list::FreeList;
use crate::index::Index;
use crate::iter::{Drain, IntoIter, Iter, IterMut};

/// A generational arena with bitset-accelerated, cache-friendly operations.
///
/// `Arena<T>` stores values of type `T` and returns [`Index`] handles
/// for later retrieval. Indices are generational: removing a value and
/// inserting a new one at the same slot produces a new generation,
/// invalidating old indices.
///
/// # Performance Characteristics
///
/// | Operation       | Time    | Cache Behavior                              |
/// |-----------------|---------|---------------------------------------------|
/// | `insert`        | O(1)*   | Writes to values + bitset + generations     |
/// | `remove`        | O(1)    | Reads value, clears bit, pushes free list   |
/// | `get` / `get_mut` | O(1) | Checks bit + generation, then loads value   |
/// | `contains`      | O(1)    | Checks bit + generation only (no value load)|
/// | `iter`          | O(n/64) | Scans bitset, loads only occupied values     |
/// | `len`           | O(1)    | Cached count                                |
///
/// *Amortized due to potential Vec reallocation on growth.
///
/// # Comparison to thunderdome
///
/// For sparse arenas (many removed elements), iteration is dramatically
/// faster because empty slots are skipped via bitset scanning (hardware
/// `tzcnt` instruction) rather than per-element enum matching.
///
/// # Example
///
/// ```
/// use bitarena::Arena;
///
/// let mut arena = Arena::new();
///
/// let foo = arena.insert("foo");
/// let bar = arena.insert("bar");
///
/// assert_eq!(arena.get(foo), Some(&"foo"));
/// assert_eq!(arena.get(bar), Some(&"bar"));
///
/// arena.remove(foo);
/// assert_eq!(arena.get(foo), None); // Stale index
///
/// let baz = arena.insert("baz"); // Reuses foo's slot
/// assert_eq!(arena.get(baz), Some(&"baz"));
/// ```
pub struct Arena<T> {
    // ──────────────────────────────────────────────────────────────────
    // FIELD ORDER NOTE
    // ──────────────────────────────────────────────────────────────────
    // Fields are ordered by access frequency in hot paths:
    //   1. occupancy    — checked on EVERY get/insert/remove/iter
    //   2. values       — raw storage, only valid when bit is set
    //   3. generations  — generation counter per slot
    //   4. free_list    — only on insert/remove
    //   5. next_fresh   — only on insert when free list empty
    //   6. len          — rarely accessed in hot loops
    //
    // SoA layout: during iteration we ONLY touch occupancy + values.
    // The generations array is never accessed in the iteration hot path,
    // halving cache footprint compared to an AoS layout.
    // ──────────────────────────────────────────────────────────────────
    /// Bitset tracking which slots are occupied.
    /// Bit i is set ⟺ values[i] is initialized.
    pub(crate) occupancy: Bitset,

    /// Raw value storage. values[i] is initialized ⟺ occupancy.is_set(i).
    pub(crate) values: Vec<MaybeUninit<T>>,

    /// One generation counter per slot. Always valid (initialized on first use).
    pub(crate) generations: Vec<u32>,

    /// Stack of previously-used slots available for reuse.
    /// LIFO order: most recently freed slot is popped first (cache-warm).
    pub(crate) free_list: FreeList,

    /// The next slot index that has NEVER been used.
    /// When free_list is empty, we use this slot and increment.
    /// When next_fresh == capacity, we need to grow all arrays.
    pub(crate) next_fresh: u32,

    /// Number of occupied slots. Always equals occupancy.count().
    /// Cached here for O(1) len() without scanning the bitset.
    pub(crate) len: u32,
}

// ══════════════════════════════════════════════════════════════════════
// CONSTRUCTION
// ══════════════════════════════════════════════════════════════════════

impl<T> Arena<T> {
    /// Create an empty arena with no allocated memory.
    ///
    /// No allocation occurs until the first `insert`.
    pub const fn new() -> Self {
        Self {
            occupancy: Bitset::new(),
            values: Vec::new(),
            generations: Vec::new(),
            free_list: FreeList::new(),
            next_fresh: 0,
            len: 0,
        }
    }

    /// Create an empty arena pre-allocated for `capacity` elements.
    ///
    /// This avoids reallocation during the first `capacity` inserts.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut values: Vec<MaybeUninit<T>> = Vec::with_capacity(capacity);
        unsafe { values.set_len(capacity) };
        let generations = alloc::vec![0u32; capacity];
        Self {
            occupancy: Bitset::with_capacity(capacity),
            values,
            generations,
            free_list: FreeList::new(),
            next_fresh: 0,
            len: 0,
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CORE OPERATIONS — insert, remove, get
    // ══════════════════════════════════════════════════════════════════

    /// Insert a value, returning an [`Index`] handle.
    ///
    /// If a previously removed slot is available, it will be reused
    /// (with an incremented generation). Otherwise, a new slot is allocated.
    ///
    /// # Panics
    /// Panics if the arena contains `u32::MAX` elements.
    ///
    /// # Implementation Strategy
    /// ```text
    /// 1. Try to pop a slot from free_list (O(1), cache-warm)
    ///    → If success: reuse slot, increment generation
    ///
    /// 2. Else if next_fresh < capacity:
    ///    → Use next_fresh, set generation to 1, increment next_fresh
    ///
    /// 3. Else: grow all arrays (double capacity), then use step 2
    ///
    /// In ALL cases:
    ///    a. Write value into values[slot]        (initialize MaybeUninit)
    ///    b. Set occupancy bit for slot            (mark as occupied)
    ///    c. Set/update generations[slot]          (set generation)
    ///    d. Increment len
    ///    e. Return Index { slot, generation }
    /// ```
    ///
    /// # Safety Notes
    /// - Step (a) MUST happen BEFORE step (b). If we set the bit first
    ///   and then crash/panic before writing the value, the invariant
    ///   "bit set ⟹ value initialized" is violated.
    ///   Actually, since Rust panics unwind and drop runs... we need to
    ///   think carefully about panic safety. See PANIC SAFETY section below.
    #[inline]
    pub fn insert(&mut self, value: T) -> Index {
        let (slot, gen) = if let Some(recycled) = self.free_list.pop() {
            let s = recycled as usize;
            let old_raw = unsafe { *self.generations.get_unchecked(s) };
            let next = old_raw.wrapping_add(1);
            let new_raw = next + (next == 0) as u32;
            unsafe { *self.generations.get_unchecked_mut(s) = new_raw };
            unsafe {
                self.values
                    .as_mut_ptr()
                    .add(s)
                    .write(MaybeUninit::new(value))
            };
            (s, new_raw)
        } else {
            let slot = self.next_fresh as usize;
            if slot >= self.values.len() {
                self.grow();
            }
            unsafe { *self.generations.get_unchecked_mut(slot) = 1 };
            unsafe {
                self.values
                    .as_mut_ptr()
                    .add(slot)
                    .write(MaybeUninit::new(value))
            };
            self.next_fresh += 1;
            (slot, 1u32)
        };

        self.occupancy.set_unchecked(slot);
        self.len += 1;

        Index {
            slot: slot as u32,
            generation: unsafe { NonZeroU32::new_unchecked(gen) },
        }
    }

    /// Grow all arrays (occupancy bitset, generations, values).
    /// Called when `next_fresh >= self.values.len()`.
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        // Base the new capacity on next_fresh (the actually-used high watermark),
        // not values.len() (allocated capacity). This prevents double-grow when
        // next_fresh has already advanced past the old doubling threshold.
        // Example: with_capacity(4) then insert 5 items:
        //   old_cap = 4, next_fresh = 5 → new_cap = max(5*2, 4*2) = 10
        //   (without this fix: new_cap = max(4*2, 4) = 8, triggers grow again at 9)
        let old_cap = self.values.len();
        let new_cap = (self.next_fresh as usize * 2).max(old_cap * 2).max(4);
        self.occupancy.grow_to_include(new_cap - 1);
        self.values.reserve(new_cap - old_cap);
        unsafe { self.values.set_len(new_cap) };
        self.generations.resize(new_cap, 0);
    }

    /// Grow all arrays to ensure `min_cap` slots are available.
    ///
    /// Used by serde deserialization to pre-allocate before writing entries.
    /// Does NOT change `next_fresh` — the caller is responsible for updating
    /// it after writing entries.
    #[cold]
    #[cfg(feature = "serde")]
    pub(crate) fn grow_to(&mut self, min_cap: usize) {
        let old_cap = self.values.len();
        if old_cap >= min_cap {
            return;
        }
        let new_cap = min_cap.max(old_cap * 2).max(4);
        self.occupancy.grow_to_include(new_cap - 1);
        self.values.reserve(new_cap - old_cap);
        unsafe { self.values.set_len(new_cap) };
        self.generations.resize(new_cap, 0);
    }

    /// Remove the element at `index`, returning it if valid.
    ///
    /// Returns `None` if the index is stale (wrong generation) or
    /// out of bounds.
    ///
    /// The slot becomes available for reuse by future `insert` calls.
    ///
    /// # Implementation Strategy
    /// ```text
    /// 1. Bounds check: index.slot < next_fresh (not capacity! slots beyond
    ///    next_fresh have never been used)
    /// 2. Occupancy check: occupancy.is_set(slot)
    /// 3. Generation check: generations[slot] == index.generation
    /// 4. If all pass:
    ///    a. Read value out of values[slot] via ptr::read  (UNSAFE)
    ///    b. Clear occupancy bit
    ///    c. Push slot onto free_list
    ///    d. Decrement len
    ///    e. Return Some(value)
    /// 5. If any check fails: return None
    /// ```
    ///
    /// # Safety
    /// Step (a) uses unsafe ptr::read on MaybeUninit.
    /// Justified by invariants I1 (bit set ⟹ initialized) + I3 (generation matches).
    ///
    /// After ptr::read, the slot's value is logically moved out.
    /// We MUST NOT read it again or drop it. Clearing the bit (step b)
    /// ensures no future code path will try to read it.
    #[inline]
    pub fn remove(&mut self, index: Index) -> Option<T> {
        let slot = index.slot as usize;
        if slot >= self.next_fresh as usize {
            return None;
        }
        if !self.occupancy.is_set(slot) {
            return None;
        }
        if unsafe { *self.generations.get_unchecked(slot) } != index.generation.get() {
            return None;
        }
        let value = unsafe { self.values.as_ptr().add(slot).cast::<T>().read() };
        self.occupancy.clear(slot);
        self.free_list.push(slot as u32);
        self.len -= 1;
        Some(value)
    }

    /// Get a reference to the value at `index`.
    ///
    /// Returns `None` if the index is stale or out of bounds.
    ///
    /// # Performance
    /// This is the most latency-sensitive operation. Hot path:
    ///   1. Bounds check (1 comparison)
    ///   2. Bit check (1 memory load from bitset, 1 AND, 1 comparison)
    ///   3. Generation check (1 memory load from generations, 1 comparison)
    ///   4. Return reference to values\[slot\]
    ///
    /// Total: ~3 memory loads. For thunderdome: 1 large load + discriminant check.
    /// Our loads are smaller and more cache-friendly (bitset word may already
    /// be in L1 from a recent check on a nearby slot).
    #[must_use]
    #[inline]
    pub fn get(&self, index: Index) -> Option<&T> {
        let slot = index.slot as usize;
        if slot >= self.next_fresh as usize {
            return None;
        }
        if !self.occupancy.is_set(slot) {
            return None;
        }
        if unsafe { *self.generations.get_unchecked(slot) } != index.generation.get() {
            return None;
        }
        Some(unsafe { self.values.get_unchecked(slot).assume_init_ref() })
    }

    /// Get a mutable reference to the value at `index`.
    ///
    /// Same checks as `get`, but returns `&mut T`.
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self, index: Index) -> Option<&mut T> {
        let slot = index.slot as usize;
        if slot >= self.next_fresh as usize {
            return None;
        }
        if !self.occupancy.is_set(slot) {
            return None;
        }
        if unsafe { *self.generations.get_unchecked(slot) } != index.generation.get() {
            return None;
        }
        Some(unsafe { self.values.get_unchecked_mut(slot).assume_init_mut() })
    }

    /// Get mutable references to two different slots simultaneously.
    ///
    /// # Panics
    /// Panics if `a` and `b` refer to the same slot (even with different generations).
    ///
    /// # Why This Exists
    /// A common pattern in game engines:
    /// ```text
    /// let (player, enemy) = arena.get2_mut(player_idx, enemy_idx);
    /// player.hp -= enemy.damage;
    /// enemy.hp -= player.damage;
    /// ```
    /// Without get2_mut, you'd need unsafe code or index juggling.
    #[must_use]
    #[track_caller]
    pub fn get2_mut(&mut self, a: Index, b: Index) -> (Option<&mut T>, Option<&mut T>) {
        assert_ne!(a.slot, b.slot, "get2_mut called with the same slot");
        let len = self.values.len();
        let ptr = self.values.as_mut_ptr();
        let gens = self.generations.as_ptr();

        let get_one =
            |slot: usize, gen: u32, next_fresh: u32, occupancy: &Bitset| -> Option<*mut T> {
                if slot >= next_fresh as usize {
                    return None;
                }
                if !occupancy.is_set(slot) {
                    return None;
                }
                if slot >= len {
                    return None;
                }
                if unsafe { *gens.add(slot) } != gen {
                    return None;
                }
                Some(unsafe { (*ptr.add(slot)).as_mut_ptr() })
            };

        let ra = get_one(
            a.slot as usize,
            a.generation.get(),
            self.next_fresh,
            &self.occupancy,
        );
        let rb = get_one(
            b.slot as usize,
            b.generation.get(),
            self.next_fresh,
            &self.occupancy,
        );

        (
            ra.map(|p| unsafe { &mut *p }),
            rb.map(|p| unsafe { &mut *p }),
        )
    }

    /// Check if `index` is valid (occupied with matching generation).
    ///
    /// Unlike `get`, this does NOT load the value — it only checks
    /// the bitset and generation. Even faster than `get` for large T.
    #[must_use]
    #[inline]
    pub fn contains(&self, index: Index) -> bool {
        let slot = index.slot as usize;
        slot < self.next_fresh as usize
            && self.occupancy.is_set(slot)
            && unsafe { *self.generations.get_unchecked(slot) } == index.generation.get()
    }

    /// Check if a slot is occupied, regardless of generation.
    ///
    /// Returns the full `Index` (slot + current generation) if occupied.
    /// This is useful for iterating by slot number or for debugging.
    #[must_use]
    pub fn contains_slot(&self, slot: u32) -> Option<Index> {
        let s = slot as usize;
        if s >= self.next_fresh as usize || !self.occupancy.is_set(s) {
            return None;
        }
        let gen = unsafe { *self.generations.get_unchecked(s) };
        Some(Index {
            slot,
            generation: unsafe { NonZeroU32::new_unchecked(gen) },
        })
    }

    // ══════════════════════════════════════════════════════════════════
    // SLOT-BASED ACCESS (Thunderdome API compatibility)
    // ══════════════════════════════════════════════════════════════════
    //
    // These methods access by raw slot number, ignoring generation.
    // They exist for thunderdome API parity and for use cases where
    // the caller tracks validity externally.
    // ══════════════════════════════════════════════════════════════════

    /// Get a reference to the value at `slot`, ignoring generation.
    ///
    /// Returns `(current_index, &value)` if the slot is occupied, or `None` if empty.
    /// Prefer [`get`](Arena::get) with a full [`Index`] for normal usage — this
    /// bypasses the generation check and is intended for low-level or migration code.
    #[must_use]
    pub fn get_by_slot(&self, slot: u32) -> Option<(Index, &T)> {
        let s = slot as usize;
        if s >= self.next_fresh as usize || !self.occupancy.is_set(s) {
            return None;
        }
        let gen = unsafe { NonZeroU32::new_unchecked(*self.generations.get_unchecked(s)) };
        let value = unsafe { self.values.get_unchecked(s).assume_init_ref() };
        Some((
            Index {
                slot,
                generation: gen,
            },
            value,
        ))
    }

    /// Get a mutable reference to the value at `slot`, ignoring generation.
    ///
    /// Returns `(current_index, &mut value)` if the slot is occupied, or `None` if empty.
    /// See [`get_by_slot`](Arena::get_by_slot) for caveats.
    #[must_use]
    pub fn get_by_slot_mut(&mut self, slot: u32) -> Option<(Index, &mut T)> {
        let s = slot as usize;
        if s >= self.next_fresh as usize || !self.occupancy.is_set(s) {
            return None;
        }
        let gen = unsafe { NonZeroU32::new_unchecked(*self.generations.get_unchecked(s)) };
        let value = unsafe { self.values.get_unchecked_mut(s).assume_init_mut() };
        Some((
            Index {
                slot,
                generation: gen,
            },
            value,
        ))
    }

    /// Remove the element at `slot`, ignoring generation.
    ///
    /// Returns `(index_that_was_removed, value)` if the slot was occupied.
    /// The generation of the returned `Index` reflects the generation that
    /// was active at the time of removal.
    /// Prefer [`remove`](Arena::remove) with a full [`Index`] for normal usage.
    #[must_use]
    pub fn remove_by_slot(&mut self, slot: u32) -> Option<(Index, T)> {
        let s = slot as usize;
        if s >= self.next_fresh as usize || !self.occupancy.is_set(s) {
            return None;
        }
        let gen = unsafe { NonZeroU32::new_unchecked(*self.generations.get_unchecked(s)) };
        let value = unsafe { self.values.as_ptr().add(s).cast::<T>().read() };
        self.occupancy.clear(s);
        self.free_list.push(slot);
        self.len -= 1;
        Some((
            Index {
                slot,
                generation: gen,
            },
            value,
        ))
    }

    // ══════════════════════════════════════════════════════════════════
    // INSERT_AT — Targeted insertion (thunderdome API compatibility)
    // ══════════════════════════════════════════════════════════════════

    /// Insert at a specific index (slot + generation).
    ///
    /// If the slot is already occupied, replaces the value and returns the old one.
    /// If the slot is empty, inserts and returns None.
    /// If the slot is beyond current capacity, grows the arena.
    ///
    /// # Caveats
    /// This can "resurrect" old indices. See thunderdome docs for details.
    pub fn insert_at(&mut self, index: Index, value: T) -> Option<T> {
        let slot = index.slot as usize;
        while slot >= self.values.len() {
            let old_cap = self.values.len();
            let new_cap = (old_cap * 2).max(slot + 1);
            self.occupancy.grow_to_include(new_cap - 1);
            self.values.reserve(new_cap - old_cap);
            unsafe { self.values.set_len(new_cap) };
            self.generations.resize(new_cap, 0);
        }
        if slot >= self.next_fresh as usize {
            self.next_fresh = (slot + 1) as u32;
        }
        unsafe { *self.generations.get_unchecked_mut(slot) = index.generation.get() };

        if self.occupancy.is_set(slot) {
            let old = unsafe { self.values.as_ptr().add(slot).cast::<T>().read() };
            unsafe {
                self.values
                    .as_mut_ptr()
                    .add(slot)
                    .write(MaybeUninit::new(value))
            };
            Some(old)
        } else {
            self.free_list.remove_slot(slot as u32);
            unsafe {
                self.values
                    .as_mut_ptr()
                    .add(slot)
                    .write(MaybeUninit::new(value))
            };
            self.occupancy.set(slot);
            self.len += 1;
            None
        }
    }

    /// Insert a value at a specific `slot`, automatically choosing the next generation.
    ///
    /// The generation is incremented from the slot's last known generation (or starts
    /// at 1 for a never-used slot), so all previously issued indices for that slot
    /// become invalid.
    ///
    /// Returns `(new_index, Option<old_value>)`. If the slot was already occupied the
    /// old value is returned; otherwise `None`.
    ///
    /// See also [`insert_at`](Arena::insert_at) for inserting at a fully-specified
    /// `(slot, generation)` pair.
    #[must_use]
    #[track_caller]
    pub fn insert_at_slot(&mut self, slot: u32, value: T) -> (Index, Option<T>) {
        let s = slot as usize;
        let gen = if s < self.next_fresh as usize {
            let old_gen = unsafe { *self.generations.get_unchecked(s) };
            if old_gen != 0 {
                let old = unsafe { NonZeroU32::new_unchecked(old_gen) };
                let next = old.get().wrapping_add(1);
                unsafe { NonZeroU32::new_unchecked(next + (next == 0) as u32) }
            } else {
                unsafe { NonZeroU32::new_unchecked(1) }
            }
        } else {
            unsafe { NonZeroU32::new_unchecked(1) }
        };
        let idx = Index {
            slot,
            generation: gen,
        };
        let old = self.insert_at(idx, value);
        (idx, old)
    }

    /// Invalidate an index, returning a new index for the same value.
    ///
    /// The old index becomes invalid; the new index has an incremented generation.
    /// The value is NOT moved — only the generation counter changes.
    #[must_use]
    pub fn invalidate(&mut self, index: Index) -> Option<Index> {
        if !self.contains(index) {
            return None;
        }
        let slot = index.slot as usize;
        let old_raw = unsafe { *self.generations.get_unchecked(slot) };
        let next = old_raw.wrapping_add(1);
        let new_raw = next + (next == 0) as u32;
        unsafe { *self.generations.get_unchecked_mut(slot) = new_raw };
        Some(Index {
            slot: index.slot,
            generation: unsafe { NonZeroU32::new_unchecked(new_raw) },
        })
    }

    // ══════════════════════════════════════════════════════════════════
    // CAPACITY & LENGTH
    // ══════════════════════════════════════════════════════════════════

    /// Number of occupied elements.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the arena is empty.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total capacity: the number of slots currently allocated (occupied + free).
    ///
    /// Note: this reflects the number of allocated slots, not the underlying
    /// `Vec` capacity. Newly allocated slots beyond `next_fresh` are available
    /// via the fresh-slot counter before the free list is consulted.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.values.len()
    }

    /// Reserve space for at least `additional` more elements without reallocation.
    ///
    /// Accounts for existing free slots; if enough are already available this
    /// is a no-op.
    pub fn reserve(&mut self, additional: usize) {
        let currently_free = (self.next_fresh as usize).saturating_sub(self.len as usize);
        let to_reserve = additional.saturating_sub(currently_free);
        if to_reserve == 0 {
            return;
        }
        let old_len = self.values.len();
        let new_len = old_len + to_reserve;
        self.occupancy.grow_to_include(new_len - 1);
        self.values.reserve(to_reserve);
        unsafe { self.values.set_len(new_len) };
        self.generations.resize(new_len, 0);
    }

    // ══════════════════════════════════════════════════════════════════
    // ITERATION
    // ══════════════════════════════════════════════════════════════════
    //
    // This is WHERE WE WIN. The bitset-accelerated iteration is the
    // primary reason this crate exists.
    //
    // ══════════════════════════════════════════════════════════════════

    /// Iterate over all occupied (Index, &T) pairs.
    ///
    /// # Performance
    /// Iteration scans the occupancy bitset using hardware bit-scanning
    /// instructions (`tzcnt`/`blsr` on x86, `rbit`/`clz` on ARM).
    /// Empty slots are skipped in bulk — 64 empty slots cost one `u64 == 0`
    /// check. Values are only loaded for occupied slots.
    ///
    /// For a sparse arena (many removed elements), this is dramatically
    /// faster than crates using enum-based entry scanning.
    pub fn iter(&self) -> Iter<'_, T> {
        let words = self.occupancy.words();
        let mut word_idx = 0;
        let mut current_word = 0u64;
        for (i, &w) in words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                current_word = w;
                break;
            }
        }
        Iter {
            occupancy_words: words,
            values: &self.values[..],
            generations: &self.generations[..],
            word_idx,
            current_word: if current_word == u64::MAX {
                0
            } else {
                current_word
            },
            remaining: self.len,
            base_slot: word_idx << 6,
            dense_end: if current_word == u64::MAX {
                (word_idx << 6) + 64
            } else {
                0
            },
        }
    }

    /// Parallel iterate over all occupied `(Index, &T)` pairs.
    ///
    /// Work is split by occupancy word ranges (`u64` words), so each thread
    /// scans disjoint slot ranges with no synchronization.
    #[cfg(feature = "rayon")]
    pub fn par_iter(&self) -> impl rayon::iter::ParallelIterator<Item = (Index, &T)> + '_
    where
        T: Sync,
    {
        crate::iter::par_iter(
            self.occupancy.words(),
            &self.values[..],
            &self.generations[..],
        )
    }

    /// Mutably iterate over all occupied `(Index, &mut T)` pairs.
    ///
    /// Allows in-place mutation of every live value while also yielding its handle.
    /// Use [`values_mut`](Arena::values_mut) if you only need `&mut T` without
    /// the index.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let remaining = self.len;
        let values_ptr = self.values.as_mut_ptr();
        let generations_ptr = self.generations.as_ptr();
        let occupancy_words: &[u64] = unsafe {
            core::slice::from_raw_parts(
                self.occupancy.words().as_ptr(),
                self.occupancy.words().len(),
            )
        };
        let mut word_idx = 0;
        let mut current_word = 0u64;
        for (i, &w) in occupancy_words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                current_word = w;
                break;
            }
        }
        IterMut {
            occupancy_words,
            values_ptr,
            generations_ptr,
            word_idx,
            current_word: if current_word == u64::MAX {
                0
            } else {
                current_word
            },
            remaining,
            base_slot: word_idx << 6,
            dense_end: if current_word == u64::MAX {
                (word_idx << 6) + 64
            } else {
                0
            },
            _marker: core::marker::PhantomData,
        }
    }

    /// Iterate over references to all occupied values, without yielding indices.
    ///
    /// This is the **highest-performance** iteration pattern when you only need
    /// values and not their keys. Because `Index` construction is skipped entirely,
    /// LLVM can auto-vectorize the inner loop with AVX2 VPADDQ instructions.
    ///
    /// # Example
    /// ```
    /// # use bitarena::Arena;
    /// let mut arena = Arena::new();
    /// arena.insert(1u64);
    /// arena.insert(2u64);
    /// let sum: u64 = arena.values().sum();
    /// assert_eq!(sum, 3);
    /// ```
    pub fn values(&self) -> crate::iter::Values<'_, T> {
        crate::iter::Values::new(self.occupancy.words(), &self.values[..], self.len)
    }

    /// Parallel iterate over references to all occupied values.
    ///
    /// This is the highest-throughput parallel iteration path when keys
    /// are not needed.
    #[cfg(feature = "rayon")]
    pub fn par_values(&self) -> impl rayon::iter::ParallelIterator<Item = &T> + '_
    where
        T: Sync,
    {
        crate::iter::par_values(self.occupancy.words(), &self.values[..])
    }

    /// Iterate over mutable references to all occupied values, without yielding indices.
    ///
    /// Slightly cheaper than [`iter_mut`](Arena::iter_mut) because no
    /// [`Index`] is constructed per element.
    pub fn values_mut(&mut self) -> crate::iter::ValuesMut<'_, T> {
        let values_len = self.values.len();
        let values_ptr = self.values.as_mut_ptr();
        let remaining = self.len;
        let occupancy_words: &[u64] = unsafe {
            core::slice::from_raw_parts(
                self.occupancy.words().as_ptr(),
                self.occupancy.words().len(),
            )
        };
        crate::iter::ValuesMut::new(occupancy_words, values_ptr, values_len, remaining)
    }

    /// Iterate over the [`Index`] of every occupied slot, without loading values.
    ///
    /// Useful for collecting all live handles without touching value data.
    pub fn keys(&self) -> crate::iter::Keys<'_> {
        crate::iter::Keys::new(self.occupancy.words(), &self.generations[..], self.len)
    }

    /// Parallel iterate over the [`Index`] of every occupied slot.
    #[cfg(feature = "rayon")]
    pub fn par_keys(&self) -> impl rayon::iter::ParallelIterator<Item = Index> + '_ {
        crate::iter::par_keys(self.occupancy.words(), &self.generations[..])
    }

    /// Remove and yield every element from the arena.
    ///
    /// The arena is left empty after the iterator is consumed (or dropped).
    /// Unlike [`clear`](Arena::clear), `drain` lets you inspect each
    /// `(Index, T)` pair as it is removed.
    pub fn drain(&mut self) -> Drain<'_, T> {
        let words = self.occupancy.words();
        let mut word_idx = 0;
        let mut current_word = 0u64;
        for (i, &w) in words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                current_word = w;
                break;
            }
        }
        Drain {
            arena: self as *mut Arena<T>,
            word_idx,
            current_word,
            _marker: core::marker::PhantomData,
        }
    }

    /// Remove all elements, dropping them in place.
    ///
    /// Resets the arena to the same state as [`Arena::new`]: `len` becomes 0,
    /// `next_fresh` is reset to 0, and the free list is cleared.  Allocated
    /// memory (capacity) is retained.
    pub fn clear(&mut self) {
        if core::mem::needs_drop::<T>() {
            for slot in self.occupancy.iter_set_bits() {
                unsafe { self.values.get_unchecked_mut(slot).assume_init_drop() };
            }
        }
        self.occupancy.clear_all();
        self.free_list.clear();
        self.next_fresh = 0;
        self.len = 0;
    }

    /// Retain only the elements for which `f(index, &mut value)` returns `true`.
    ///
    /// Elements for which `f` returns `false` are removed and dropped.
    /// Iteration order is ascending slot order.
    ///
    /// # Note on key argument
    /// The [`Index`] passed to `f` is passed **by value** (not reference) because
    /// `Index` is `Copy`. This matches the ergonomics of `HashMap::retain` where
    /// the key is a plain `&K`, but avoids a redundant dereference for a
    /// two-word copy type.
    pub fn retain<F: FnMut(Index, &mut T) -> bool>(&mut self, mut f: F) {
        // Walk all occupancy words directly to avoid the intermediate Vec
        // allocation that .collect() would require. We iterate by word index
        // rather than using iter_set_bits() so that removing a slot (clearing
        // its bit) doesn't confuse the outer loop — we've already loaded the
        // word into `word` before checking individual bits.
        let num_words = self.occupancy.words().len();
        for wi in 0..num_words {
            // Load the word once; we'll clear bits from our local copy as
            // we process them, leaving the authoritative bitset untouched
            // until we decide to remove.
            let mut word = self.occupancy.words()[wi];
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1; // clear lowest set bit (blsr)
                let slot = wi * 64 + bit;
                let gen =
                    unsafe { NonZeroU32::new_unchecked(*self.generations.get_unchecked(slot)) };
                let index = Index {
                    slot: slot as u32,
                    generation: gen,
                };
                let keep = f(index, unsafe {
                    self.values.get_unchecked_mut(slot).assume_init_mut()
                });
                if !keep {
                    unsafe { self.values.get_unchecked_mut(slot).assume_init_drop() };
                    self.occupancy.clear(slot);
                    self.free_list.push(slot as u32);
                    self.len -= 1;
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// TRAIT IMPLEMENTATIONS
// ══════════════════════════════════════════════════════════════════════

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// `arena[index]` — panics if index is invalid.
impl<T> ops::Index<Index> for Arena<T> {
    type Output = T;

    #[track_caller]
    fn index(&self, index: Index) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("No entry at index {:?}", index))
    }
}

/// `arena[index] = value` — panics if index is invalid.
impl<T> ops::IndexMut<Index> for Arena<T> {
    #[track_caller]
    fn index_mut(&mut self, index: Index) -> &mut Self::Output {
        self.get_mut(index)
            .unwrap_or_else(|| panic!("No entry at index {:?}", index))
    }
}

// ══════════════════════════════════════════════════════════════════════
// SEND + SYNC
// ══════════════════════════════════════════════════════════════════════
// Arena<T> holds no thread-local state and all raw pointers are owned
// exclusively. The auto-derived Send/Sync bounds propagate T's bounds
// correctly.  These compile-time assertions catch regressions.
// ══════════════════════════════════════════════════════════════════════

const _: () = {
    fn _assert_send<T: Send>() {
        fn check<U: Send>() {}
        check::<Arena<T>>();
    }
    fn _assert_sync<T: Sync>() {
        fn check<U: Sync>() {}
        check::<Arena<T>>();
    }
};

impl<T> IntoIterator for Arena<T> {
    type Item = (Index, T);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let words = self.occupancy.words();
        let mut word_idx = 0;
        let mut current_word = 0u64;
        for (i, &w) in words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                current_word = w;
                break;
            }
        }
        IntoIter {
            arena: self,
            word_idx,
            current_word,
        }
    }
}

impl<'a, T> IntoIterator for &'a Arena<T> {
    type Item = (Index, &'a T);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Arena<T> {
    type Item = (Index, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ══════════════════════════════════════════════════════════════════════
// DROP — Critical for Safety
// ══════════════════════════════════════════════════════════════════════
//
// We MUST drop all occupied values when the arena is dropped.
// MaybeUninit<T> does NOT auto-drop its contents — that's the whole point
// of MaybeUninit. If T has a Drop impl (e.g., String, Vec, Box),
// failing to drop it leaks memory.
//
// The bitset tells us exactly which slots need dropping.
//
// ══════════════════════════════════════════════════════════════════════

impl<T> Drop for Arena<T> {
    fn drop(&mut self) {
        if core::mem::needs_drop::<T>() {
            for slot in self.occupancy.iter_set_bits() {
                unsafe { self.values.get_unchecked_mut(slot).assume_init_drop() };
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// CLONE — Must deep-clone only occupied values
// ══════════════════════════════════════════════════════════════════════

impl<T: Clone> Clone for Arena<T> {
    fn clone(&self) -> Self {
        let cap = self.values.len();
        let mut new_values: Vec<MaybeUninit<T>> = Vec::with_capacity(cap);
        unsafe { new_values.set_len(cap) };
        for slot in self.occupancy.iter_set_bits() {
            let cloned = unsafe { self.values.get_unchecked(slot).assume_init_ref() }.clone();
            unsafe {
                new_values
                    .as_mut_ptr()
                    .add(slot)
                    .write(MaybeUninit::new(cloned))
            };
        }
        Self {
            occupancy: self.occupancy.clone(),
            values: new_values,
            generations: self.generations.clone(),
            free_list: self.free_list.clone(),
            next_fresh: self.next_fresh,
            len: self.len,
        }
    }
}

impl<T: PartialEq> PartialEq for Arena<T> {
    /// Two arenas are equal if they contain the same (Index, value) pairs.
    /// Capacity and free-list ordering are ignored.
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        // For every occupied slot in self, other must have the same
        // generation and equal value at the same slot.
        for slot in self.occupancy.iter_set_bits() {
            if !other.occupancy.is_set(slot) {
                return false;
            }
            let self_gen = unsafe { *self.generations.get_unchecked(slot) };
            let other_gen = unsafe { *other.generations.get_unchecked(slot) };
            if self_gen != other_gen {
                return false;
            }
            let self_val = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
            let other_val = unsafe { other.values.get_unchecked(slot).assume_init_ref() };
            if self_val != other_val {
                return false;
            }
        }
        true
    }
}

impl<T: Eq> Eq for Arena<T> {}

impl<T> Extend<T> for Arena<T> {
    /// Insert all values from an iterator, discarding the returned indices.
    ///
    /// Use [`Arena::insert`] directly if you need the indices.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl<T> core::iter::FromIterator<T> for Arena<T> {
    /// Build an arena from an iterator of values.
    ///
    /// Slots are assigned in insertion order starting from slot 0.
    /// To keep the returned indices, use [`Arena::insert`] in a loop instead.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lo, _) = iter.size_hint();
        let mut arena = if lo > 0 {
            Arena::with_capacity(lo)
        } else {
            Arena::new()
        };
        for value in iter {
            arena.insert(value);
        }
        arena
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut d = f.debug_map();
        for (index, value) in self.iter() {
            d.entry(&index, value);
        }
        d.finish()
    }
}

// ══════════════════════════════════════════════════════════════════════
// TESTS
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    // ──────────────────────────────────────────────────────────────────
    // BASIC CRUD
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn insert_and_get() {
        let mut arena = Arena::new();
        let idx = arena.insert(42u32);
        assert_eq!(arena.get(idx), Some(&42u32));
    }

    #[test]
    fn insert_multiple() {
        let mut arena = Arena::new();
        let a = arena.insert(1u32);
        let b = arena.insert(2u32);
        let c = arena.insert(3u32);
        assert_eq!(arena[a], 1);
        assert_eq!(arena[b], 2);
        assert_eq!(arena[c], 3);
    }

    #[test]
    fn remove_and_verify_gone() {
        let mut arena = Arena::new();
        let idx = arena.insert(99u32);
        arena.remove(idx);
        assert_eq!(arena.get(idx), None);
    }

    #[test]
    fn remove_returns_value() {
        let mut arena = Arena::new();
        let idx = arena.insert("hello");
        assert_eq!(arena.remove(idx), Some("hello"));
    }

    #[test]
    fn stale_index_returns_none() {
        let mut arena: Arena<u32> = Arena::new();
        let old = arena.insert(1);
        arena.remove(old);
        let _new = arena.insert(2); // reuses slot, bumps generation
        assert_eq!(arena.get(old), None);
    }

    #[test]
    fn slot_reuse() {
        let mut arena = Arena::new();
        let a = arena.insert('A');
        let _b = arena.insert('B');
        arena.remove(a);
        let c = arena.insert('C');
        // LIFO: c should reuse a's slot.
        assert_eq!(a.slot, c.slot);
        assert_ne!(a.generation, c.generation);
        assert_eq!(arena[c], 'C');
    }

    #[test]
    fn with_capacity_does_not_allocate_extra() {
        let arena: Arena<u32> = Arena::with_capacity(8);
        assert!(arena.capacity() >= 8);
    }

    #[test]
    fn grows_on_insert_beyond_capacity() {
        let mut arena = Arena::with_capacity(2);
        let indices: alloc::vec::Vec<_> = (0..10u32).map(|i| arena.insert(i)).collect();
        for (i, idx) in indices.iter().enumerate() {
            assert_eq!(arena[*idx], i as u32);
        }
    }

    #[test]
    fn iter_returns_all_occupied() {
        let mut arena = Arena::new();
        let a = arena.insert(1u32);
        let b = arena.insert(2u32);
        let c = arena.insert(3u32);
        let d = arena.insert(4u32);
        let e = arena.insert(5u32);
        arena.remove(b);
        arena.remove(d);
        let values: alloc::collections::BTreeSet<u32> = arena.iter().map(|(_, &v)| v).collect();
        assert_eq!(values, [1, 3, 5].iter().copied().collect());
        let _ = (a, c, e);
    }

    #[test]
    fn iter_empty_arena() {
        let arena: Arena<u32> = Arena::new();
        assert!(arena.iter().next().is_none());
    }

    #[test]
    fn iter_exact_size() {
        let mut arena = Arena::new();
        arena.insert(1u32);
        arena.insert(2u32);
        arena.insert(3u32);
        assert_eq!(arena.iter().len(), arena.len());
    }

    #[test]
    fn drop_calls_destructor() {
        // Verify no panic when arena with String values is dropped.
        let mut arena: Arena<alloc::string::String> = Arena::new();
        let i1 = arena.insert(alloc::string::String::from("hello"));
        let i2 = arena.insert(alloc::string::String::from("world"));
        arena.remove(i1);
        drop(arena); // must not double-free or leak
        let _ = i2;
    }

    #[test]
    fn drop_calls_destructor_precise() {
        use alloc::rc::Rc;
        use core::cell::Cell;

        struct CountDrop(Rc<Cell<u32>>);
        impl Drop for CountDrop {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let counter = Rc::new(Cell::new(0u32));
        {
            let mut arena = Arena::new();
            for _ in 0..5 {
                arena.insert(CountDrop(counter.clone()));
            }
            // Remove 2
            let ids: alloc::vec::Vec<_> = arena.iter().map(|(i, _)| i).take(2).collect();
            for id in ids {
                arena.remove(id);
            }
            // counter should be 2 now (from the removes)
            assert_eq!(counter.get(), 2);
            // Drop the arena: remaining 3 should be dropped
        }
        assert_eq!(counter.get(), 5);
    }

    #[test]
    fn clone_independence() {
        let mut arena = Arena::new();
        arena.insert(alloc::string::String::from("original"));
        let mut clone = arena.clone();
        // Modify the clone
        for (_, v) in clone.iter_mut() {
            v.push_str(" modified");
        }
        // Original unchanged
        for (_, v) in arena.iter() {
            assert_eq!(v, "original");
        }
    }

    #[test]
    fn index_size() {
        assert_eq!(core::mem::size_of::<Index>(), 8);
        assert_eq!(core::mem::size_of::<Option<Index>>(), 8);
    }

    #[test]
    fn get2_mut_different_slots() {
        let mut arena = Arena::new();
        let a = arena.insert(10u32);
        let b = arena.insert(20u32);
        let (ra, rb) = arena.get2_mut(a, b);
        assert_eq!(ra, Some(&mut 10u32));
        assert_eq!(rb, Some(&mut 20u32));
    }

    #[test]
    #[should_panic]
    fn get2_mut_same_slot_panics() {
        let mut arena = Arena::new();
        let a = arena.insert(1u32);
        arena.get2_mut(a, a);
    }

    #[test]
    fn retain_keeps_matching() {
        let mut arena = Arena::new();
        for i in 0u32..10 {
            arena.insert(i);
        }
        arena.retain(|_, v| *v % 2 == 0);
        assert_eq!(arena.len(), 5);
        for (_, v) in arena.iter() {
            assert_eq!(v % 2, 0);
        }
    }

    #[test]
    fn insert_at_specific_index() {
        let mut arena: Arena<u32> = Arena::new();
        let idx = arena.insert(1);
        let old = arena.insert_at(idx, 99);
        // Replacing an existing slot returns the old value.
        assert_eq!(old, Some(1));
        assert_eq!(arena[idx], 99);
    }

    #[test]
    fn invalidate_old_index() {
        let mut arena = Arena::new();
        let old = arena.insert(42u32);
        let new = arena.invalidate(old).unwrap();
        assert!(!arena.contains(old));
        assert!(arena.contains(new));
        assert_eq!(arena[new], 42);
    }

    // ──────────────────────────────────────────────────────────────────
    // values() / values_mut() / keys() iterator tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn values_iter_sum() {
        let mut arena = Arena::new();
        arena.insert(1u64);
        arena.insert(2u64);
        arena.insert(3u64);
        let sum: u64 = arena.values().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn values_mut_iter_modifies() {
        let mut arena = Arena::new();
        arena.insert(1u32);
        arena.insert(2u32);
        arena.insert(3u32);
        arena.values_mut().for_each(|v| *v *= 10);
        let mut vals: alloc::vec::Vec<u32> = arena.values().copied().collect();
        vals.sort();
        assert_eq!(vals, [10, 20, 30]);
    }

    #[test]
    fn keys_iter_count() {
        let mut arena = Arena::new();
        let a = arena.insert(1u32);
        let b = arena.insert(2u32);
        let c = arena.insert(3u32);
        let _ = c;
        arena.remove(b);
        let keys: alloc::vec::Vec<_> = arena.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&a));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_values_match_values() {
        let mut arena = Arena::with_capacity(320);
        let mut ids = alloc::vec::Vec::new();
        for i in 0..320u64 {
            ids.push(arena.insert(i));
        }
        for (i, idx) in ids.iter().enumerate() {
            if i % 3 == 0 || i % 11 == 1 {
                arena.remove(*idx);
            }
        }

        let mut seq: alloc::vec::Vec<u64> = arena.values().copied().collect();
        let mut par: alloc::vec::Vec<u64> = arena.par_values().copied().collect();
        seq.sort_unstable();
        par.sort_unstable();
        assert_eq!(seq, par);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_iter_matches_iter() {
        let mut arena = Arena::with_capacity(384);
        let mut ids = alloc::vec::Vec::new();
        for i in 0..384u64 {
            ids.push(arena.insert(i.wrapping_mul(13)));
        }
        for (i, idx) in ids.iter().enumerate() {
            if i % 4 == 0 || i % 7 == 2 {
                arena.remove(*idx);
            }
        }

        let mut seq: alloc::vec::Vec<(u64, u64)> = arena
            .iter()
            .map(|(idx, &v)| (idx.to_bits(), v))
            .collect();
        let mut par: alloc::vec::Vec<(u64, u64)> = arena
            .par_iter()
            .map(|(idx, v)| (idx.to_bits(), *v))
            .collect();
        seq.sort_unstable();
        par.sort_unstable();
        assert_eq!(seq, par);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_keys_match_keys() {
        let mut arena = Arena::with_capacity(256);
        let mut ids = alloc::vec::Vec::new();
        for i in 0..256u32 {
            ids.push(arena.insert(i));
        }
        for (i, idx) in ids.iter().enumerate() {
            if i % 5 == 0 {
                arena.remove(*idx);
            }
        }

        let mut seq: alloc::vec::Vec<u64> = arena.keys().map(|k| k.to_bits()).collect();
        let mut par: alloc::vec::Vec<u64> = arena.par_keys().map(|k| k.to_bits()).collect();
        seq.sort_unstable();
        par.sort_unstable();
        assert_eq!(seq, par);
    }

    #[test]
    fn from_iterator_and_extend() {
        let arena: Arena<u32> = (0u32..5).collect();
        assert_eq!(arena.len(), 5);

        let mut arena2: Arena<u32> = Arena::new();
        arena2.extend([10u32, 20, 30]);
        assert_eq!(arena2.len(), 3);
    }

    #[test]
    fn partial_eq_arenas() {
        let arena1: Arena<u32> = (0u32..3).collect();
        let arena2: Arena<u32> = (0u32..3).collect();
        assert_eq!(arena1, arena2);
    }
}
