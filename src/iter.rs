// ══════════════════════════════════════════════════════════════════════
// Iterators — Bitset-Driven Iteration
// ══════════════════════════════════════════════════════════════════════
//
// SoA ITERATION STRATEGY
// ──────────────────────
// The arena uses Structure-of-Arrays layout:
//   - occupancy: Vec<u64>         (bitset, ~1.3KB for 10k slots, fits in L1)
//   - values:    Vec<MaybeUninit<T>>  (only touched data during iteration)
//   - generations: Vec<u32>       (NOT touched during the iteration hot path)
//
// During iteration, we scan the occupancy bitset and load values.
// The generation is only read to construct the Index that we yield.
// LLVM can dead-code-eliminate the generation load when the caller
// discards the Index (e.g., `for (_, &v) in arena.iter()`).
//
// This means iteration touches only 2 memory streams:
//   1. occupancy words (tiny, L1-resident)
//   2. values array (8 bytes per element for T=u64)
//
// Compare to DenseSlotMap which touches:
//   1. keys array (8 bytes per element)
//   2. values array (8 bytes per element)
//
// We match their data volume but can skip empty regions in bulk.
//
// PREFETCH STRATEGY
// ─────────────────
// In fold(), we issue software prefetch hints one word ahead using:
//   - x86_64: _mm_prefetch(_MM_HINT_T0)
//   - aarch64: core::arch::aarch64::_prefetch (PREFETCH_READ + PREFETCH_LOCALITY3)
//
// ══════════════════════════════════════════════════════════════════════

use core::iter::{ExactSizeIterator, FusedIterator};
use core::mem::MaybeUninit;
use core::num::NonZeroU32;

use crate::index::Index;

// ──────────────────────────────────────────────────────────────────────
// Iter — Immutable iteration: yields (Index, &T)
// ──────────────────────────────────────────────────────────────────────

pub struct Iter<'a, T> {
    pub(crate) occupancy_words: &'a [u64],
    pub(crate) values: &'a [MaybeUninit<T>],
    pub(crate) generations: &'a [u32],
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
    pub(crate) remaining: u32,
    pub(crate) base_slot: usize,
    /// Non-zero when iterating a fully-occupied word (word == u64::MAX).
    /// Holds the exclusive end slot of the dense run.
    /// In dense mode: `base_slot` is the *current* slot; `dense_end` is the
    /// one-past-the-end.  We just increment `base_slot` each call — no
    /// tzcnt/blsr needed.
    pub(crate) dense_end: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Index, &'a T);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // ── DENSE MODE: sequential scan through a fully-occupied word ──
        if self.base_slot < self.dense_end {
            let slot = self.base_slot;
            self.base_slot = slot + 1;
            self.remaining -= 1;
            // Cast via raw pointer so LLVM sees a simple pointer dereference
            // (same pattern as slice::Iter) rather than a bounds-checked index.
            let value = unsafe { &*self.values.as_ptr().add(slot).cast::<T>() };
            let gen = unsafe { *self.generations.as_ptr().add(slot) };
            return Some((
                Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                },
                value,
            ));
        }

        // ── SPARSE MODE: tzcnt+blsr through partial words ──────────────
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.base_slot + bit;
                let value = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
                let gen = unsafe { *self.generations.get_unchecked(slot) };
                let index = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                self.remaining -= 1;
                return Some((index, value));
            }
            self.word_idx += 1;
            loop {
                if self.word_idx >= self.occupancy_words.len() {
                    return None;
                }
                let w = unsafe { *self.occupancy_words.get_unchecked(self.word_idx) };
                let base = self.word_idx << 6;
                if w == u64::MAX {
                    // Full word: enter dense mode — zero tzcnt/blsr overhead.
                    self.base_slot = base + 1;
                    self.dense_end = base + 64;
                    self.current_word = 0;
                    self.remaining -= 1;
                    let value = unsafe { &*self.values.as_ptr().add(base).cast::<T>() };
                    let gen = unsafe { *self.generations.as_ptr().add(base) };
                    return Some((
                        Index {
                            slot: base as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        },
                        value,
                    ));
                } else if w != 0 {
                    self.current_word = w;
                    self.base_slot = base;
                    self.dense_end = 0; // leave dense mode
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }

    /// Override `fold()` — called by `.for_each()`, `.sum()`, `.collect()`, etc.
    ///
    /// When a 64-slot occupancy word is `u64::MAX` (all occupied), we use a
    /// simple `for i in 0..64` loop with constant bounds — LLVM can auto-vectorise
    /// this with AVX2 (4 × u64 per cycle) when the Index is dead-code-eliminated.
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        // Drain any remaining items in the current active run.
        if self.base_slot < self.dense_end {
            // Currently in a dense run — emit the rest of it.
            let end = self.dense_end;
            for slot in self.base_slot..end {
                let v = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
                let g = unsafe { *self.generations.get_unchecked(slot) };
                let idx = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(g) },
                };
                acc = f(acc, (idx, v));
            }
        } else if self.current_word != 0 {
            // Partial word — emit remaining bits.
            acc = unsafe {
                Self::fold_partial_word(
                    self.base_slot,
                    self.current_word,
                    self.values,
                    self.generations,
                    acc,
                    &mut f,
                )
            };
        }

        // Process all remaining whole words.
        let start = self
            .word_idx
            .saturating_add(1)
            .min(self.occupancy_words.len());
        for (wi, &word) in self.occupancy_words[start..].iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (start + wi) << 6;

            // Prefetch the values cache line for the NEXT word's base slot.
            let next_base = base + 64;
            #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
            unsafe {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                if next_base < self.values.len() {
                    _mm_prefetch(
                        self.values.as_ptr().add(next_base) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
                if next_base < self.values.len() {
                    _prefetch(
                        self.values.as_ptr().add(next_base) as *const i8,
                        _PREFETCH_READ,
                        _PREFETCH_LOCALITY3,
                    );
                }
            }

            if word == u64::MAX {
                acc = unsafe {
                    Self::fold_full_word(base, self.values, self.generations, acc, &mut f)
                };
            } else {
                acc = unsafe {
                    Self::fold_partial_word(base, word, self.values, self.generations, acc, &mut f)
                };
            }
        }
        acc
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}
impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> Iter<'a, T> {
    /// Emit all 64 items in a fully-occupied word (`word == u64::MAX`).
    ///
    /// A `for i in 0..64` loop with constant bounds lets LLVM auto-vectorise
    /// the body with AVX2 (4 × u64 per cycle) instead of the tzcnt+blsr path.
    /// At 99% occupancy ~155/157 words hit this fast path.
    #[inline(always)]
    unsafe fn fold_full_word<B, F>(
        base: usize,
        values: &'a [MaybeUninit<T>],
        generations: &'a [u32],
        mut acc: B,
        f: &mut F,
    ) -> B
    where
        F: FnMut(B, (Index, &'a T)) -> B,
    {
        for i in 0..64usize {
            let slot = base + i;
            let v = unsafe { values.get_unchecked(slot).assume_init_ref() };
            let g = unsafe { *generations.get_unchecked(slot) };
            let idx = Index {
                slot: slot as u32,
                generation: unsafe { NonZeroU32::new_unchecked(g) },
            };
            acc = f(acc, (idx, v));
        }
        acc
    }

    /// Emit all set bits in a partial word via tzcnt+blsr.
    #[inline(always)]
    unsafe fn fold_partial_word<B, F>(
        base: usize,
        mut word: u64,
        values: &'a [MaybeUninit<T>],
        generations: &'a [u32],
        mut acc: B,
        f: &mut F,
    ) -> B
    where
        F: FnMut(B, (Index, &'a T)) -> B,
    {
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= word - 1;
            let slot = base + bit;
            let v = unsafe { values.get_unchecked(slot).assume_init_ref() };
            let g = unsafe { *generations.get_unchecked(slot) };
            let idx = Index {
                slot: slot as u32,
                generation: unsafe { NonZeroU32::new_unchecked(g) },
            };
            acc = f(acc, (idx, v));
        }
        acc
    }
}

// ──────────────────────────────────────────────────────────────────────
// IterMut — Mutable iteration: yields (Index, &mut T)
// ──────────────────────────────────────────────────────────────────────

pub struct IterMut<'a, T> {
    pub(crate) occupancy_words: &'a [u64],
    pub(crate) values_ptr: *mut MaybeUninit<T>,
    pub(crate) generations_ptr: *const u32,
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
    pub(crate) remaining: u32,
    pub(crate) base_slot: usize,
    /// Non-zero when iterating a fully-occupied word.  Same semantics as
    /// `Iter::dense_end`.
    pub(crate) dense_end: usize,
    pub(crate) _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (Index, &'a mut T);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ── DENSE MODE ────────────────────────────────────────────────
        if self.base_slot < self.dense_end {
            let slot = self.base_slot;
            self.base_slot = slot + 1;
            self.remaining -= 1;
            let value = unsafe { &mut *self.values_ptr.add(slot).cast::<T>() };
            let gen = unsafe { *self.generations_ptr.add(slot) };
            return Some((
                Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                },
                value,
            ));
        }

        // ── SPARSE MODE ───────────────────────────────────────────────
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.base_slot + bit;
                let value = unsafe { (*self.values_ptr.add(slot)).assume_init_mut() };
                let gen = unsafe { *self.generations_ptr.add(slot) };
                let index = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                self.remaining -= 1;
                return Some((index, value));
            }
            self.word_idx += 1;
            loop {
                if self.word_idx >= self.occupancy_words.len() {
                    return None;
                }
                let w = unsafe { *self.occupancy_words.get_unchecked(self.word_idx) };
                let base = self.word_idx << 6;
                if w == u64::MAX {
                    self.base_slot = base + 1;
                    self.dense_end = base + 64;
                    self.current_word = 0;
                    self.remaining -= 1;
                    let value = unsafe { &mut *self.values_ptr.add(base).cast::<T>() };
                    let gen = unsafe { *self.generations_ptr.add(base) };
                    return Some((
                        Index {
                            slot: base as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        },
                        value,
                    ));
                } else if w != 0 {
                    self.current_word = w;
                    self.base_slot = base;
                    self.dense_end = 0;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        // Drain any remaining items in the current active run.
        if self.base_slot < self.dense_end {
            for slot in self.base_slot..self.dense_end {
                let value = unsafe { (*self.values_ptr.add(slot)).assume_init_mut() };
                let gen = unsafe { *self.generations_ptr.add(slot) };
                let idx = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                acc = f(acc, (idx, value));
            }
        } else if self.current_word != 0 {
            let mut w = self.current_word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                let slot = self.base_slot + bit;
                let value = unsafe { (*self.values_ptr.add(slot)).assume_init_mut() };
                let gen = unsafe { *self.generations_ptr.add(slot) };
                let idx = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                acc = f(acc, (idx, value));
            }
        }

        let start = self
            .word_idx
            .saturating_add(1)
            .min(self.occupancy_words.len());
        for (wi, &word) in self.occupancy_words[start..].iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (start + wi) << 6;

            let next_base = base + 64;
            let values_len = self.occupancy_words.len() << 6;
            #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
            unsafe {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                if next_base < values_len {
                    _mm_prefetch(self.values_ptr.add(next_base) as *const i8, _MM_HINT_T0);
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
                if next_base < values_len {
                    _prefetch(
                        self.values_ptr.add(next_base) as *const i8,
                        _PREFETCH_READ,
                        _PREFETCH_LOCALITY3,
                    );
                }
            }

            if word == u64::MAX {
                for i in 0..64usize {
                    let slot = base + i;
                    let value = unsafe { (*self.values_ptr.add(slot)).assume_init_mut() };
                    let gen = unsafe { *self.generations_ptr.add(slot) };
                    let idx = Index {
                        slot: slot as u32,
                        generation: unsafe { NonZeroU32::new_unchecked(gen) },
                    };
                    acc = f(acc, (idx, value));
                }
            } else {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    let slot = base + bit;
                    let value = unsafe { (*self.values_ptr.add(slot)).assume_init_mut() };
                    let gen = unsafe { *self.generations_ptr.add(slot) };
                    let idx = Index {
                        slot: slot as u32,
                        generation: unsafe { NonZeroU32::new_unchecked(gen) },
                    };
                    acc = f(acc, (idx, value));
                }
            }
        }
        acc
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}
impl<'a, T> FusedIterator for IterMut<'a, T> {}

// ──────────────────────────────────────────────────────────────────────
// Values — Immutable iteration over values only: yields &T
//
// Unlike Iter, Values never reads the generations array, so there is
// no Index construction overhead.  This lets LLVM freely auto-vectorise
// the full-word (word == u64::MAX) fold() path with AVX2 / NEON.
// ──────────────────────────────────────────────────────────────────────

pub struct Values<'a, T> {
    pub(crate) occupancy_words: &'a [u64],
    pub(crate) values: &'a [MaybeUninit<T>],
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
    pub(crate) remaining: u32,
    pub(crate) base_slot: usize,
    pub(crate) dense_end: usize,
}

impl<'a, T> Values<'a, T> {
    pub(crate) fn new(
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
        remaining: u32,
    ) -> Self {
        let mut word_idx = 0usize;
        let mut current_word = 0u64;
        let mut base_slot = 0usize;
        let mut dense_end = 0usize;
        for (i, &w) in occupancy_words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                base_slot = i << 6;
                if w == u64::MAX {
                    dense_end = base_slot + 64;
                } else {
                    current_word = w;
                }
                break;
            }
        }
        Self {
            occupancy_words,
            values,
            word_idx,
            current_word,
            remaining,
            base_slot,
            dense_end,
        }
    }
}

impl<'a, T> Iterator for Values<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.base_slot < self.dense_end {
            let slot = self.base_slot;
            self.base_slot = slot + 1;
            self.remaining -= 1;
            return Some(unsafe { &*self.values.as_ptr().add(slot).cast::<T>() });
        }
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.base_slot + bit;
                self.remaining -= 1;
                return Some(unsafe { self.values.get_unchecked(slot).assume_init_ref() });
            }
            self.word_idx += 1;
            loop {
                if self.word_idx >= self.occupancy_words.len() {
                    return None;
                }
                let w = unsafe { *self.occupancy_words.get_unchecked(self.word_idx) };
                let base = self.word_idx << 6;
                if w == u64::MAX {
                    self.base_slot = base + 1;
                    self.dense_end = base + 64;
                    self.current_word = 0;
                    self.remaining -= 1;
                    return Some(unsafe { &*self.values.as_ptr().add(base).cast::<T>() });
                } else if w != 0 {
                    self.current_word = w;
                    self.base_slot = base;
                    self.dense_end = 0;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a T) -> B,
    {
        let mut acc = init;
        if self.base_slot < self.dense_end {
            for slot in self.base_slot..self.dense_end {
                acc = f(acc, unsafe {
                    self.values.get_unchecked(slot).assume_init_ref()
                });
            }
        } else if self.current_word != 0 {
            let mut w = self.current_word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                acc = f(acc, unsafe {
                    self.values
                        .get_unchecked(self.base_slot + bit)
                        .assume_init_ref()
                });
            }
        }
        let start = self
            .word_idx
            .saturating_add(1)
            .min(self.occupancy_words.len());
        for (wi, &word) in self.occupancy_words[start..].iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (start + wi) << 6;
            let next_base = base + 64;
            #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
            unsafe {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                if next_base < self.values.len() {
                    _mm_prefetch(
                        self.values.as_ptr().add(next_base) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
                if next_base < self.values.len() {
                    _prefetch(
                        self.values.as_ptr().add(next_base) as *const i8,
                        _PREFETCH_READ,
                        _PREFETCH_LOCALITY3,
                    );
                }
            }
            if word == u64::MAX {
                for i in 0..64usize {
                    acc = f(acc, unsafe {
                        self.values.get_unchecked(base + i).assume_init_ref()
                    });
                }
            } else {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    acc = f(acc, unsafe {
                        self.values.get_unchecked(base + bit).assume_init_ref()
                    });
                }
            }
        }
        acc
    }
}

impl<'a, T> ExactSizeIterator for Values<'a, T> {}
impl<'a, T> FusedIterator for Values<'a, T> {}

// ──────────────────────────────────────────────────────────────────────
// ValuesMut — Mutable iteration over values only: yields &mut T
// ──────────────────────────────────────────────────────────────────────

pub struct ValuesMut<'a, T> {
    pub(crate) occupancy_words: &'a [u64],
    pub(crate) values_ptr: *mut MaybeUninit<T>,
    pub(crate) values_len: usize,
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
    pub(crate) remaining: u32,
    pub(crate) base_slot: usize,
    pub(crate) dense_end: usize,
    pub(crate) _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T> ValuesMut<'a, T> {
    pub(crate) fn new(
        occupancy_words: &'a [u64],
        values_ptr: *mut MaybeUninit<T>,
        values_len: usize,
        remaining: u32,
    ) -> Self {
        let mut word_idx = 0usize;
        let mut current_word = 0u64;
        let mut base_slot = 0usize;
        let mut dense_end = 0usize;
        for (i, &w) in occupancy_words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                base_slot = i << 6;
                if w == u64::MAX {
                    dense_end = base_slot + 64;
                } else {
                    current_word = w;
                }
                break;
            }
        }
        Self {
            occupancy_words,
            values_ptr,
            values_len,
            word_idx,
            current_word,
            remaining,
            base_slot,
            dense_end,
            _marker: core::marker::PhantomData,
        }
    }
}

unsafe impl<'a, T: Send> Send for ValuesMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for ValuesMut<'a, T> {}

impl<'a, T> Iterator for ValuesMut<'a, T> {
    type Item = &'a mut T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        if self.base_slot < self.dense_end {
            let slot = self.base_slot;
            self.base_slot = slot + 1;
            self.remaining -= 1;
            return Some(unsafe { &mut *self.values_ptr.add(slot).cast::<T>() });
        }
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.base_slot + bit;
                self.remaining -= 1;
                return Some(unsafe { (*self.values_ptr.add(slot)).assume_init_mut() });
            }
            self.word_idx += 1;
            loop {
                if self.word_idx >= self.occupancy_words.len() {
                    return None;
                }
                let w = unsafe { *self.occupancy_words.get_unchecked(self.word_idx) };
                let base = self.word_idx << 6;
                if w == u64::MAX {
                    self.base_slot = base + 1;
                    self.dense_end = base + 64;
                    self.current_word = 0;
                    self.remaining -= 1;
                    return Some(unsafe { &mut *self.values_ptr.add(base).cast::<T>() });
                } else if w != 0 {
                    self.current_word = w;
                    self.base_slot = base;
                    self.dense_end = 0;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a mut T) -> B,
    {
        let mut acc = init;
        if self.base_slot < self.dense_end {
            for slot in self.base_slot..self.dense_end {
                acc = f(acc, unsafe {
                    (*self.values_ptr.add(slot)).assume_init_mut()
                });
            }
        } else if self.current_word != 0 {
            let mut w = self.current_word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                acc = f(acc, unsafe {
                    (*self.values_ptr.add(self.base_slot + bit)).assume_init_mut()
                });
            }
        }
        let start = self
            .word_idx
            .saturating_add(1)
            .min(self.occupancy_words.len());
        for (wi, &word) in self.occupancy_words[start..].iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (start + wi) << 6;
            let next_base = base + 64;
            #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
            unsafe {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                if next_base < self.values_len {
                    _mm_prefetch(self.values_ptr.add(next_base) as *const i8, _MM_HINT_T0);
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
                if next_base < self.values_len {
                    _prefetch(
                        self.values_ptr.add(next_base) as *const i8,
                        _PREFETCH_READ,
                        _PREFETCH_LOCALITY3,
                    );
                }
            }
            if word == u64::MAX {
                for i in 0..64usize {
                    acc = f(acc, unsafe {
                        (*self.values_ptr.add(base + i)).assume_init_mut()
                    });
                }
            } else {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    acc = f(acc, unsafe {
                        (*self.values_ptr.add(base + bit)).assume_init_mut()
                    });
                }
            }
        }
        acc
    }
}

impl<'a, T> ExactSizeIterator for ValuesMut<'a, T> {}
impl<'a, T> FusedIterator for ValuesMut<'a, T> {}

// ──────────────────────────────────────────────────────────────────────
// Keys — Iterate over valid Index values only: yields Index
//
// Only reads occupancy_words and generations — never touches values.
// Useful for "collect all live keys" before a mutation pass.
// ──────────────────────────────────────────────────────────────────────

/// An iterator over the live [`Index`] handles of an [`Arena`](crate::Arena).
///
/// Obtained via [`Arena::keys`](crate::Arena::keys). Yields every valid
/// [`Index`] without loading values, making it the cheapest way to collect
/// all live handles.
pub struct Keys<'a> {
    pub(crate) occupancy_words: &'a [u64],
    pub(crate) generations: &'a [u32],
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
    pub(crate) remaining: u32,
    pub(crate) base_slot: usize,
}

impl<'a> Keys<'a> {
    pub(crate) fn new(occupancy_words: &'a [u64], generations: &'a [u32], remaining: u32) -> Self {
        let mut word_idx = 0usize;
        let mut current_word = 0u64;
        let mut base_slot = 0usize;
        for (i, &w) in occupancy_words.iter().enumerate() {
            if w != 0 {
                word_idx = i;
                base_slot = i << 6;
                current_word = w;
                break;
            }
        }
        Self {
            occupancy_words,
            generations,
            word_idx,
            current_word,
            remaining,
            base_slot,
        }
    }
}

impl<'a> Iterator for Keys<'a> {
    type Item = Index;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.base_slot + bit;
                let gen = unsafe { *self.generations.get_unchecked(slot) };
                self.remaining -= 1;
                return Some(Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                });
            }
            self.word_idx += 1;
            loop {
                if self.word_idx >= self.occupancy_words.len() {
                    return None;
                }
                let w = unsafe { *self.occupancy_words.get_unchecked(self.word_idx) };
                if w != 0 {
                    self.current_word = w;
                    self.base_slot = self.word_idx << 6;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Index) -> B,
    {
        let mut acc = init;
        if self.current_word != 0 {
            let mut w = self.current_word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                let slot = self.base_slot + bit;
                let gen = unsafe { *self.generations.get_unchecked(slot) };
                acc = f(
                    acc,
                    Index {
                        slot: slot as u32,
                        generation: unsafe { NonZeroU32::new_unchecked(gen) },
                    },
                );
            }
        }
        let start = self
            .word_idx
            .saturating_add(1)
            .min(self.occupancy_words.len());
        for (wi, &word) in self.occupancy_words[start..].iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (start + wi) << 6;
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                let slot = base + bit;
                let gen = unsafe { *self.generations.get_unchecked(slot) };
                acc = f(
                    acc,
                    Index {
                        slot: slot as u32,
                        generation: unsafe { NonZeroU32::new_unchecked(gen) },
                    },
                );
            }
        }
        acc
    }
}

impl<'a> ExactSizeIterator for Keys<'a> {}
impl<'a> FusedIterator for Keys<'a> {}

// ──────────────────────────────────────────────────────────────────────
// IntoIter — Consuming iteration: yields (Index, T)
// ──────────────────────────────────────────────────────────────────────

pub struct IntoIter<T> {
    pub(crate) arena: crate::arena::Arena<T>,
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (Index, T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.word_idx * 64 + bit;
                let gen = unsafe { *self.arena.generations.get_unchecked(slot) };
                let value = unsafe { self.arena.values.as_ptr().add(slot).cast::<T>().read() };
                self.arena.occupancy.clear(slot);
                self.arena.len -= 1;
                let index = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                return Some((index, value));
            }
            self.word_idx += 1;
            let words = self.arena.occupancy.words();
            loop {
                if self.word_idx >= words.len() {
                    return None;
                }
                let w = unsafe { *words.get_unchecked(self.word_idx) };
                if w != 0 {
                    self.current_word = w;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.arena.len as usize;
        (r, Some(r))
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        // Drain current partial word first.
        if self.current_word != 0 {
            let mut w = self.current_word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                let slot = self.word_idx * 64 + bit;
                let gen = unsafe { *self.arena.generations.get_unchecked(slot) };
                let value = unsafe { self.arena.values.as_ptr().add(slot).cast::<T>().read() };
                self.arena.occupancy.clear(slot);
                self.arena.len -= 1;
                let idx = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                acc = f(acc, (idx, value));
            }
            self.current_word = 0;
        }

        let start = self
            .word_idx
            .saturating_add(1)
            .min(self.arena.occupancy.words().len());
        let num_words = self.arena.occupancy.words().len();
        for wi in start..num_words {
            let word = unsafe { *self.arena.occupancy.words().get_unchecked(wi) };
            if word == 0 {
                continue;
            }
            let base = wi << 6;
            if word == u64::MAX {
                for i in 0..64usize {
                    let slot = base + i;
                    let gen = unsafe { *self.arena.generations.get_unchecked(slot) };
                    let value = unsafe { self.arena.values.as_ptr().add(slot).cast::<T>().read() };
                    self.arena.len -= 1;
                    acc = f(
                        acc,
                        (
                            Index {
                                slot: slot as u32,
                                generation: unsafe { NonZeroU32::new_unchecked(gen) },
                            },
                            value,
                        ),
                    );
                }
                unsafe { *self.arena.occupancy.words_mut().get_unchecked_mut(wi) = 0 };
            } else {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    let slot = base + bit;
                    let gen = unsafe { *self.arena.generations.get_unchecked(slot) };
                    let value = unsafe { self.arena.values.as_ptr().add(slot).cast::<T>().read() };
                    self.arena.occupancy.clear(slot);
                    self.arena.len -= 1;
                    acc = f(
                        acc,
                        (
                            Index {
                                slot: slot as u32,
                                generation: unsafe { NonZeroU32::new_unchecked(gen) },
                            },
                            value,
                        ),
                    );
                }
            }
        }
        // Signal to Drop that everything is consumed.
        self.word_idx = num_words;
        acc
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}
impl<T> FusedIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}
    }
}

// ──────────────────────────────────────────────────────────────────────
// Drain — Draining iteration: yields (Index, T), empties the arena
// ──────────────────────────────────────────────────────────────────────

pub struct Drain<'a, T> {
    pub(crate) arena: *mut crate::arena::Arena<T>,
    pub(crate) word_idx: usize,
    pub(crate) current_word: u64,
    pub(crate) _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = (Index, T);

    fn next(&mut self) -> Option<Self::Item> {
        let arena = unsafe { &mut *self.arena };
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                let slot = self.word_idx * 64 + bit;
                let gen = unsafe { *arena.generations.get_unchecked(slot) };
                let value = unsafe { arena.values.as_ptr().add(slot).cast::<T>().read() };
                arena.occupancy.clear(slot);
                arena.free_list.push(slot as u32);
                arena.len -= 1;
                let index = Index {
                    slot: slot as u32,
                    generation: unsafe { NonZeroU32::new_unchecked(gen) },
                };
                return Some((index, value));
            }
            self.word_idx += 1;
            let words = arena.occupancy.words();
            loop {
                if self.word_idx >= words.len() {
                    return None;
                }
                let w = unsafe { *words.get_unchecked(self.word_idx) };
                if w != 0 {
                    self.current_word = w;
                    break;
                }
                self.word_idx += 1;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let arena = unsafe { &*self.arena };
        let r = arena.len as usize;
        (r, Some(r))
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        let arena = unsafe { &mut *self.arena };

        // Drain current partial word first.
        if self.current_word != 0 {
            let mut w = self.current_word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                let slot = self.word_idx * 64 + bit;
                let gen = unsafe { *arena.generations.get_unchecked(slot) };
                let value = unsafe { arena.values.as_ptr().add(slot).cast::<T>().read() };
                arena.occupancy.clear(slot);
                arena.free_list.push(slot as u32);
                arena.len -= 1;
                acc = f(
                    acc,
                    (
                        Index {
                            slot: slot as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        },
                        value,
                    ),
                );
            }
            self.current_word = 0;
        }

        let start = self
            .word_idx
            .saturating_add(1)
            .min(arena.occupancy.words().len());
        let num_words = arena.occupancy.words().len();
        for wi in start..num_words {
            let word = unsafe { *arena.occupancy.words().get_unchecked(wi) };
            if word == 0 {
                continue;
            }
            let base = wi << 6;
            if word == u64::MAX {
                for i in 0..64usize {
                    let slot = base + i;
                    let gen = unsafe { *arena.generations.get_unchecked(slot) };
                    let value = unsafe { arena.values.as_ptr().add(slot).cast::<T>().read() };
                    arena.free_list.push(slot as u32);
                    arena.len -= 1;
                    acc = f(
                        acc,
                        (
                            Index {
                                slot: slot as u32,
                                generation: unsafe { NonZeroU32::new_unchecked(gen) },
                            },
                            value,
                        ),
                    );
                }
                unsafe { *arena.occupancy.words_mut().get_unchecked_mut(wi) = 0 };
            } else {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    let slot = base + bit;
                    let gen = unsafe { *arena.generations.get_unchecked(slot) };
                    let value = unsafe { arena.values.as_ptr().add(slot).cast::<T>().read() };
                    arena.occupancy.clear(slot);
                    arena.free_list.push(slot as u32);
                    arena.len -= 1;
                    acc = f(
                        acc,
                        (
                            Index {
                                slot: slot as u32,
                                generation: unsafe { NonZeroU32::new_unchecked(gen) },
                            },
                            value,
                        ),
                    );
                }
            }
        }
        // Signal to Drop that everything is consumed.
        self.word_idx = num_words;
        acc
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}
impl<'a, T> FusedIterator for Drain<'a, T> {}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}
    }
}

// ══════════════════════════════════════════════════════════════════════
// RAYON PARALLEL ITERATORS (Feature-gated)
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
pub(crate) use rayon_impl::{par_iter, par_keys, par_values};

#[cfg(feature = "rayon")]
mod rayon_impl {
    use core::mem::MaybeUninit;
    use core::num::NonZeroU32;

    use rayon::iter::plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer};
    use rayon::iter::ParallelIterator;

    use crate::index::Index;

    const GRAIN_WORDS: usize = 128;

    #[inline(always)]
    fn split_mid(start_word: usize, end_word: usize) -> Option<usize> {
        let len = end_word - start_word;
        if len <= GRAIN_WORDS {
            None
        } else {
            Some(start_word + (len >> 1))
        }
    }

    pub(crate) fn par_values<'a, T: Sync>(
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
    ) -> ParValues<'a, T> {
        ParValues {
            occupancy_words,
            values,
        }
    }

    pub(crate) fn par_iter<'a, T: Sync>(
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
        generations: &'a [u32],
    ) -> ParIter<'a, T> {
        ParIter {
            occupancy_words,
            values,
            generations,
        }
    }

    pub(crate) fn par_keys<'a>(occupancy_words: &'a [u64], generations: &'a [u32]) -> ParKeys<'a> {
        ParKeys {
            occupancy_words,
            generations,
        }
    }

    pub(crate) struct ParValues<'a, T> {
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
    }

    impl<'a, T: Sync> ParallelIterator for ParValues<'a, T> {
        type Item = &'a T;

        #[inline]
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge_unindexed(
                ParValuesProducer {
                    occupancy_words: self.occupancy_words,
                    values: self.values,
                    start_word: 0,
                    end_word: self.occupancy_words.len(),
                },
                consumer,
            )
        }
    }

    #[derive(Clone, Copy)]
    struct ParValuesProducer<'a, T> {
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
        start_word: usize,
        end_word: usize,
    }

    impl<'a, T: Sync> UnindexedProducer for ParValuesProducer<'a, T> {
        type Item = &'a T;

        #[inline]
        fn split(self) -> (Self, Option<Self>) {
            let Some(mid) = split_mid(self.start_word, self.end_word) else {
                return (self, None);
            };
            (
                Self {
                    occupancy_words: self.occupancy_words,
                    values: self.values,
                    start_word: self.start_word,
                    end_word: mid,
                },
                Some(Self {
                    occupancy_words: self.occupancy_words,
                    values: self.values,
                    start_word: mid,
                    end_word: self.end_word,
                }),
            )
        }

        #[inline]
        fn fold_with<F>(self, mut folder: F) -> F
        where
            F: Folder<Self::Item>,
        {
            for wi in self.start_word..self.end_word {
                let word = unsafe { *self.occupancy_words.get_unchecked(wi) };
                if word == 0 {
                    continue;
                }

                let base = wi << 6;
                if word == u64::MAX {
                    for i in 0..64usize {
                        let slot = base + i;
                        let value = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
                        folder = folder.consume(value);
                        if folder.full() {
                            return folder;
                        }
                    }
                } else {
                    let mut w = word;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        let slot = base + bit;
                        let value = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
                        folder = folder.consume(value);
                        if folder.full() {
                            return folder;
                        }
                    }
                }
            }
            folder
        }
    }

    pub(crate) struct ParIter<'a, T> {
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
        generations: &'a [u32],
    }

    impl<'a, T: Sync> ParallelIterator for ParIter<'a, T> {
        type Item = (Index, &'a T);

        #[inline]
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge_unindexed(
                ParIterProducer {
                    occupancy_words: self.occupancy_words,
                    values: self.values,
                    generations: self.generations,
                    start_word: 0,
                    end_word: self.occupancy_words.len(),
                },
                consumer,
            )
        }
    }

    #[derive(Clone, Copy)]
    struct ParIterProducer<'a, T> {
        occupancy_words: &'a [u64],
        values: &'a [MaybeUninit<T>],
        generations: &'a [u32],
        start_word: usize,
        end_word: usize,
    }

    impl<'a, T: Sync> UnindexedProducer for ParIterProducer<'a, T> {
        type Item = (Index, &'a T);

        #[inline]
        fn split(self) -> (Self, Option<Self>) {
            let Some(mid) = split_mid(self.start_word, self.end_word) else {
                return (self, None);
            };
            (
                Self {
                    occupancy_words: self.occupancy_words,
                    values: self.values,
                    generations: self.generations,
                    start_word: self.start_word,
                    end_word: mid,
                },
                Some(Self {
                    occupancy_words: self.occupancy_words,
                    values: self.values,
                    generations: self.generations,
                    start_word: mid,
                    end_word: self.end_word,
                }),
            )
        }

        #[inline]
        fn fold_with<F>(self, mut folder: F) -> F
        where
            F: Folder<Self::Item>,
        {
            for wi in self.start_word..self.end_word {
                let word = unsafe { *self.occupancy_words.get_unchecked(wi) };
                if word == 0 {
                    continue;
                }

                let base = wi << 6;
                if word == u64::MAX {
                    for i in 0..64usize {
                        let slot = base + i;
                        let value = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
                        let gen = unsafe { *self.generations.get_unchecked(slot) };
                        let idx = Index {
                            slot: slot as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        };
                        folder = folder.consume((idx, value));
                        if folder.full() {
                            return folder;
                        }
                    }
                } else {
                    let mut w = word;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        let slot = base + bit;
                        let value = unsafe { self.values.get_unchecked(slot).assume_init_ref() };
                        let gen = unsafe { *self.generations.get_unchecked(slot) };
                        let idx = Index {
                            slot: slot as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        };
                        folder = folder.consume((idx, value));
                        if folder.full() {
                            return folder;
                        }
                    }
                }
            }
            folder
        }
    }

    pub(crate) struct ParKeys<'a> {
        occupancy_words: &'a [u64],
        generations: &'a [u32],
    }

    impl<'a> ParallelIterator for ParKeys<'a> {
        type Item = Index;

        #[inline]
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge_unindexed(
                ParKeysProducer {
                    occupancy_words: self.occupancy_words,
                    generations: self.generations,
                    start_word: 0,
                    end_word: self.occupancy_words.len(),
                },
                consumer,
            )
        }
    }

    #[derive(Clone, Copy)]
    struct ParKeysProducer<'a> {
        occupancy_words: &'a [u64],
        generations: &'a [u32],
        start_word: usize,
        end_word: usize,
    }

    impl<'a> UnindexedProducer for ParKeysProducer<'a> {
        type Item = Index;

        #[inline]
        fn split(self) -> (Self, Option<Self>) {
            let Some(mid) = split_mid(self.start_word, self.end_word) else {
                return (self, None);
            };
            (
                Self {
                    occupancy_words: self.occupancy_words,
                    generations: self.generations,
                    start_word: self.start_word,
                    end_word: mid,
                },
                Some(Self {
                    occupancy_words: self.occupancy_words,
                    generations: self.generations,
                    start_word: mid,
                    end_word: self.end_word,
                }),
            )
        }

        #[inline]
        fn fold_with<F>(self, mut folder: F) -> F
        where
            F: Folder<Self::Item>,
        {
            for wi in self.start_word..self.end_word {
                let word = unsafe { *self.occupancy_words.get_unchecked(wi) };
                if word == 0 {
                    continue;
                }

                let base = wi << 6;
                if word == u64::MAX {
                    for i in 0..64usize {
                        let slot = base + i;
                        let gen = unsafe { *self.generations.get_unchecked(slot) };
                        let idx = Index {
                            slot: slot as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        };
                        folder = folder.consume(idx);
                        if folder.full() {
                            return folder;
                        }
                    }
                } else {
                    let mut w = word;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        let slot = base + bit;
                        let gen = unsafe { *self.generations.get_unchecked(slot) };
                        let idx = Index {
                            slot: slot as u32,
                            generation: unsafe { NonZeroU32::new_unchecked(gen) },
                        };
                        folder = folder.consume(idx);
                        if folder.full() {
                            return folder;
                        }
                    }
                }
            }
            folder
        }
    }
}
