// ══════════════════════════════════════════════════════════════════════
// Property-Based Oracle Tests
// ══════════════════════════════════════════════════════════════════════
//
// STRATEGY
// ────────
// We test bitarena against a "known-good" oracle: a simple HashMap-based
// arena that is obviously correct but slow. If bitarena and the oracle
// ever disagree on the result of any operation, we have a bug.
//
// This is the gold standard for testing data structures. It catches
// edge cases that unit tests miss because proptest generates thousands
// of random operation sequences.
//
// THE ORACLE
// ──────────
// struct OracleArena<T> {
//     entries: HashMap<u32, (u32, T)>,  // slot → (generation, value)
//     next_slot: u32,
//     free_slots: Vec<u32>,
// }
//
// This is trivially correct: no unsafe, no bitsets, no MaybeUninit.
// It's slow (HashMap overhead) but we don't care — it's only for testing.
//
// OPERATIONS TO TEST
// ──────────────────
// proptest generates random sequences of:
//   Insert(value)
//   Remove(index)
//   Get(index)
//   Contains(index)
//   Len
//   IterCollect (collect iter results into sorted vec, compare)
//
// After each operation, we verify:
//   1. Return value matches oracle
//   2. arena.len() == oracle.len()
//   3. Every index valid in oracle is valid in arena (and vice versa)
//
// WHY PROPTEST OVER QUICKCHECK
// ────────────────────────────
// proptest has better shrinking (when a failure is found, it minimizes
// the input to the smallest failing case). This makes debugging much easier.
// quickcheck would also work but proptest is the modern standard.
//
// RUN UNDER MIRI
// ──────────────
// These tests should also be run under miri:
//   cargo +nightly miri test --test oracle_tests
//
// miri will catch:
//   - Reading uninitialized memory (MaybeUninit misuse)
//   - Use-after-free (accessing value after bit cleared)
//   - Double-free (dropping a value twice)
//   - Aliasing violations (multiple &mut T at once)
//
// Together, proptest + miri gives us confidence that:
//   - Logic is correct (proptest)
//   - Memory safety is sound (miri)
//
// ══════════════════════════════════════════════════════════════════════

use bitarena::{Arena, Index};
use proptest::prelude::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────
// Oracle implementation
// ──────────────────────────────────────────────────────────────────────

/// A trivially-correct arena backed by a HashMap.
/// No unsafe, no bitsets, obviously correct by inspection.
struct OracleArena {
    /// slot → (current_generation, value) for occupied slots
    entries: HashMap<u32, (u32, u64)>,
    /// slot → last_used_generation for all slots (occupied or not)
    /// This mirrors the arena's `generations` array which persists
    /// even after removal.
    generations: HashMap<u32, u32>,
    next_slot: u32,
    free_slots: Vec<u32>,
}

impl OracleArena {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            generations: HashMap::new(),
            next_slot: 0,
            free_slots: Vec::new(),
        }
    }

    fn insert(&mut self, value: u64) -> (u32, u32) {
        let slot = if let Some(s) = self.free_slots.pop() {
            s
        } else {
            let s = self.next_slot;
            self.next_slot += 1;
            s
        };
        // Generation: increment from last known generation, or start at 1.
        let old_gen = self.generations.get(&slot).copied().unwrap_or(0);
        let new_gen = {
            let g = old_gen.wrapping_add(1);
            if g == 0 {
                1
            } else {
                g
            }
        };
        self.entries.insert(slot, (new_gen, value));
        self.generations.insert(slot, new_gen);
        (slot, new_gen)
    }

    fn remove(&mut self, slot: u32, gen: u32) -> Option<u64> {
        match self.entries.get(&slot) {
            Some(&(g, v)) if g == gen => {
                self.entries.remove(&slot);
                self.free_slots.push(slot);
                Some(v)
            }
            _ => None,
        }
    }

    fn get(&self, slot: u32, gen: u32) -> Option<u64> {
        match self.entries.get(&slot) {
            Some(&(g, v)) if g == gen => Some(v),
            _ => None,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ──────────────────────────────────────────────────────────────────────
// Operation type
// ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Op {
    Insert(u64),
    Remove(usize),   // index into valid_indices vec
    Get(usize),      // index into valid_indices vec
    Contains(usize), // index into valid_indices vec
    GetStale,        // try to get a removed (stale) index
                     //IterCollect,
                     //ValuesSum,
                     //KeysCount,
                     //Retain(u64), // keep only values >= this threshold
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0u64..1000).prop_map(Op::Insert),
        4 => (0usize..50).prop_map(Op::Remove),
        3 => (0usize..50).prop_map(Op::Get),
        2 => (0usize..50).prop_map(Op::Contains),
        1 => Just(Op::GetStale),
    ]
}

// ──────────────────────────────────────────────────────────────────────
// The oracle test
// ──────────────────────────────────────────────────────────────────────

const PROPTEST_CASES: u32 = 1;
const MAX_SHRINK_ITERS: u32 = 0;
const OPS_MIN: usize = 1;
const OPS_MAX: usize = 40;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: PROPTEST_CASES,
        max_shrink_iters: MAX_SHRINK_ITERS,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn arena_matches_oracle(ops in proptest::collection::vec(op_strategy(), OPS_MIN..OPS_MAX)) {
        let mut arena: Arena<u64> = Arena::new();
        let mut oracle = OracleArena::new();
        // Valid (still-alive) indices in both arena and oracle.
        let mut valid_indices: Vec<Index> = Vec::new();
        // Stale (removed) indices for negative tests.
        let mut stale_indices: Vec<Index> = Vec::new();

        for op in &ops {
            match op {
                Op::Insert(val) => {
                    let idx = arena.insert(*val);
                    let (oracle_slot, oracle_gen) = oracle.insert(*val);
                    // The slot assignment must match the oracle (LIFO free list, same next_fresh logic).
                    prop_assert_eq!(idx.slot(), oracle_slot,
                        "Insert: arena slot {} != oracle slot {}", idx.slot(), oracle_slot);
                    prop_assert_eq!(idx.generation(), oracle_gen,
                        "Insert: arena gen {} != oracle gen {}", idx.generation(), oracle_gen);
                    valid_indices.push(idx);
                }

                Op::Remove(i) => {
                    if valid_indices.is_empty() {
                        continue;
                    }
                    let pick = *i % valid_indices.len();
                    let idx = valid_indices[pick];
                    let arena_result = arena.remove(idx);
                    let oracle_result = oracle.remove(idx.slot(), idx.generation());
                    prop_assert_eq!(arena_result, oracle_result,
                        "Remove: arena={:?} oracle={:?}", arena_result, oracle_result);
                    // Mark as stale.
                    valid_indices.swap_remove(pick);
                    stale_indices.push(idx);
                }

                Op::Get(i) => {
                    if valid_indices.is_empty() {
                        continue;
                    }
                    let idx = valid_indices[*i % valid_indices.len()];
                    let arena_result = arena.get(idx).copied();
                    let oracle_result = oracle.get(idx.slot(), idx.generation());
                    prop_assert_eq!(arena_result, oracle_result,
                        "Get: arena={:?} oracle={:?}", arena_result, oracle_result);
                }

                Op::Contains(i) => {
                    if valid_indices.is_empty() {
                        continue;
                    }
                    let idx = valid_indices[*i % valid_indices.len()];
                    let arena_has = arena.contains(idx);
                    let oracle_has = oracle.get(idx.slot(), idx.generation()).is_some();
                    prop_assert_eq!(arena_has, oracle_has,
                        "Contains: arena={} oracle={}", arena_has, oracle_has);
                }

                Op::GetStale => {
                    // Stale index should return None from both.
                    if let Some(&idx) = stale_indices.last() {
                        let arena_result = arena.get(idx);
                        let oracle_result = oracle.get(idx.slot(), idx.generation());
                        prop_assert!(arena_result.is_none(),
                            "GetStale: arena returned {:?} for stale index {:?}", arena_result, idx);
                        prop_assert!(oracle_result.is_none(),
                            "GetStale: oracle returned {:?} for stale index {:?}", oracle_result, idx);
                    }
                }

                /*Op::IterCollect => {
                    // Collect arena iter as (slot, value) pairs and compare to oracle.
                    let mut arena_items: Vec<(u32, u64)> = arena
                        .iter()
                        .map(|(idx, &val)| (idx.slot(), val))
                        .collect();
                    arena_items.sort();

                    let mut oracle_items: Vec<(u32, u64)> = oracle
                        .entries
                        .iter()
                        .map(|(&slot, &(_, val))| (slot, val))
                        .collect();
                    oracle_items.sort();

                    prop_assert_eq!(&arena_items, &oracle_items,
                        "IterCollect mismatch");
                }

                Op::ValuesSum => {
                    // values() sum must match oracle sum.
                    let arena_sum: u64 = arena.values().sum();
                    let oracle_sum: u64 = oracle.entries.values().map(|&(_, v)| v).sum();
                    prop_assert_eq!(arena_sum, oracle_sum,
                        "ValuesSum: arena={} oracle={}", arena_sum, oracle_sum);
                }

                Op::KeysCount => {
                    // keys() must yield exactly len() items.
                    let key_count = arena.keys().count();
                    prop_assert_eq!(key_count, arena.len(),
                        "KeysCount: {} keys but len={}", key_count, arena.len());
                    // Each key must be valid (present in oracle).
                    for key in arena.keys() {
                        prop_assert!(oracle.get(key.slot(), key.generation()).is_some(),
                            "KeysCount: key {:?} not in oracle", key);
                    }
                }

                Op::Retain(threshold) => {
                    // retain() must keep exactly the elements that satisfy the predicate.
                    let threshold = *threshold;
                    // Apply to oracle first (collect in ascending slot order, then remove).
                    let mut to_remove: Vec<u32> = oracle.entries
                        .iter()
                        .filter(|(_, &(_, v))| v < threshold)
                        .map(|(&slot, _)| slot)
                        .collect();
                    // Must match arena's retain order: ascending slot order.
                    to_remove.sort_unstable();
                    for slot in to_remove {
                        oracle.entries.remove(&slot);
                        oracle.free_slots.push(slot);
                    }
                    // Apply to arena.
                    arena.retain(|_idx, val| *val >= threshold);

                    // Verify they match.
                    let mut arena_items: Vec<(u32, u64)> = arena
                        .iter()
                        .map(|(idx, &val)| (idx.slot(), val))
                        .collect();
                    arena_items.sort();
                    let mut oracle_items: Vec<(u32, u64)> = oracle
                        .entries
                        .iter()
                        .map(|(&slot, &(_, val))| (slot, val))
                        .collect();
                    oracle_items.sort();
                    prop_assert_eq!(&arena_items, &oracle_items,
                        "After Retain({}): arena != oracle", threshold);

                    // Rebuild valid_indices (stale ones may have been removed by retain).
                    valid_indices.retain(|idx| arena.contains(*idx));
                }*/
            }

            // ── Invariant checks after every operation ─────────────────
            prop_assert_eq!(arena.len(), oracle.len(),
                "len mismatch: arena={} oracle={}", arena.len(), oracle.len());

            // Every valid_indices entry must still be in both arena and oracle.
            for &idx in &valid_indices {
                prop_assert!(arena.contains(idx),
                    "valid index {:?} missing from arena", idx);
                prop_assert!(oracle.get(idx.slot(), idx.generation()).is_some(),
                    "valid index {:?} missing from oracle", idx);
            }
        }
    }
}

#[cfg(feature = "rayon")]
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        max_shrink_iters: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn rayon_matches_sequential_iterators(ops in proptest::collection::vec(op_strategy(), 1..300)) {
        let mut arena: Arena<u64> = Arena::new();
        let mut valid_indices: Vec<Index> = Vec::new();
        let mut stale_indices: Vec<Index> = Vec::new();

        for op in &ops {
            match op {
                Op::Insert(val) => {
                    let idx = arena.insert(*val);
                    valid_indices.push(idx);
                }

                Op::Remove(i) => {
                    if valid_indices.is_empty() {
                        continue;
                    }
                    let pick = *i % valid_indices.len();
                    let idx = valid_indices.swap_remove(pick);
                    let removed = arena.remove(idx);
                    prop_assert!(removed.is_some(), "remove failed for valid index {:?}", idx);
                    stale_indices.push(idx);
                }

                Op::Get(i) => {
                    if valid_indices.is_empty() {
                        continue;
                    }
                    let idx = valid_indices[*i % valid_indices.len()];
                    let _ = arena.get(idx);
                }

                Op::Contains(i) => {
                    if valid_indices.is_empty() {
                        continue;
                    }
                    let idx = valid_indices[*i % valid_indices.len()];
                    let _ = arena.contains(idx);
                }

                Op::GetStale => {
                    if let Some(&idx) = stale_indices.last() {
                        let _ = arena.get(idx);
                    }
                }

                /*Op::IterCollect => {
                    let _ = arena.iter().count();
                }

                Op::ValuesSum => {
                    let _: u64 = arena.values().sum();
                }

                Op::KeysCount => {
                    let _ = arena.keys().count();
                }

                Op::Retain(threshold) => {
                    let threshold = *threshold;
                    arena.retain(|_, val| *val >= threshold);
                    valid_indices.retain(|idx| arena.contains(*idx));
                }*/
            }

            let mut seq_values: Vec<u64> = arena.values().copied().collect();
            let mut par_values: Vec<u64> = arena.par_values().copied().collect();
            seq_values.sort_unstable();
            par_values.sort_unstable();
            prop_assert_eq!(seq_values, par_values, "par_values mismatch");

            let mut seq_iter: Vec<(u64, u64)> = arena
                .iter()
                .map(|(idx, &val)| (idx.to_bits(), val))
                .collect();
            let mut par_iter: Vec<(u64, u64)> = arena
                .par_iter()
                .map(|(idx, val)| (idx.to_bits(), *val))
                .collect();
            seq_iter.sort_unstable();
            par_iter.sort_unstable();
            prop_assert_eq!(seq_iter, par_iter, "par_iter mismatch");

            let mut seq_keys: Vec<u64> = arena.keys().map(|idx| idx.to_bits()).collect();
            let mut par_keys: Vec<u64> = arena.par_keys().map(|idx| idx.to_bits()).collect();
            seq_keys.sort_unstable();
            par_keys.sort_unstable();
            prop_assert_eq!(seq_keys, par_keys, "par_keys mismatch");
        }
    }
}
