use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

use bitarena::Arena;

struct DropProbe<'a> {
    drops: &'a Cell<usize>,
    panic_once: &'a Cell<bool>,
}

impl Drop for DropProbe<'_> {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
        if self.panic_once.replace(false) {
            panic!("destructor panic");
        }
    }
}

fn probe_arena<'a>(
    count: usize,
    drops: &'a Cell<usize>,
    panic_once: &'a Cell<bool>,
) -> Arena<DropProbe<'a>> {
    (0..count)
        .map(|_| DropProbe { drops, panic_once })
        .collect()
}

fn check_fold_panic(draining: bool, count: usize, skip: usize, panic_at: usize, sparse: bool) {
    let drops = Cell::new(0);
    let panic_once = Cell::new(false);
    let mut arena = probe_arena(count, &drops, &panic_once);
    if sparse {
        arena.retain(|index, _| index.slot() % 3 != 0);
    }
    let visited = Cell::new(0);
    let consume = |_| {
        let position = visited.get();
        visited.set(position + 1);
        if position == panic_at {
            panic!("consumer panic");
        }
    };

    if draining {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut iter = arena.drain();
            for _ in 0..skip {
                drop(iter.next());
            }
            iter.for_each(consume);
        }));
        assert!(result.is_err());
        assert!(arena.is_empty());
    } else {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut iter = arena.into_iter();
            for _ in 0..skip {
                drop(iter.next());
            }
            iter.for_each(consume);
        }));
        assert!(result.is_err());
    }
    assert_eq!(visited.get(), panic_at + 1);
    assert_eq!(drops.get(), count);
}

#[test]
fn into_iter_cleans_up_after_first_word_consumer_panic() {
    check_fold_panic(false, 65, 0, 0, false);
}

#[test]
fn drain_cleans_up_after_first_word_consumer_panic() {
    check_fold_panic(true, 65, 0, 0, false);
}

#[test]
fn into_iter_cleans_up_after_later_dense_word_consumer_panic() {
    check_fold_panic(false, 128, 0, 64, false);
}

#[test]
fn drain_cleans_up_after_later_dense_word_consumer_panic() {
    check_fold_panic(true, 128, 0, 64, false);
}

#[test]
fn into_iter_cleans_up_after_partial_sparse_consumer_panic() {
    check_fold_panic(false, 130, 1, 7, true);
}

#[test]
fn drain_cleans_up_after_partial_sparse_consumer_panic() {
    check_fold_panic(true, 130, 1, 7, true);
}

#[test]
fn dropping_partial_iterators_drops_every_value_once() {
    let drops = Cell::new(0);
    let panic_once = Cell::new(false);
    let arena = probe_arena(65, &drops, &panic_once);
    let mut iter = arena.into_iter();
    drop(iter.next());
    drop(iter);
    assert_eq!(drops.get(), 65);

    let mut arena = probe_arena(65, &drops, &panic_once);
    let mut iter = arena.drain();
    drop(iter.next());
    drop(iter);
    assert!(arena.is_empty());
    assert_eq!(drops.get(), 130);
}

#[test]
fn clear_keeps_old_handles_stale() {
    let mut arena = Arena::new();
    let first = arena.insert(1_u8);
    let second = arena.insert(2_u8);
    let capacity = arena.capacity();
    arena.clear();
    assert!(arena.is_empty());
    assert_eq!(arena.capacity(), capacity);
    arena.extend([3_u8, 4]);
    assert_eq!(arena.get(first), None);
    assert_eq!(arena.get(second), None);
}

#[test]
fn clear_cleans_up_after_a_destructor_panics() {
    let drops = Cell::new(0);
    let panic_once = Cell::new(true);
    let mut arena = probe_arena(3, &drops, &panic_once);
    let result = catch_unwind(AssertUnwindSafe(|| arena.clear()));
    assert!(result.is_err());
    assert!(arena.is_empty());
    assert_eq!(drops.get(), 3);
    drop(arena);
    assert_eq!(drops.get(), 3);
}

#[test]
fn retain_updates_occupancy_before_a_destructor_panics() {
    let drops = Cell::new(0);
    let panic_once = Cell::new(true);
    let mut arena = Arena::new();
    let removed = arena.insert(DropProbe {
        drops: &drops,
        panic_once: &panic_once,
    });
    let retained = arena.insert(DropProbe {
        drops: &drops,
        panic_once: &panic_once,
    });
    let result = catch_unwind(AssertUnwindSafe(|| {
        arena.retain(|index, _| index != removed);
    }));
    assert!(result.is_err());
    assert_eq!(arena.len(), 1);
    assert!(!arena.contains(removed));
    assert!(arena.contains(retained));
    assert_eq!(drops.get(), 1);
    drop(arena);
    assert_eq!(drops.get(), 2);
}

#[test]
fn dropping_an_arena_cleans_up_after_a_destructor_panics() {
    let drops = Cell::new(0);
    let panic_once = Cell::new(true);
    let arena = probe_arena(3, &drops, &panic_once);
    let result = catch_unwind(AssertUnwindSafe(|| drop(arena)));
    assert!(result.is_err());
    assert_eq!(drops.get(), 3);
}

struct CloneProbe<'a> {
    calls: &'a Cell<usize>,
    cloned_drops: &'a Cell<usize>,
    cloned: bool,
}

impl Clone for CloneProbe<'_> {
    fn clone(&self) -> Self {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == 2 {
            panic!("clone panic");
        }
        Self {
            calls: self.calls,
            cloned_drops: self.cloned_drops,
            cloned: true,
        }
    }
}

impl Drop for CloneProbe<'_> {
    fn drop(&mut self) {
        if self.cloned {
            self.cloned_drops.set(self.cloned_drops.get() + 1);
        }
    }
}

#[test]
fn clone_drops_completed_values_when_a_later_clone_panics() {
    let calls = Cell::new(0);
    let cloned_drops = Cell::new(0);
    let mut arena = Arena::new();
    for _ in 0..2 {
        arena.insert(CloneProbe {
            calls: &calls,
            cloned_drops: &cloned_drops,
            cloned: false,
        });
    }
    let result = catch_unwind(AssertUnwindSafe(|| drop(arena.clone())));
    assert!(result.is_err());
    assert_eq!(calls.get(), 2);
    assert_eq!(cloned_drops.get(), 1);
    assert_eq!(arena.len(), 2);
}
