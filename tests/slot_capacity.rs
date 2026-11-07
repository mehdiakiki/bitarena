use bitarena::{Arena, Index};

#[test]
fn reserve_uses_unused_allocated_slots() {
    let mut arena = Arena::<u32>::with_capacity(64);

    arena.reserve(1);

    assert_eq!(arena.capacity(), 64);
}

#[test]
fn reserve_counts_removed_and_unused_slots() {
    let mut arena = Arena::with_capacity(8);
    let removed = arena.insert(10_u32);
    arena.insert(20);
    arena.remove(removed);

    arena.reserve(7);

    assert_eq!(arena.capacity(), 8);
    for value in 0..7 {
        arena.insert(value);
    }
    assert_eq!(arena.len(), 8);
    assert_eq!(arena.capacity(), 8);
}

fn assert_skipped_slots_are_reused(mut arena: Arena<u32>, target: Index) {
    let capacity = arena.capacity();
    arena.reserve(3);

    let mut slots = Vec::new();
    for value in 0..3 {
        let index = arena.insert(value);
        slots.push(index.slot());
        assert_eq!(index.generation(), 1);
    }
    slots.sort_unstable();

    assert_eq!(slots, [0, 1, 2]);
    assert_eq!(arena.get(target), Some(&99));
    assert_eq!(arena.len(), 4);
    assert_eq!(arena.capacity(), capacity);
}

#[test]
fn insert_at_slot_preserves_skipped_slots_for_insert() {
    let mut arena = Arena::new();
    let (target, previous) = arena.insert_at_slot(3, 99);
    assert_eq!(previous, None);

    assert_skipped_slots_are_reused(arena, target);
}

#[test]
fn insert_at_preserves_skipped_slots_for_insert() {
    let mut arena = Arena::new();
    let target = Index::from_bits((7_u64 << 32) | 3).unwrap();
    assert_eq!(arena.insert_at(target, 99), None);

    assert_skipped_slots_are_reused(arena, target);
}
