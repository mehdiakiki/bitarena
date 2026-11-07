#![cfg(feature = "serde")]

use bitarena::Arena;

fn roundtrip(arena: &Arena<u32>) -> Arena<u32> {
    let encoded = serde_json::to_vec(arena).unwrap();
    serde_json::from_slice(&encoded).unwrap()
}

#[test]
fn roundtrip_preserves_removed_handle_history() {
    let mut arena = Arena::new();
    let removed = arena.insert(7);
    assert_eq!(arena.remove(removed), Some(7));

    let mut restored = roundtrip(&arena);
    assert!(restored.is_empty());
    let replacement = restored.insert(8);

    assert_ne!(replacement, removed);
    assert_eq!(restored.get(removed), None);
    assert_eq!(restored.get(replacement), Some(&8));
}

#[test]
fn roundtrip_restores_reusable_slots_with_their_generations() {
    let mut arena = Arena::with_capacity(4);
    let original: Vec<_> = (0..4).map(|value| arena.insert(value)).collect();
    for (value, &index) in original[..3].iter().enumerate() {
        assert_eq!(arena.remove(index), Some(value as u32));
    }

    let mut restored = roundtrip(&arena);
    let capacity = restored.capacity();
    let replacements: Vec<_> = (10..13).map(|value| restored.insert(value)).collect();
    let mut slots: Vec<_> = replacements.iter().map(|index| index.slot()).collect();
    slots.sort_unstable();

    assert_eq!(slots, [0, 1, 2]);
    for index in replacements {
        assert_eq!(index.generation(), 2);
    }
    for &index in &original[..3] {
        assert_eq!(restored.get(index), None);
    }
    assert_eq!(restored.get(original[3]), Some(&3));
    assert_eq!(restored.len(), 4);
    assert_eq!(restored.capacity(), capacity);
}
