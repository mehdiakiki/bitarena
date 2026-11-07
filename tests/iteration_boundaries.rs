use bitarena::{Arena, Index};

fn collect_with_fold(iter: impl Iterator<Item = u64>) -> Vec<u64> {
    iter.fold(Vec::new(), |mut values, value| {
        values.push(value);
        values
    })
}

#[test]
fn iterators_cover_dense_and_sparse_word_boundaries() {
    for removed in [&[][..], &[0_usize, 63, 64, 127, 128][..]] {
        let mut arena: Arena<u64> = (0..130).collect();
        let original_keys: Vec<Index> = arena.keys().collect();
        for &slot in removed {
            assert_eq!(arena.remove(original_keys[slot]), Some(slot as u64));
        }
        let mut expected: Vec<u64> = (0..130)
            .filter(|value| !removed.contains(&(*value as usize)))
            .collect();

        assert_eq!(arena.values().copied().collect::<Vec<_>>(), expected);
        assert_eq!(
            arena.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            arena
                .keys()
                .map(|key| key.slot() as u64)
                .collect::<Vec<_>>(),
            expected
        );

        let mut iter = arena.iter();
        assert_eq!(iter.len(), expected.len());
        assert_eq!(
            iter.next().map(|(_, value)| *value),
            expected.first().copied()
        );
        assert_eq!(iter.len(), expected.len() - 1);
        let remaining = collect_with_fold(iter.map(|(_, value)| *value));
        assert_eq!(remaining.as_slice(), &expected[1..]);

        let mut values = arena.values();
        assert_eq!(values.len(), expected.len());
        assert_eq!(values.next().copied(), expected.first().copied());
        assert_eq!(values.len(), expected.len() - 1);
        let remaining = collect_with_fold(values.copied());
        assert_eq!(remaining.as_slice(), &expected[1..]);

        let mut keys = arena.keys();
        assert_eq!(keys.len(), expected.len());
        assert_eq!(
            keys.next().map(|key| key.slot() as u64),
            expected.first().copied()
        );
        assert_eq!(keys.len(), expected.len() - 1);
        let remaining = collect_with_fold(keys.map(|key| key.slot() as u64));
        assert_eq!(remaining.as_slice(), &expected[1..]);

        {
            let mut iter = arena.iter_mut();
            let (_, first) = iter.next().unwrap();
            *first += 10;
            iter.for_each(|(_, value)| *value += 10);
            assert_eq!(*first, expected[0] + 10);
        }
        for value in &mut expected {
            *value += 10;
        }
        assert_eq!(arena.values().copied().collect::<Vec<_>>(), expected);

        {
            let mut values = arena.values_mut();
            let first = values.next().unwrap();
            *first += 100;
            values.for_each(|value| *value += 100);
            assert_eq!(*first, expected[0] + 100);
        }
        for value in &mut expected {
            *value += 100;
        }
        assert_eq!(arena.values().copied().collect::<Vec<_>>(), expected);
    }
}
