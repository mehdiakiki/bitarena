# bitarena

[![Crates.io](https://img.shields.io/crates/v/bitarena.svg)](https://crates.io/crates/bitarena)
[![Docs.rs](https://docs.rs/bitarena/badge.svg)](https://docs.rs/bitarena)
[![CI](https://github.com/mehdiakiki/bitarena/actions/workflows/ci.yml/badge.svg)](https://github.com/mehdiakiki/bitarena/actions/workflows/ci.yml)
[![Checked with Miri](https://img.shields.io/badge/miri-checked-green.svg)](https://github.com/rust-lang/miri)

`bitarena` is a bitset-accelerated generational arena for stable handles, optimized for fast sweeps over sparse tables
(proptest + Miri validated).

- Stable generational handles (`Index`) (stale after `remove`)
- Bitset-accelerated iteration (skips 64 empty slots at a time)
- `no_std + alloc`, optional `serde`, optional `rayon`

## Quick start

```rust
use bitarena::Arena;

let mut arena = Arena::new();

let a = arena.insert("foo");
let b = arena.insert("bar");

assert_eq!(arena.get(a), Some(&"foo"));
assert_eq!(arena[b], "bar");

arena.remove(a);
assert!(arena.get(a).is_none()); // stale index
```

## When it wins

- You keep a long-lived table keyed by stable handles.
- You delete a lot over time (the table gets holey).
- You periodically or frequently sweep the live set (tick/update/expire/GC).

## Not a fit

- Your hot loop is packed iteration (ECS archetypes/chunks, dense `Vec`, `DenseSlotMap`).
- Your hot loop is a dense sweep over large inline `T` but only touches a small part of each object (AoS can win).
- You need insertion-order iteration or stable ordering across churn.
- You need more than `u32::MAX` slots (`Index` stores the slot as `u32`).

## Performance tips

- Prefer `.values().for_each(...)` / `.sum()` when you don’t need keys.
- If `T` is large, split hot/cold data (or store cold payload out-of-line) so the thing you sweep stays small.

## Benchmarks

Full results + reproduction: [BENCHMARKS.md](BENCHMARKS.md).

- `cargo bench --bench comparative`
- `cargo bench --bench realistic_sizes`

## Migration

Coming from `thunderdome`: [MIGRATION.md](MIGRATION.md).

## Safety

Validated with unit tests, a proptest oracle ([tests/oracle_tests.rs](tests/oracle_tests.rs)), and Miri.
Design/invariants: [DESIGN.md](DESIGN.md).

## License

MIT (see [LICENSE](LICENSE)).
