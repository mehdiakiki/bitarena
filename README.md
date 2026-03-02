# bitarena

[![Crates.io](https://img.shields.io/crates/v/bitarena.svg)](https://crates.io/crates/bitarena)
[![Docs.rs](https://docs.rs/bitarena/badge.svg)](https://docs.rs/bitarena)
[![CI](https://github.com/TODO/bitarena/workflows/CI/badge.svg)](https://github.com/TODO/bitarena/actions)
[![Checked with Miri](https://img.shields.io/badge/miri-checked-green.svg)](https://github.com/rust-lang/miri)

A high-performance generational arena for Rust focused on iteration throughput.

`bitarena` keeps the same generational-handle model developers already use, but replaces entry-by-entry scanning with bitset-driven scanning so sparse iteration is dramatically faster.

## Why use bitarena

- Fast sparse iteration: skip empty regions 64 slots at a time.
- Stable generational indices (`Index`) for safe handle invalidation.
- Drop-in migration path from `thunderdome` for most codebases.
- `no_std` support (with `alloc`).
- Optional `serde` and read-only `rayon` parallel iteration.

## Strong Drop-In Story (thunderdome)

For most projects, migration starts with this diff:

```diff
- use thunderdome::{Arena, Index};
+ use bitarena::{Arena, Index};
```

Compatibility highlights:

- Same mental model: `Arena<T>` + generational `Index`.
- Stale handles become invalid after remove/reuse.
- `Index` size is 8 bytes, `Option<Index>` is 8 bytes.
- `Index::to_bits()` / `from_bits()` matches thunderdome encoding.

## Quick Start

```rust
use bitarena::Arena;

let mut arena = Arena::new();

let a = arena.insert("foo");
let b = arena.insert("bar");

assert_eq!(arena[a], "foo");
assert_eq!(arena[b], "bar");

arena.remove(a);
assert_eq!(arena.get(a), None); // stale index

let c = arena.insert("baz"); // slot reuse with bumped generation
assert_eq!(arena[c], "baz");
```

## Performance Snapshot

Benchmarked on Intel i7-1260P, Linux, `--release`, 100,000 slots.

### Iteration (bitarena vs thunderdome)

| Sparsity | bitarena | thunderdome | Speedup |
|----------|----------|-------------|---------|
| 0% empty | 29.5 us | 50.0 us | 1.7x |
| 50% empty | 29.9 us | 304 us | 10x |
| 90% empty | 6.1 us | 108 us | 17.6x |
| 99% empty | 1.3 us | 49.4 us | 39x |
| 99.9% empty | 435 ns | 38.4 us | 88x |

### Point Ops (random get is very competitive)

| Operation | bitarena | thunderdome |
|-----------|----------|-------------|
| Get (random) | 1.5 ns | 2.0 ns |
| Insert (per item) | 2.4 ns | 1.3 ns |
| Remove (per item) | 3.4 ns | 2.1 ns |

Interpretation:

- If your workload iterates sparse arenas heavily, bitarena is usually a strong upgrade.
- If your workload is dominated by inserts/removes, benchmark your exact mix.

## Parallel Iteration (Rayon)

Enable with:

```toml
bitarena = { version = "0.1", features = ["rayon"] }
```

Read-only parallel APIs:

- `arena.par_values()`
- `arena.par_iter()`
- `arena.par_keys()`

Usage guidance from current measurements:

- `par_values()` is the best default parallel path when keys are not needed.
- `par_iter()` wins on dense/moderately sparse arenas.
- At extreme sparsity, sequential iteration can be faster due to scheduling overhead.

## Feature Flags

```toml
[dependencies]
bitarena = "0.1"                                             # std (default)
bitarena = { version = "0.1", default-features = false }    # no_std + alloc
bitarena = { version = "0.1", features = ["serde"] }      # serialization
bitarena = { version = "0.1", features = ["rayon"] }      # read-only parallel iter
```

## Safety and Validation

Unsafe code is constrained to `MaybeUninit` storage and iterator hot paths, with a documented invariant model.

Validation stack:

- Unit tests for core behavior and edge cases.
- Property tests against a HashMap oracle.
- Miri checks for UB-sensitive paths.

## Documentation

- Design rationale and detailed benchmarks: [DESIGN.md](DESIGN.md)

## License

Licensed under [MIT](LICENSE-MIT).
