# bitarena

[![Crates.io](https://img.shields.io/crates/v/bitarena.svg)](https://crates.io/crates/bitarena)
[![Docs.rs](https://docs.rs/bitarena/badge.svg)](https://docs.rs/bitarena)
[![CI](https://github.com/mehdiakiki/bitarena/actions/workflows/ci.yml/badge.svg)](https://github.com/mehdiakiki/bitarena/actions/workflows/ci.yml)
[![Checked with Miri](https://img.shields.io/badge/miri-checked-green.svg)](https://github.com/rust-lang/miri)

A high-performance generational arena for Rust optimized for iteration-heavy workloads.

`bitarena` keeps the familiar handle-based API (`Arena<T>` + generational `Index`) but makes iteration fast by scanning a packed occupancy bitset instead of touching every slot.

## Why use bitarena

- Fast sparse iteration: skip empty regions 64 slots at a time (bitset word scan).
- Stable generational indices: remove/reuse invalidates stale handles.
- Great fit for game/ECS-like loops: churn (spawn/despawn) + iterate every frame.
- Drop-in migration path from `thunderdome` for most codebases.
- `no_std` support (with `alloc`), plus optional `serde` and read-only `rayon`.

## Real-World Scenario: Game Loop Frame

This is the shape of workload `bitarena` is built for: stable handles, frequent churn, and frequent iteration.

```rust
use bitarena::Arena;

#[derive(Clone, Copy)]
struct Position { x: f32, y: f32 }

fn main() {
    let mut pos: Arena<Position> = Arena::with_capacity(10_000);
    let mut live = Vec::new();

    // Spawn
    for i in 0..10_000u32 {
        live.push(pos.insert(Position { x: i as f32, y: 0.0 }));
    }

    // Frame: despawn some, spawn some, iterate all
    for frame in 0..60u32 {
        // churn: remove 100
        for i in 0..100usize {
            let idx = live.swap_remove(i);
            pos.remove(idx);
        }
        // churn: insert 100
        for i in 0..100u32 {
            live.push(pos.insert(Position { x: (frame * 100 + i) as f32, y: 0.0 }));
        }

        // iterate
        let mut sum = 0.0f32;
        for (_, p) in pos.iter() {
            sum += p.x;
        }
        std::hint::black_box(sum);
    }
}
```

## Benchmarks (How It Compares)

Measured on Intel i7-1260P, Linux, `--release`. Arena size: 100,000 slots.

> `dense_slotmap` is included for context: it wins iteration by keeping a compact dense array, but has different trade-offs (not a drop-in replacement for generational arenas).

### Iteration (same API shape, different sparsity)

| Sparsity | bitarena | thunderdome | slotmap | dense_slotmap |
|----------|----------|-------------|---------|---------------|
| 0% empty | 29.5 us | 50.0 us | 43.1 us | 6.2 us |
| 50% empty | 29.9 us | 304 us | 303 us | 2.8 us |
| 90% empty | 6.1 us | 108 us | 104 us | 0.78 us |
| 99% empty | 1.3 us | 49.4 us | 39.6 us | 0.07 us |
| 99.9% empty | 435 ns | 38.4 us | 29.1 us | 6.9 ns |

### Simulated Game Loop Frame (10k entities)

Each frame: remove 100, insert 100, iterate all.

| Container | Frame time |
|----------|------------|
| bitarena | 27.6 us |
| thunderdome | 29.7 us |
| slotmap | 28.8 us |
| dense_slotmap | 25.6 us |

### Point Ops (random get is competitive)

| Operation | bitarena | thunderdome |
|-----------|----------|-------------|
| Get (random) | 1.5 ns | 2.0 ns |
| Insert (per item) | 2.4 ns | 1.3 ns |
| Remove (per item) | 3.4 ns | 2.1 ns |

Interpretation:

- If your workload iterates sparse arenas heavily, bitarena is usually a strong upgrade over enum-scanning arenas.
- If your workload is dominated by inserts/removes and rarely iterates, benchmark your exact mix.

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

Licensed under MIT OR Apache-2.0.
