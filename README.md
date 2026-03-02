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

        // iterate (recommended: values() + for_each() hits the optimized fold() path)
        let mut sum = 0.0f32;
        pos.values().for_each(|p| sum += p.x);
        std::hint::black_box(sum);
    }
}
```

## Benchmarks (Real-World First)

Measured on Intel i7-1260P, Linux, `--release`, Criterion.rs (100 samples), LTO, `codegen-units=1`,
`target-cpu=native`. Arena size: 100,000 slots. T = `u64`.

Highlights (this machine):

- Game loop frame (10k entities; churn + iterate, ~99% occupancy): with `.for_each()`, bitarena is **24.9 us**,
  beating `dense_slotmap` (26.9 us).
- Sparse iteration: up to **72x faster** than `thunderdome` at 99.9% empty.
- Tip: prefer `values()`/`values_mut()` and `.for_each()`/`.sum()` to hit the optimized `fold()` fast path.

> `dense_slotmap` is included for context: it keeps a compact dense array (great for iteration), but it has different
> trade-offs and API shape than arena-style containers (not a drop-in replacement).

### Simulated Game Loop Frame (10k entities)

Each frame: remove 100, insert 100, iterate all (~99% occupancy).

`for` loop variant (`for (_, &v) in arena.iter()`):

| Container | Frame time |
|----------|------------|
| bitarena | 28.3 us |
| slotmap | 28.0 us |
| dense_slotmap | **25.1 us** |
| thunderdome | 28.6 us |

`.for_each()` variant (recommended: `arena.iter().for_each(...)`):

| Container | Frame time |
|----------|------------|
| bitarena | **24.9 us** |
| dense_slotmap | 26.9 us |

### Iteration (`for` loop / `next()`)

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 29.2 us | 45.2 us | 9.98 us | 52.6 us |
| 50% empty | 44.8 us | 311.0 us | 4.41 us | 289.7 us |
| 90% empty | 5.87 us | 131.9 us | 959 ns | 109.5 us |
| 99% empty | 1.71 us | 34.0 us | 55.4 ns | 42.7 us |
| 99.9% empty | 480 ns | 26.4 us | 6.30 ns | 34.7 us |

### Iteration (recommended: `.for_each()` / `fold()`)

| Sparsity | bitarena | dense_slotmap |
|----------|----------|---------------|
| 0% (dense) | **6.31 us** | 22.7 us |
| 50% empty | 31.1 us | **11.3 us** |
| 90% empty | 4.9 us | **2.3 us** |
| 99% empty | 1.5 us | **236 ns** |

### Point Ops (batch of 1,000 ops)

| Operation | bitarena | slotmap | dense_slotmap | thunderdome |
|-----------|----------|---------|---------------|-------------|
| Get (random, per-item) | **1.42 ns** | 2.22 ns | 1.60 ns | 1.76 ns |
| Insert (1K batch) | 2.22 us | **1.32 us** | 3.39 us | **1.28 us** |
| Remove (1K batch) | 2.80 us | 2.22 us | 4.46 us | **2.55 us** |

Interpretation:

- If your workload iterates sparse arenas heavily, bitarena is usually a strong upgrade over enum-scanning arenas.
- If your workload is dominated by inserts/removes and rarely iterates, benchmark your exact mix.
- If you use bitarena in hot loops, prefer `.for_each()`/`.sum()` to hit the optimized `fold()` fast path.

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
