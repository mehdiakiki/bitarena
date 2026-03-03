# bitarena

[![Crates.io](https://img.shields.io/crates/v/bitarena.svg)](https://crates.io/crates/bitarena)
[![Docs.rs](https://docs.rs/bitarena/badge.svg)](https://docs.rs/bitarena)
[![CI](https://github.com/mehdiakiki/bitarena/actions/workflows/ci.yml/badge.svg)](https://github.com/mehdiakiki/bitarena/actions/workflows/ci.yml)
[![Checked with Miri](https://img.shields.io/badge/miri-checked-green.svg)](https://github.com/rust-lang/miri)

`bitarena` is a generational arena (`Arena<T>` + `Index`) for Rust.

It's designed for workloads where you keep a large arena around, do churn (insert/remove), and iterate the live set a lot
(games/ECS, simulations, schedulers, caches).

The core trick is simple: track occupancy in a packed bitset and scan that bitset during iteration, so empty slots are
skipped 64 at a time.

> **TL;DR**
> - Use `bitarena` when iteration is hot, especially when the arena is sparse.
> - If you mostly do inserts/removes and rarely iterate, you may not see a win.
>
> **Not a fit if...**
> - You need insertion-order iteration or stable ordering across churn.
> - You want packed "only-live-items" storage (consider `DenseSlotMap` or a dense `Vec`).
> - You need more than `u32::MAX` slots (`Index` stores the slot as `u32`).

## Contents

- [Quick start](#quick-start)
- [When to use](#when-to-use)
- [Semantics](#semantics)
- [Benchmarks](#benchmarks)
- [Migration from thunderdome](#migration-from-thunderdome)
- [Features](#features)
- [Safety](#safety)
- [Contributing](#contributing)
- [License](#license)

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

> **Performance tip**
> In hot loops, prefer `.values().for_each(...)` / `.sum()`.
> A `for` loop desugars to repeated `next()` calls; `.for_each()` goes through `fold()`, and `bitarena` overrides
> `fold()` to process words in bulk (often enabling SIMD on dense words).

## When to use

`bitarena` is a good fit when:

- You want stable handles (`Index`) and you're OK with them going stale after `remove`.
- You iterate "all live entities" frequently (every frame/tick).
- Your arena can get sparse (many holes) and you still want iteration to be fast.

You might prefer a different container when:

- You primarily care about iterating packed live elements: `DenseSlotMap`, a dense `Vec`, or an ECS archetype layout.
- You rarely iterate and are dominated by insert/remove throughput: benchmark your exact mix.

## Semantics

Iteration order:

- Iteration is in ascending slot order (not insertion order).
- Removes/inserts can reuse slots, so the order of live elements is not stable across churn.

Handle semantics (`Index`):

- An `Index` becomes invalid as soon as its slot is removed.
- If that slot is later reused, the generation is incremented, so stale indices stop matching.
- Generations are per-slot `u32` and wrap from `u32::MAX` to `1` (practically unreachable).

Complexity and overhead (rough):

- `insert` / `remove` / `get` / `contains`: O(1) amortized.
- Iteration cost: O(bitset_words_scanned + live_elements).
- Per-slot overhead: 1 bit occupancy + 4 bytes generation + `MaybeUninit<T>` storage, plus free-list bookkeeping for
  empty slots (up to 4 bytes/slot when fully empty).

## Benchmarks

Results on my machine (Intel i7-1260P, Linux). Treat them as directional and reproduce on your workload.

Takeaways:

- Sparse iteration is the main win: at 99.9% empty, `bitarena` is ~72x faster than `thunderdome` here (480 ns vs 34.7 us).
- Dense iteration (within `bitarena`): `.for_each()` is ~4.6x faster than a `for` loop at 0% empty (6.31 us vs 29.2 us).
- In a churn + iterate "frame" benchmark, `.for_each()` is ~1.08x faster than `DenseSlotMap` here (24.9 us vs 26.9 us).

### Scenario: game loop frame

10,000 entities. Each frame: remove 100, insert 100, iterate all (~99% occupancy). Numbers are Criterion medians.

`for` loop (`for (_, &v) in arena.iter()`):

| Container | Median |
|----------|-------:|
| bitarena | 28.3 us |
| slotmap | 28.0 us |
| dense_slotmap | **25.1 us** |
| thunderdome | 28.6 us |

`.for_each()` (`arena.iter().for_each(...)`):

| Container | Median |
|----------|-------:|
| bitarena | **24.9 us** |
| dense_slotmap | 26.9 us |

<details>
<summary>Microbenchmarks</summary>

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

### Point operations

Batch of 1,000 operations. Times are per-batch, except Get (per-item).

| Operation | bitarena | slotmap | dense_slotmap | thunderdome |
|-----------|----------|---------|---------------|-------------|
| Get (random, per-item) | **1.42 ns** | 2.22 ns | 1.60 ns | 1.76 ns |
| Insert (1K batch) | 2.22 us | **1.32 us** | 3.39 us | **1.28 us** |
| Remove (1K batch) | 2.80 us | 2.22 us | 4.46 us | **2.55 us** |

</details>

### Reproducing

- Benchmark code: `benches/comparative.rs`
- Command: `cargo +stable bench --bench comparative`
- Rust: 1.93.0 (stable). MSRV: 1.70.0.
- Compared crates: `thunderdome` 0.6, `slotmap` 1.0 (`DenseSlotMap` comes from `slotmap`)
- Settings: LTO + `codegen-units=1` (`profile.bench`), `-C target-cpu=native` (repo `.cargo/config.toml`)
- Criterion HTML reports: `target/criterion` (open `report/index.html`)
- Full writeup: [DESIGN.md](DESIGN.md)

> Note: `dense_slotmap` is shown for context. It keeps a packed array of live values, so it often wins at high sparsity by
> iterating fewer elements. It's a different trade-off (values move on removal, different API surface).

## Migration from thunderdome

The goal is to be a low-friction upgrade path if your workload is iteration-heavy.

Typical first diff:

```diff
- use thunderdome::{Arena, Index};
+ use bitarena::{Arena, Index};
```

Compatibility checklist (common surface area):

- [x] `Arena<T>` + generational `Index`
- [x] `arena.insert`, `arena.remove`, `arena.get`, `arena.get_mut`, `arena.contains`, `arena.len`
- [x] Indexing: `arena[idx]` / `arena[idx] = ...` (panics on invalid index)
- [x] `Index` is 8 bytes; `Option<Index>` is 8 bytes
- [x] `Index::to_bits()` / `Index::from_bits()` matches thunderdome's encoding

Differences / gotchas to watch for:

- Iteration order is slot order, not insertion order, and is not stable across churn.
- `no_std` is supported (disable default features); `rayon` is read-only (no parallel mutation).

If you hit an incompatibility that blocks a drop-in migration, open an issue with a minimal repro.

## Features

Feature matrix:

| Capability | How | Notes |
|-----------|-----|-------|
| `std` | default | Default build. |
| `no_std` + `alloc` | `default-features = false` | Still uses `alloc` types like `Vec`. |
| `serde` | `features = ["serde"]` | `Index` as `u64` bits; `Arena<T>` as occupied entries. Requires `alloc`. |
| `rayon` | `features = ["rayon"]` | Read-only parallel iteration: `par_values`, `par_iter`, `par_keys`. |

MSRV:

- MSRV: Rust 1.70.0 (CI checks it). This may be bumped in minor releases.

Threading:

- `Arena<T>` is `Send`/`Sync` when `T` is `Send`/`Sync` (no internal locking).
- Rayon APIs are read-only by design.

## Safety

Unsafe code is used for `MaybeUninit` storage and iterator hot paths. The design relies on a small set of invariants:

- `values[i]` is initialized iff the occupancy bit for `i` is set.
- `free_list` contains only empty slots (no duplicates).
- `len` stays in sync with the bitset popcount.

Verification:

- [x] Unit tests
- [x] Proptest oracle vs `HashMap` (`tests/oracle_tests.rs`)
- [x] Miri (`cargo +nightly miri test`)

Safety argument and full invariant set: [DESIGN.md](DESIGN.md).

## Contributing

- Tests: `cargo test` (also try `cargo test --no-default-features` and `cargo test --all-features`)
- Benchmarks: `cargo +stable bench --bench comparative`
- PRs/issues welcome. Performance regressions should include a benchmark diff.

## License

Licensed under MIT.
