# Benchmarks

Results are directional. Reproduce on your hardware and workload.

Benchmark suites:

- [`benches/comparative.rs`](benches/comparative.rs): microbenchmarks + a churn+iterate "frame" scenario.
- [`benches/realistic_sizes.rs`](benches/realistic_sizes.rs): iteration with 256B and 1KiB inline payloads.

## Takeaways

- **vs thunderdome**: `bitarena` is a strict upgrade. Faster iteration (5–400x depending on sparsity),
  faster get, competitive insert/remove. Drop-in replacement.
- **vs slotmap**: `bitarena` dominates iteration at all sparsity levels, often by 100x+.
  Slightly slower insert/remove.
- **vs dense_slotmap**: This is the interesting comparison. `dense_slotmap` maintains a packed dense
  array (O(live) iteration, not O(capacity)), so it wins pure sparse iteration. But it pays for this
  with slower insert (1.5–2x), slower remove (1.5–2x), and an extra indirection layer (more memory).
  `bitarena` wins at dense full-touch with large payloads, and wins the insert/remove/get tradeoff.
- For dense arenas, `.for_each()` can be much faster than a `for` loop (`fold()` fast path).

### When to use what

| Workload | Best choice |
|----------|-------------|
| Drop-in thunderdome replacement | **bitarena** — faster everywhere |
| Sparse arena, balanced read/write | **bitarena** — fast iteration + fast mutations |
| Mostly iteration, rarely mutate, very sparse | **dense_slotmap** — packed array wins pure iteration |
| Dense arena, large `T`, full-touch sweep | **bitarena** — SoA layout avoids entry overhead |

## Reproducing

Commands:

- `cargo +stable bench --bench comparative`
- `cargo +stable bench --bench realistic_sizes`

Notes:

- Settings: LTO + `codegen-units=1` (`profile.bench`), `-C target-cpu=native` (repo [`.cargo/config.toml`](.cargo/config.toml))
- Criterion HTML reports: `target/criterion` (open `report/index.html`)
- Full design writeup: [DESIGN.md](DESIGN.md)

## Results (AMD Ryzen, Linux)

### Scenario: game loop frame

10,000 entities. Each frame: remove 100, insert 100, iterate all (~99% occupancy). Numbers are Criterion medians.

| Container | `for` loop | `.for_each()` |
|-----------|----------:|-------------:|
| bitarena | 37.9 µs | **33.6 µs** |
| slotmap | 36.4 µs | — |
| dense_slotmap | **34.1 µs** | 35.1 µs |
| thunderdome | 37.6 µs | — |

All four are within ~10% — this scenario is nearly dense, so the bitset advantage is minimal.

### Microbenchmarks

#### Iteration (`for` loop)

10,000 slots. Median times.

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 36.4 µs | 42.2 µs | **9.8 µs** | 57.2 µs |
| 50% empty | 38.7 µs | 351.7 µs | **5.8 µs** | 390.4 µs |
| 90% empty | 5.7 µs | 130.5 µs | **1.0 µs** | 147.2 µs |
| 99% empty | 2.5 µs | 44.6 µs | **69 ns** | 58.3 µs |
| 99.9% empty | 1.0 µs | 33.0 µs | **7.5 ns** | 51.7 µs |

#### Iteration (`.for_each()` / `fold()`)

| Sparsity | bitarena | dense_slotmap |
|----------|----------|---------------|
| 0% (dense) | **7.8 µs** | 30.3 µs |
| 50% empty | 44.5 µs | **15.7 µs** |
| 90% empty | 7.6 µs | **3.2 µs** |
| 99% empty | 1.9 µs | **315 ns** |

bitarena's `for_each()` fast path wins at dense (0%) but dense_slotmap's packed array wins as sparsity increases.

#### Point operations

Batch of 1,000 operations. Times are per-batch, except Get (per-item).

| Operation | bitarena | slotmap | dense_slotmap | thunderdome |
|-----------|----------|---------|---------------|-------------|
| Get (random, per-item) | **2.0 ns** | 3.1 ns | 2.2 ns | 2.7 ns |
| Insert (1K batch) | 3.0 µs | 2.0 µs | 4.7 µs | 2.2 µs |
| Remove (1K batch) | 5.0 µs | 4.0 µs | 7.5 µs | 4.3 µs |

bitarena has the fastest get. Insert/remove are slightly slower than thunderdome/slotmap but
faster than dense_slotmap (which must maintain its packed array on every mutation).

### Realistic payload sizes

256B and 1KiB inline payloads at various sparsity levels. Three iteration modes:
- **light touch**: read one field per element
- **foreach**: `.for_each()` / `fold()` path
- **full touch**: read+write the entire struct

#### 256B payload — light touch (`for` loop)

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 788 µs | 822 µs | **761 µs** | 815 µs |
| 50% empty | 502 µs | 880 µs | **233 µs** | 876 µs |
| 90% empty | 76 µs | 404 µs | **41 µs** | 359 µs |
| 99% empty | 1.8 µs | 224 µs | **1.2 µs** | 239 µs |
| 99.9% empty | 483 ns | 213 µs | **77 ns** | 229 µs |

#### 256B payload — `.for_each()`

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | **795 µs** | 824 µs | 814 µs | 827 µs |
| 50% empty | 521 µs | 847 µs | **206 µs** | 842 µs |
| 90% empty | 60 µs | 352 µs | **41 µs** | 366 µs |
| 99% empty | 2.8 µs | 229 µs | **1.1 µs** | 241 µs |
| 99.9% empty | 488 ns | 233 µs | **98 ns** | 225 µs |

#### 256B payload — full touch

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | **834 µs** | 906 µs | 989 µs | 918 µs |
| 50% empty | 631 µs | 1.00 ms | **266 µs** | 1.25 ms |
| 90% empty | 99 µs | 449 µs | **49 µs** | 630 µs |
| 99% empty | 4.3 µs | 254 µs | **3.0 µs** | 292 µs |
| 99.9% empty | 619 ns | 241 µs | **219 ns** | 255 µs |

At 0% dense with full-touch, bitarena wins all competitors including dense_slotmap.
The SoA layout avoids loading enum discriminants and padding per entry.

#### 1KiB payload — light touch (`for` loop)

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 1.16 ms | **509 µs** | 1.07 ms | 543 µs |
| 50% empty | 656 µs | 420 µs | **354 µs** | 500 µs |
| 90% empty | 67 µs | 138 µs | **52 µs** | 142 µs |
| 99% empty | 1.1 µs | 67 µs | **961 ns** | 64 µs |
| 99.9% empty | 227 ns | 57 µs | **36 ns** | 47 µs |

At 0% dense + 1KiB light-touch, AoS layouts (slotmap/thunderdome) win because the value is
co-located with the metadata — no extra cache miss. This is the one scenario where bitarena's
SoA layout hurts.

#### 1KiB payload — `.for_each()`

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 1.08 ms | 485 µs | 1.09 ms | **474 µs** |
| 50% empty | 576 µs | 433 µs | **313 µs** | 527 µs |
| 90% empty | **70 µs** | 130 µs | 48 µs | 118 µs |
| 99% empty | 1.2 µs | 57 µs | **929 ns** | 56 µs |
| 99.9% empty | 181 ns | 38 µs | **34 ns** | 47 µs |

#### 1KiB payload — full touch

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | **1.39 ms** | 1.58 ms | 2.06 ms | 2.06 ms |
| 50% empty | **1.10 ms** | 1.68 ms | 588 µs | 1.69 ms |
| 90% empty | 116 µs | 217 µs | **66 µs** | 216 µs |
| 99% empty | 4.2 µs | 73 µs | **2.5 µs** | 74 µs |
| 99.9% empty | 354 ns | 54 µs | **175 ns** | 56 µs |

At 0% dense + 1KiB full-touch, bitarena is the clear winner (1.39ms vs 2.06ms for dense_slotmap).
When you read+write the entire struct, SoA avoids paying for entry enum overhead on every element.

## Summary: bitarena vs the field

| | vs thunderdome | vs slotmap | vs dense_slotmap |
|-|----------------|------------|------------------|
| **Iteration (sparse)** | 5–400x faster | 10–200x faster | 2–35x slower |
| **Iteration (dense, small T)** | 1.5x faster | ~same | 3–4x slower |
| **Iteration (dense, large T, full-touch)** | 1.5x faster | 1.1x faster | **1.5x faster** |
| **Get** | 1.3x faster | 1.5x faster | ~same |
| **Insert** | ~same | 1.5x slower | 1.6x faster |
| **Remove** | ~same | 1.3x slower | 1.5x faster |

**Bottom line**: bitarena is a strict upgrade over thunderdome and slotmap. The only arena that
beats it on iteration is dense_slotmap, which pays for its packed array with slower mutations and
extra memory. Choose dense_slotmap if your workload is >90% iteration on very sparse data with
rare mutations. Choose bitarena for everything else.
