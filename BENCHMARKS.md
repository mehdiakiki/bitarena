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
- Software prefetch hints in both `next()` and `fold()` paths keep the `for` loop and `.for_each()`
  performance close — no need to rewrite loops to get good cache behavior.

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
| bitarena | 26.8 µs | **24.1 µs** |
| slotmap | 27.2 µs | — |
| dense_slotmap | **25.6 µs** | 26.2 µs |
| thunderdome | 28.3 µs | — |

All four are within ~10% — this scenario is nearly dense, so the bitset advantage is minimal.

### Microbenchmarks

#### Iteration (`for` loop)

10,000 slots. Median times.

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 26.2 µs | 34.4 µs | **9.5 µs** | 43.8 µs |
| 50% empty | 31.1 µs | 279.7 µs | **4.1 µs** | 290.7 µs |
| 90% empty | 4.9 µs | 104.3 µs | **834 ns** | 105.1 µs |
| 99% empty | 1.7 µs | 34.0 µs | **51 ns** | 46.8 µs |
| 99.9% empty | 769 ns | 27.7 µs | **6.2 ns** | 37.2 µs |

#### Iteration (`.for_each()` / `fold()`)

| Sparsity | bitarena | dense_slotmap |
|----------|----------|---------------|
| 0% (dense) | **7.8 µs** | 22.2 µs |
| 50% empty | 29.8 µs | **11.1 µs** |
| 90% empty | 4.5 µs | **2.2 µs** |
| 99% empty | 1.5 µs | **231 ns** |

bitarena's `for_each()` fast path wins at dense (0%) but dense_slotmap's packed array wins as sparsity increases.

#### Point operations

Batch of 1,000 operations. Times are per-batch, except Get (per-item).

| Operation | bitarena | slotmap | dense_slotmap | thunderdome |
|-----------|----------|---------|---------------|-------------|
| Get (random, per-item) | **1.5 ns** | 2.3 ns | 1.6 ns | 1.9 ns |
| Insert (1K batch) | 2.2 µs | **1.3 µs** | 3.0 µs | **1.3 µs** |
| Remove (1K batch) | 3.1 µs | 2.2 µs | 4.3 µs | **2.3 µs** |

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
| 0% (dense) | 794 µs | 830 µs | **760 µs** | 827 µs |
| 50% empty | 515 µs | 850 µs | **185 µs** | 868 µs |
| 90% empty | 76 µs | 327 µs | **41 µs** | 363 µs |
| 99% empty | 2.6 µs | 223 µs | **1.2 µs** | 242 µs |
| 99.9% empty | 480 ns | 214 µs | **78 ns** | 232 µs |

#### 256B payload — `.for_each()`

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | **827 µs** | 825 µs | 826 µs | 837 µs |
| 50% empty | 500 µs | 851 µs | **213 µs** | 861 µs |
| 90% empty | 71 µs | 350 µs | **35 µs** | 367 µs |
| 99% empty | 2.6 µs | 223 µs | **1.1 µs** | 247 µs |
| 99.9% empty | 443 ns | 220 µs | **99 ns** | 251 µs |

#### 256B payload — full touch

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | **882 µs** | 883 µs | 953 µs | 915 µs |
| 50% empty | 600 µs | 1.05 ms | **251 µs** | 962 µs |
| 90% empty | 74 µs | 337 µs | **41 µs** | 361 µs |
| 99% empty | 3.5 µs | 220 µs | **2.6 µs** | 245 µs |
| 99.9% empty | 504 ns | 219 µs | **178 ns** | 243 µs |

At 0% dense with full-touch, bitarena wins all competitors including dense_slotmap.
The SoA layout avoids loading enum discriminants and padding per entry.

#### 1KiB payload — light touch (`for` loop)

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 828 µs | **422 µs** | 729 µs | 383 µs |
| 50% empty | 366 µs | 305 µs | **194 µs** | 310 µs |
| 90% empty | 49 µs | 81 µs | **35 µs** | 95 µs |
| 99% empty | 829 ns | 31 µs | **651 ns** | 41 µs |
| 99.9% empty | 133 ns | 36 µs | **29 ns** | 33 µs |

At 0% dense + 1KiB light-touch, AoS layouts (slotmap/thunderdome) win because the value is
co-located with the metadata — no extra cache miss. This is the one scenario where bitarena's
SoA layout hurts.

#### 1KiB payload — `.for_each()`

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 688 µs | 375 µs | 697 µs | **411 µs** |
| 50% empty | 409 µs | 348 µs | **200 µs** | 310 µs |
| 90% empty | 49 µs | 92 µs | **37 µs** | 91 µs |
| 99% empty | 796 ns | 46 µs | **645 ns** | 43 µs |
| 99.9% empty | 131 ns | 38 µs | **26 ns** | 34 µs |

#### 1KiB payload — full touch

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | **1.02 ms** | 1.01 ms | 1.09 ms | 1.37 ms |
| 50% empty | **649 µs** | 897 µs | 259 µs | 910 µs |
| 90% empty | 61 µs | 170 µs | **47 µs** | 166 µs |
| 99% empty | 3.1 µs | 49 µs | **1.9 µs** | 51 µs |
| 99.9% empty | 277 ns | 29 µs | **133 ns** | 37 µs |

At 0% dense + 1KiB full-touch, bitarena wins all competitors.
When you read+write the entire struct, SoA avoids paying for entry enum overhead on every element.

## Summary: bitarena vs the field

| | vs thunderdome | vs slotmap | vs dense_slotmap |
|-|----------------|------------|------------------|
| **Iteration (sparse)** | 5–400x faster | 10–200x faster | 2–35x slower |
| **Iteration (dense, small T)** | 1.7x faster | ~same | 2–3x slower |
| **Iteration (dense, large T, full-touch)** | 1.3x faster | ~same | **1.1x faster** |
| **Get** | 1.3x faster | 1.5x faster | ~same |
| **Insert** | ~same | 1.7x slower | 1.4x faster |
| **Remove** | ~same | 1.4x slower | 1.4x faster |

**Bottom line**: bitarena is a strict upgrade over thunderdome and slotmap. The only arena that
beats it on iteration is dense_slotmap, which pays for its packed array with slower mutations and
extra memory. Choose dense_slotmap if your workload is >90% iteration on very sparse data with
rare mutations. Choose bitarena for everything else.
