# Benchmarks

Results are directional. Reproduce on your hardware and workload.

Benchmark suites:

- [`benches/comparative.rs`](benches/comparative.rs): microbenchmarks + a churn+iterate “frame” scenario.
- [`benches/realistic_sizes.rs`](benches/realistic_sizes.rs): iteration with 256B and 1KiB inline payloads.

## Takeaways

- The main win is **sparse iteration**: `bitarena` skips empty slots without probing `T`.
- For dense arenas, `.for_each()` can be much faster than a `for` loop (`fold()` fast path).
- With large inline `T`, a dense “light touch” sweep can favor AoS arenas (like `thunderdome`); a sparse sweep still
  strongly favors `bitarena`.

## Reproducing

Commands:

- `cargo +stable bench --bench comparative`
- `cargo +stable bench --bench realistic_sizes`

Notes:

- Settings: LTO + `codegen-units=1` (`profile.bench`), `-C target-cpu=native` (repo [`.cargo/config.toml`](.cargo/config.toml))
- Criterion HTML reports: `target/criterion` (open `report/index.html`)
- Full design writeup: [DESIGN.md](DESIGN.md)

## Example results (Intel i7-1260P, Linux)

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

### Microbenchmarks

Iteration (`for` loop / `next()`):

| Sparsity | bitarena | slotmap | dense_slotmap | thunderdome |
|----------|----------|---------|---------------|-------------|
| 0% (dense) | 29.2 us | 45.2 us | 9.98 us | 52.6 us |
| 50% empty | 44.8 us | 311.0 us | 4.41 us | 289.7 us |
| 90% empty | 5.87 us | 131.9 us | 959 ns | 109.5 us |
| 99% empty | 1.71 us | 34.0 us | 55.4 ns | 42.7 us |
| 99.9% empty | 480 ns | 26.4 us | 6.30 ns | 34.7 us |

Iteration (recommended: `.for_each()` / `fold()`):

| Sparsity | bitarena | dense_slotmap |
|----------|----------|---------------|
| 0% (dense) | **6.31 us** | 22.7 us |
| 50% empty | 31.1 us | **11.3 us** |
| 90% empty | 4.9 us | **2.3 us** |
| 99% empty | 1.5 us | **236 ns** |

Point operations:

Batch of 1,000 operations. Times are per-batch, except Get (per-item).

| Operation | bitarena | slotmap | dense_slotmap | thunderdome |
|-----------|----------|---------|---------------|-------------|
| Get (random, per-item) | **1.42 ns** | 2.22 ns | 1.60 ns | 1.76 ns |
| Insert (1K batch) | 2.22 us | **1.32 us** | 3.39 us | **1.28 us** |
| Remove (1K batch) | 2.80 us | 2.22 us | 4.46 us | **2.55 us** |

### Realistic payload sizes ([`benches/realistic_sizes.rs`](benches/realistic_sizes.rs))

This suite uses fixed-size inline payloads (256B and 1KiB) and compares iteration across different sparsity levels.

It highlights an important nuance:

- Dense + large inline `T` + “light touch” tends to reward AoS layouts (like `thunderdome`).
- As sparsity increases, `bitarena` tends to pull ahead strongly because empty slots don’t touch `T` at all.
- If your loop touches *most* of a large `T`, results tend to become bandwidth-bound and the gap often narrows.
