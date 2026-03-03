// ══════════════════════════════════════════════════════════════════════
// Comparative Benchmarks — bitarena vs slotmap (SlotMap & DenseSlotMap)
//                          vs thunderdome
// ══════════════════════════════════════════════════════════════════════
//
// Run with:
//   cargo bench --bench comparative
//
// The primary insight: at high sparsity bitarena's bitset scanner skips
// entire 64-slot words in 2 CPU instructions (tzcnt + blsr), while
// every other arena must inspect each slot individually.
// ══════════════════════════════════════════════════════════════════════

use bitarena::Arena as BitArena;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use slotmap::{DenseSlotMap, SlotMap};

// ──────────────────────────────────────────────────────────────────────
// Tiny deterministic LCG — no extra dependency needed
// ──────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        // Knuth's multiplicative LCG (64-bit)
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Fisher-Yates shuffle
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Setup helpers — build (arena, remaining_valid_indices) at a given
// sparsity level.  sparsity = fraction of slots that are EMPTY.
// ──────────────────────────────────────────────────────────────────────

const N: usize = 100_000;

fn make_bitarena(sparsity: f64) -> (BitArena<u64>, Vec<bitarena::Index>) {
    let mut arena: BitArena<u64> = BitArena::with_capacity(N);
    let mut indices: Vec<bitarena::Index> = (0..N as u64).map(|v| arena.insert(v)).collect();
    let to_remove = (N as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut indices);
    for &idx in &indices[..to_remove] {
        arena.remove(idx);
    }
    let remaining: Vec<bitarena::Index> = indices[to_remove..].to_vec();
    (arena, remaining)
}

fn make_slotmap(sparsity: f64) -> (SlotMap<slotmap::DefaultKey, u64>, Vec<slotmap::DefaultKey>) {
    let mut sm: SlotMap<slotmap::DefaultKey, u64> = SlotMap::with_capacity(N);
    let mut keys: Vec<slotmap::DefaultKey> = (0..N as u64).map(|v| sm.insert(v)).collect();
    let to_remove = (N as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..to_remove] {
        sm.remove(k);
    }
    let remaining: Vec<slotmap::DefaultKey> = keys[to_remove..].to_vec();
    (sm, remaining)
}

fn make_dense_slotmap(
    sparsity: f64,
) -> (
    DenseSlotMap<slotmap::DefaultKey, u64>,
    Vec<slotmap::DefaultKey>,
) {
    let mut sm: DenseSlotMap<slotmap::DefaultKey, u64> = DenseSlotMap::with_capacity(N);
    let mut keys: Vec<slotmap::DefaultKey> = (0..N as u64).map(|v| sm.insert(v)).collect();
    let to_remove = (N as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..to_remove] {
        sm.remove(k);
    }
    let remaining: Vec<slotmap::DefaultKey> = keys[to_remove..].to_vec();
    (sm, remaining)
}

fn make_thunderdome(sparsity: f64) -> (thunderdome::Arena<u64>, Vec<thunderdome::Index>) {
    let mut arena: thunderdome::Arena<u64> = thunderdome::Arena::with_capacity(N);
    let mut indices: Vec<thunderdome::Index> = (0..N as u64).map(|v| arena.insert(v)).collect();
    let to_remove = (N as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut indices);
    for &idx in &indices[..to_remove] {
        arena.remove(idx);
    }
    let remaining: Vec<thunderdome::Index> = indices[to_remove..].to_vec();
    (arena, remaining)
}

// ══════════════════════════════════════════════════════════════════════
// A) ITERATION — primary benchmark
//    Tests sparsity: 0%, 50%, 90%, 99%, 99.9%
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");

    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let pct = sparsity * 100.0;
        let label = if pct.fract() == 0.0 {
            format!("{:.0}%_empty", pct)
        } else {
            format!("{:.1}%_empty", pct)
        };

        // ── bitarena ──────────────────────────────────────────────────
        {
            let (arena, _) = make_bitarena(sparsity);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, arena| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in arena.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum)
                })
            });
        }

        // ── SlotMap ───────────────────────────────────────────────────
        {
            let (sm, _) = make_slotmap(sparsity);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in sm.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum)
                })
            });
        }

        // ── DenseSlotMap ──────────────────────────────────────────────
        {
            let (sm, _) = make_dense_slotmap(sparsity);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in sm.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum)
                })
            });
        }

        // ── thunderdome ───────────────────────────────────────────────
        {
            let (arena, _) = make_thunderdome(sparsity);
            group.bench_with_input(
                BenchmarkId::new("thunderdome", &label),
                &arena,
                |b, arena| {
                    b.iter(|| {
                        let mut sum = 0u64;
                        for (_, &v) in arena.iter() {
                            sum = sum.wrapping_add(v);
                        }
                        black_box(sum)
                    })
                },
            );
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// B) RANDOM ACCESS (get by key)
//    50% sparse arena, iterate through shuffled valid keys.
// ══════════════════════════════════════════════════════════════════════

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_random");

    // ── bitarena ──────────────────────────────────────────────────────
    {
        let (arena, mut keys) = make_bitarena(0.5);
        let mut rng = Lcg::new(99);
        rng.shuffle(&mut keys);
        group.bench_function("bitarena", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let k = keys[i % keys.len()];
                i += 1;
                black_box(arena.get(k))
            })
        });
    }

    // ── SlotMap ───────────────────────────────────────────────────────
    {
        let (sm, mut keys) = make_slotmap(0.5);
        let mut rng = Lcg::new(99);
        rng.shuffle(&mut keys);
        group.bench_function("slotmap", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let k = keys[i % keys.len()];
                i += 1;
                black_box(sm.get(k))
            })
        });
    }

    // ── DenseSlotMap ──────────────────────────────────────────────────
    {
        let (sm, mut keys) = make_dense_slotmap(0.5);
        let mut rng = Lcg::new(99);
        rng.shuffle(&mut keys);
        group.bench_function("dense_slotmap", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let k = keys[i % keys.len()];
                i += 1;
                black_box(sm.get(k))
            })
        });
    }

    // ── thunderdome ───────────────────────────────────────────────────
    {
        let (arena, mut keys) = make_thunderdome(0.5);
        let mut rng = Lcg::new(99);
        rng.shuffle(&mut keys);
        group.bench_function("thunderdome", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let k = keys[i % keys.len()];
                i += 1;
                black_box(arena.get(k))
            })
        });
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// C) INSERT — sequential inserts into empty arena
// ══════════════════════════════════════════════════════════════════════

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    group.bench_function("bitarena", |b| {
        b.iter(|| {
            let mut arena: BitArena<u64> = BitArena::new();
            for i in 0..1_000u64 {
                black_box(arena.insert(i));
            }
        })
    });

    group.bench_function("slotmap", |b| {
        b.iter(|| {
            let mut sm: SlotMap<slotmap::DefaultKey, u64> = SlotMap::new();
            for i in 0..1_000u64 {
                black_box(sm.insert(i));
            }
        })
    });

    group.bench_function("dense_slotmap", |b| {
        b.iter(|| {
            let mut sm: DenseSlotMap<slotmap::DefaultKey, u64> = DenseSlotMap::new();
            for i in 0..1_000u64 {
                black_box(sm.insert(i));
            }
        })
    });

    group.bench_function("thunderdome", |b| {
        b.iter(|| {
            let mut arena: thunderdome::Arena<u64> = thunderdome::Arena::new();
            for i in 0..1_000u64 {
                black_box(arena.insert(i));
            }
        })
    });

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// D) REMOVE — remove random elements from a full arena
// ══════════════════════════════════════════════════════════════════════

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    group.bench_function("bitarena", |b| {
        b.iter_batched(
            || {
                let mut arena: BitArena<u64> = BitArena::with_capacity(1_000);
                let mut keys: Vec<bitarena::Index> =
                    (0..1_000u64).map(|v| arena.insert(v)).collect();
                let mut rng = Lcg::new(7);
                rng.shuffle(&mut keys);
                (arena, keys)
            },
            |(mut arena, keys)| {
                for k in keys {
                    black_box(arena.remove(k));
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("slotmap", |b| {
        b.iter_batched(
            || {
                let mut sm: SlotMap<slotmap::DefaultKey, u64> = SlotMap::with_capacity(1_000);
                let mut keys: Vec<slotmap::DefaultKey> =
                    (0..1_000u64).map(|v| sm.insert(v)).collect();
                let mut rng = Lcg::new(7);
                rng.shuffle(&mut keys);
                (sm, keys)
            },
            |(mut sm, keys)| {
                for k in keys {
                    black_box(sm.remove(k));
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("dense_slotmap", |b| {
        b.iter_batched(
            || {
                let mut sm: DenseSlotMap<slotmap::DefaultKey, u64> =
                    DenseSlotMap::with_capacity(1_000);
                let mut keys: Vec<slotmap::DefaultKey> =
                    (0..1_000u64).map(|v| sm.insert(v)).collect();
                let mut rng = Lcg::new(7);
                rng.shuffle(&mut keys);
                (sm, keys)
            },
            |(mut sm, keys)| {
                for k in keys {
                    black_box(sm.remove(k));
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("thunderdome", |b| {
        b.iter_batched(
            || {
                let mut arena: thunderdome::Arena<u64> = thunderdome::Arena::with_capacity(1_000);
                let mut keys: Vec<thunderdome::Index> =
                    (0..1_000u64).map(|v| arena.insert(v)).collect();
                let mut rng = Lcg::new(7);
                rng.shuffle(&mut keys);
                (arena, keys)
            },
            |(mut arena, keys)| {
                for k in keys {
                    black_box(arena.remove(k));
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// E) MIXED WORKLOAD — simulated game loop
//    10k entities, each "frame": remove 100, insert 100, iterate all.
//    This is the realistic workload where our advantage shines.
// ══════════════════════════════════════════════════════════════════════

fn bench_game_loop(c: &mut Criterion) {
    const ENTITIES: usize = 10_000;
    const CHURN: usize = 100;

    let mut group = c.benchmark_group("game_loop_frame");

    // ── bitarena ──────────────────────────────────────────────────────
    group.bench_function("bitarena", |b| {
        // Set up the arena once outside iter, then benchmark one frame
        let mut arena: BitArena<u64> = BitArena::with_capacity(ENTITIES);
        let mut live: Vec<bitarena::Index> =
            (0..ENTITIES as u64).map(|v| arena.insert(v)).collect();
        let mut rng = Lcg::new(123);
        let mut frame_val = 0u64;

        b.iter(|| {
            // remove CHURN random entities
            rng.shuffle(&mut live);
            let dead: Vec<_> = live.drain(..CHURN).collect();
            for k in dead {
                arena.remove(k);
            }
            // insert CHURN new entities
            for i in 0..CHURN as u64 {
                let k = arena.insert(frame_val.wrapping_add(i));
                live.push(k);
            }
            frame_val = frame_val.wrapping_add(1);
            // iterate all
            let mut sum = 0u64;
            for (_, &v) in arena.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum)
        });
    });

    // ── SlotMap ───────────────────────────────────────────────────────
    group.bench_function("slotmap", |b| {
        let mut sm: SlotMap<slotmap::DefaultKey, u64> = SlotMap::with_capacity(ENTITIES);
        let mut live: Vec<slotmap::DefaultKey> =
            (0..ENTITIES as u64).map(|v| sm.insert(v)).collect();
        let mut rng = Lcg::new(123);
        let mut frame_val = 0u64;

        b.iter(|| {
            rng.shuffle(&mut live);
            let dead: Vec<_> = live.drain(..CHURN).collect();
            for k in dead {
                sm.remove(k);
            }
            for i in 0..CHURN as u64 {
                let k = sm.insert(frame_val.wrapping_add(i));
                live.push(k);
            }
            frame_val = frame_val.wrapping_add(1);
            let mut sum = 0u64;
            for (_, &v) in sm.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum)
        });
    });

    // ── DenseSlotMap ──────────────────────────────────────────────────
    group.bench_function("dense_slotmap", |b| {
        let mut sm: DenseSlotMap<slotmap::DefaultKey, u64> = DenseSlotMap::with_capacity(ENTITIES);
        let mut live: Vec<slotmap::DefaultKey> =
            (0..ENTITIES as u64).map(|v| sm.insert(v)).collect();
        let mut rng = Lcg::new(123);
        let mut frame_val = 0u64;

        b.iter(|| {
            rng.shuffle(&mut live);
            let dead: Vec<_> = live.drain(..CHURN).collect();
            for k in dead {
                sm.remove(k);
            }
            for i in 0..CHURN as u64 {
                let k = sm.insert(frame_val.wrapping_add(i));
                live.push(k);
            }
            frame_val = frame_val.wrapping_add(1);
            let mut sum = 0u64;
            for (_, &v) in sm.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum)
        });
    });

    // ── thunderdome ───────────────────────────────────────────────────
    group.bench_function("thunderdome", |b| {
        let mut arena: thunderdome::Arena<u64> = thunderdome::Arena::with_capacity(ENTITIES);
        let mut live: Vec<thunderdome::Index> =
            (0..ENTITIES as u64).map(|v| arena.insert(v)).collect();
        let mut rng = Lcg::new(123);
        let mut frame_val = 0u64;

        b.iter(|| {
            rng.shuffle(&mut live);
            let dead: Vec<_> = live.drain(..CHURN).collect();
            for k in dead {
                arena.remove(k);
            }
            for i in 0..CHURN as u64 {
                let k = arena.insert(frame_val.wrapping_add(i));
                live.push(k);
            }
            frame_val = frame_val.wrapping_add(1);
            let mut sum = 0u64;
            for (_, &v) in arena.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum)
        });
    });

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// F) MIXED WORKLOAD (for_each variant)
//    Same as bench_game_loop but uses arena.iter().for_each(...)
//    which triggers the fold() override and enables AVX2 vectorization.
//
//    This is the RECOMMENDED iteration pattern for maximum throughput.
//    The standard `for (_, &v) in arena.iter()` desugars to next() calls
//    which cannot be vectorized; `.for_each(|...|)` calls fold() which
//    LLVM can auto-vectorize with VPADDQ (4 × u64 per cycle on AVX2).
// ══════════════════════════════════════════════════════════════════════

fn bench_game_loop_foreach(c: &mut Criterion) {
    const ENTITIES: usize = 10_000;
    const CHURN: usize = 100;

    let mut group = c.benchmark_group("game_loop_frame_foreach");

    // ── bitarena (for_each) ───────────────────────────────────────────
    group.bench_function("bitarena", |b| {
        let mut arena: BitArena<u64> = BitArena::with_capacity(ENTITIES);
        let mut live: Vec<bitarena::Index> =
            (0..ENTITIES as u64).map(|v| arena.insert(v)).collect();
        let mut rng = Lcg::new(123);
        let mut frame_val = 0u64;

        b.iter(|| {
            rng.shuffle(&mut live);
            let dead: Vec<_> = live.drain(..CHURN).collect();
            for k in dead {
                arena.remove(k);
            }
            for i in 0..CHURN as u64 {
                let k = arena.insert(frame_val.wrapping_add(i));
                live.push(k);
            }
            frame_val = frame_val.wrapping_add(1);
            // for_each triggers our fold() override → AVX2 VPADDQ vectorization
            let mut sum = 0u64;
            arena.iter().for_each(|(_, v)| sum = sum.wrapping_add(*v));
            black_box(sum)
        });
    });

    // ── DenseSlotMap (for_each) ───────────────────────────────────────
    group.bench_function("dense_slotmap", |b| {
        let mut sm: DenseSlotMap<slotmap::DefaultKey, u64> = DenseSlotMap::with_capacity(ENTITIES);
        let mut live: Vec<slotmap::DefaultKey> =
            (0..ENTITIES as u64).map(|v| sm.insert(v)).collect();
        let mut rng = Lcg::new(123);
        let mut frame_val = 0u64;

        b.iter(|| {
            rng.shuffle(&mut live);
            let dead: Vec<_> = live.drain(..CHURN).collect();
            for k in dead {
                sm.remove(k);
            }
            for i in 0..CHURN as u64 {
                let k = sm.insert(frame_val.wrapping_add(i));
                live.push(k);
            }
            frame_val = frame_val.wrapping_add(1);
            let mut sum = 0u64;
            sm.iter().for_each(|(_, v)| sum = sum.wrapping_add(*v));
            black_box(sum)
        });
    });

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// G) PURE ITERATION (for_each variant)
//    Demonstrates AVX2-vectorized path via fold() override.
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_foreach(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_foreach");

    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99] {
        let pct = sparsity * 100.0;
        let label = format!("{:.0}%_empty", pct);

        // ── bitarena ──────────────────────────────────────────────────
        {
            let (arena, _) = make_bitarena(sparsity);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, arena| {
                b.iter(|| {
                    let mut sum = 0u64;
                    arena.iter().for_each(|(_, v)| sum = sum.wrapping_add(*v));
                    black_box(sum)
                })
            });
        }

        // ── DenseSlotMap ──────────────────────────────────────────────
        {
            let (sm, _) = make_dense_slotmap(sparsity);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    sm.iter().for_each(|(_, v)| sum = sum.wrapping_add(*v));
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// H) RAYON ITERATION (feature-gated)
//    Compare sequential iter() and parallel par_iter() at key sparsities.
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
fn bench_rayon_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_rayon");

    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99] {
        let pct = sparsity * 100.0;
        let label = format!("{:.0}%_empty", pct);
        let (arena, _) = make_bitarena(sparsity);

        group.bench_with_input(
            BenchmarkId::new("bitarena_seq", &label),
            &arena,
            |b, arena| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in arena.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bitarena_par", &label),
            &arena,
            |b, arena| {
                b.iter(|| {
                    let sum = arena
                        .par_iter()
                        .map(|(_, v)| *v)
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "rayon"))]
fn bench_rayon_iteration(_: &mut Criterion) {}

// ══════════════════════════════════════════════════════════════════════
// I) RAYON VALUES (feature-gated)
//    Compare values() and par_values() on large values.
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
fn bench_rayon_values_large(c: &mut Criterion) {
    const M: usize = 100_000;
    let mut arena: BitArena<[u8; 256]> = BitArena::with_capacity(M);
    let mut keys: Vec<bitarena::Index> = Vec::with_capacity(M);

    for i in 0..M {
        let mut payload = [0u8; 256];
        payload[0] = i as u8;
        keys.push(arena.insert(payload));
    }

    let mut rng = Lcg::new(4242);
    rng.shuffle(&mut keys);
    for &k in &keys[..M / 2] {
        arena.remove(k);
    }

    let mut group = c.benchmark_group("values_rayon_large");
    group.bench_function("bitarena_values_seq_50%_sparse_256B", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for v in arena.values() {
                sum = sum.wrapping_add(v[0] as u64);
            }
            black_box(sum)
        })
    });

    group.bench_function("bitarena_values_par_50%_sparse_256B", |b| {
        b.iter(|| {
            let sum = arena
                .par_values()
                .map(|v| v[0] as u64)
                .reduce(|| 0u64, |a, b| a.wrapping_add(b));
            black_box(sum)
        })
    });
    group.finish();
}

#[cfg(not(feature = "rayon"))]
fn bench_rayon_values_large(_: &mut Criterion) {}

criterion_group!(
    benches,
    bench_iteration,
    bench_get,
    bench_insert,
    bench_remove,
    bench_game_loop,
    bench_game_loop_foreach,
    bench_iteration_foreach,
    bench_rayon_iteration,
    bench_rayon_values_large,
);
criterion_main!(benches);
