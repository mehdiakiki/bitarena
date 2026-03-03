// ══════════════════════════════════════════════════════════════════════
// Internal Benchmarks — bitarena in isolation
// ══════════════════════════════════════════════════════════════════════
//
// Run with:
//   cargo bench --bench arena_benchmarks
//
// Focus areas:
// - Bitset scanning throughput (the core primitive)
// - Iterator overhead vs raw bitset scan
// - Effect of T size on iteration speed
// - Growth/reallocation cost
// ══════════════════════════════════════════════════════════════════════

use bitarena::Arena;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ──────────────────────────────────────────────────────────────────────
// Deterministic LCG (identical to comparative.rs — no shared module)
// ──────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// A) T SIZE EFFECT ON ITERATION
//    bitarena separates occupancy from values, so scanning for *which*
//    slots are occupied is always O(n/64) regardless of sizeof(T).
//    The value load is sizeof(T)-dependent but identical to any other
//    contiguous container. Observe that sparse iteration barely changes
//    with T size.
// ══════════════════════════════════════════════════════════════════════

// Generic helper: build a 50%-sparse arena of given value type
fn make_sparse_arena<T: Copy>(n: usize, val: T) -> Arena<T> {
    let mut arena: Arena<T> = Arena::with_capacity(n);
    let mut keys: Vec<bitarena::Index> = (0..n).map(|_| arena.insert(val)).collect();
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..n / 2] {
        arena.remove(k);
    }
    arena
}

fn bench_value_size_effect(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_by_value_size");

    // u8 — 1 byte
    {
        let arena = make_sparse_arena::<u8>(100_000, 42u8);
        group.bench_with_input(
            BenchmarkId::new("bitarena_u8", "50%_sparse"),
            &arena,
            |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in a.iter() {
                        sum = sum.wrapping_add(v as u64);
                    }
                    black_box(sum)
                })
            },
        );
    }

    // u64 — 8 bytes
    {
        let arena = make_sparse_arena::<u64>(100_000, 42u64);
        group.bench_with_input(
            BenchmarkId::new("bitarena_u64", "50%_sparse"),
            &arena,
            |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in a.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum)
                })
            },
        );
    }

    // [u8; 64] — 64 bytes (one cache line)
    {
        let arena = make_sparse_arena::<[u8; 64]>(100_000, [0u8; 64]);
        group.bench_with_input(
            BenchmarkId::new("bitarena_64B", "50%_sparse"),
            &arena,
            |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(v[0] as u64);
                    }
                    black_box(sum)
                })
            },
        );
    }

    // [u8; 256] — 256 bytes (4 cache lines)
    {
        let arena = make_sparse_arena::<[u8; 256]>(100_000, [0u8; 256]);
        group.bench_with_input(
            BenchmarkId::new("bitarena_256B", "50%_sparse"),
            &arena,
            |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(v[0] as u64);
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// B) GROWTH VS PRE-ALLOCATED
//    Measures the cost of dynamic resizing vs with_capacity()
// ══════════════════════════════════════════════════════════════════════

fn bench_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_growth");

    group.bench_function("grow_from_zero", |b| {
        b.iter(|| {
            let mut arena: Arena<u64> = Arena::new();
            for i in 0..10_000u64 {
                black_box(arena.insert(i));
            }
        })
    });

    group.bench_function("pre_allocated", |b| {
        b.iter(|| {
            let mut arena: Arena<u64> = Arena::with_capacity(10_000);
            for i in 0..10_000u64 {
                black_box(arena.insert(i));
            }
        })
    });

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// C) SLOT REUSE — insert after many removals uses free list
// ══════════════════════════════════════════════════════════════════════

fn bench_slot_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("slot_reuse");

    group.bench_function("reuse_freed_slots", |b| {
        b.iter_batched(
            || {
                // Build arena with 10k elements, remove 9k → 9k free slots
                let mut arena: Arena<u64> = Arena::with_capacity(10_000);
                let mut keys: Vec<bitarena::Index> =
                    (0..10_000u64).map(|v| arena.insert(v)).collect();
                let mut rng = Lcg::new(55);
                rng.shuffle(&mut keys);
                for &k in &keys[..9_000] {
                    arena.remove(k);
                }
                arena
            },
            |mut arena| {
                // Now insert 9k more — should reuse free slots
                for i in 0..9_000u64 {
                    black_box(arena.insert(i));
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// D) SPARSITY SCAN SCALING
//    Confirms that bitarena's iteration time drops proportionally as
//    sparsity increases (due to word-skipping in the bitset).
// ══════════════════════════════════════════════════════════════════════

fn bench_sparsity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_sparsity_scaling");

    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = format!("{}%_empty", (sparsity * 100.0) as u32);
        let mut arena: Arena<u64> = Arena::with_capacity(100_000);
        let mut keys: Vec<bitarena::Index> = (0..100_000u64).map(|v| arena.insert(v)).collect();
        let to_remove = (100_000.0 * sparsity) as usize;
        let mut rng = Lcg::new(42);
        rng.shuffle(&mut keys);
        for &k in &keys[..to_remove] {
            arena.remove(k);
        }

        group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
            b.iter(|| {
                let mut sum = 0u64;
                for (_, &v) in a.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_value_size_effect,
    bench_growth,
    bench_slot_reuse,
    bench_sparsity_scaling,
);
criterion_main!(benches);
