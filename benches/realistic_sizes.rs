// ══════════════════════════════════════════════════════════════════════
// Realistic Value Sizes — bitarena vs slotmap (SlotMap & DenseSlotMap)
//                         vs thunderdome
// ══════════════════════════════════════════════════════════════════════
//
// Run with:
//   cargo bench --bench realistic_sizes
//
// Goal:
// - Validate that bitarena still wins when `T` is large (e.g. ECS components)
// - Measure sparse iteration where AoS arenas pay to "peek" every slot
//
// Notes:
// - We use fixed-size, inline payloads (256B and 1KiB) to model common
//   component sizes without introducing extra indirections (like Vec/Box).
// - The hot loop touches multiple cache lines per element to make payload
//   size *visible* to the CPU (not just the first 8 bytes).
// ══════════════════════════════════════════════════════════════════════

use bitarena::Arena as BitArena;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
// Payloads
// ──────────────────────────────────────────────────────────────────────

type Payload256B = [u64; 32]; // 32 * 8 = 256 bytes
type Payload1KiB = [u64; 128]; // 128 * 8 = 1024 bytes

fn make_payload_256b(i: u64) -> Payload256B {
    let mut v = [0u64; 32];
    v[0] = i;
    v[8] = i.wrapping_mul(3);
    v[16] = i.wrapping_mul(5);
    v[24] = i.wrapping_mul(7);
    v[31] = i.wrapping_mul(11);
    v
}

#[inline(always)]
fn touch_payload_256b(v: &Payload256B) -> u64 {
    // Touch all 4 cache lines (assuming 64B lines): indices 0,8,16,24 are on
    // distinct lines; 31 is also on the last line.
    v[0].wrapping_add(v[8])
        .wrapping_add(v[16])
        .wrapping_add(v[24])
        .wrapping_add(v[31])
}

#[inline(always)]
fn touch_payload_256b_full(v: &Payload256B) -> u64 {
    let mut sum = 0u64;
    for &x in v {
        sum = sum.wrapping_add(x);
    }
    sum
}

fn make_payload_1kib(i: u64) -> Payload1KiB {
    let mut v = [0u64; 128];
    v[0] = i;
    v[16] = i.wrapping_mul(3);
    v[32] = i.wrapping_mul(5);
    v[64] = i.wrapping_mul(7);
    v[96] = i.wrapping_mul(11);
    v[127] = i.wrapping_mul(13);
    v
}

#[inline(always)]
fn touch_payload_1kib(v: &Payload1KiB) -> u64 {
    // Touch a spread of cache lines across the 1KiB object.
    v[0].wrapping_add(v[16])
        .wrapping_add(v[32])
        .wrapping_add(v[64])
        .wrapping_add(v[96])
        .wrapping_add(v[127])
}

#[inline(always)]
fn touch_payload_1kib_full(v: &Payload1KiB) -> u64 {
    let mut sum = 0u64;
    for &x in v {
        sum = sum.wrapping_add(x);
    }
    sum
}

// ──────────────────────────────────────────────────────────────────────
// Setup helpers — build an arena at a given sparsity level.
// `sparsity` = fraction of slots that are EMPTY.
// ──────────────────────────────────────────────────────────────────────

fn make_bitarena<T: Copy, F: FnMut(u64) -> T>(
    n: usize,
    sparsity: f64,
    mut make_value: F,
) -> BitArena<T> {
    let mut arena: BitArena<T> = BitArena::with_capacity(n);
    let mut keys: Vec<bitarena::Index> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        keys.push(arena.insert(make_value(i)));
    }

    let to_remove = (n as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..to_remove] {
        arena.remove(k);
    }
    arena
}

fn make_slotmap<T: Copy, F: FnMut(u64) -> T>(
    n: usize,
    sparsity: f64,
    mut make_value: F,
) -> SlotMap<slotmap::DefaultKey, T> {
    let mut sm: SlotMap<slotmap::DefaultKey, T> = SlotMap::with_capacity(n);
    let mut keys: Vec<slotmap::DefaultKey> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        keys.push(sm.insert(make_value(i)));
    }

    let to_remove = (n as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..to_remove] {
        sm.remove(k);
    }
    sm
}

fn make_dense_slotmap<T: Copy, F: FnMut(u64) -> T>(
    n: usize,
    sparsity: f64,
    mut make_value: F,
) -> DenseSlotMap<slotmap::DefaultKey, T> {
    let mut sm: DenseSlotMap<slotmap::DefaultKey, T> = DenseSlotMap::with_capacity(n);
    let mut keys: Vec<slotmap::DefaultKey> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        keys.push(sm.insert(make_value(i)));
    }

    let to_remove = (n as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..to_remove] {
        sm.remove(k);
    }
    sm
}

fn make_thunderdome<T: Copy, F: FnMut(u64) -> T>(
    n: usize,
    sparsity: f64,
    mut make_value: F,
) -> thunderdome::Arena<T> {
    let mut arena: thunderdome::Arena<T> = thunderdome::Arena::with_capacity(n);
    let mut keys: Vec<thunderdome::Index> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        keys.push(arena.insert(make_value(i)));
    }

    let to_remove = (n as f64 * sparsity) as usize;
    let mut rng = Lcg::new(42);
    rng.shuffle(&mut keys);
    for &k in &keys[..to_remove] {
        arena.remove(k);
    }
    arena
}

fn sparsity_label(sparsity: f64) -> String {
    let pct = sparsity * 100.0;
    if pct.fract() == 0.0 {
        format!("{:.0}%_empty", pct)
    } else {
        format!("{:.1}%_empty", pct)
    }
}

// ══════════════════════════════════════════════════════════════════════
// A) ITERATION (256B payload)
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_256b(c: &mut Criterion) {
    // 100k * 256B ≈ 25.6 MiB per arena — big enough to exercise caches,
    // but small enough to run on typical dev machines without swapping.
    const N: usize = 100_000;

    let mut group = c.benchmark_group("iter_256B");
    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = sparsity_label(sparsity);

        {
            let arena = make_bitarena(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_256b(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_slotmap(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_256b(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_dense_slotmap(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_256b(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let arena = make_thunderdome(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("thunderdome", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_256b(v));
                    }
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// A2) ITERATION (256B payload, for_each variant)
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_256b_foreach(c: &mut Criterion) {
    const N: usize = 100_000;

    let mut group = c.benchmark_group("iter_256B_foreach");
    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = sparsity_label(sparsity);

        {
            let arena = make_bitarena(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    a.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_256b(v)));
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_slotmap(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    sm.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_256b(v)));
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_dense_slotmap(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    sm.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_256b(v)));
                    black_box(sum)
                })
            });
        }

        {
            let arena = make_thunderdome(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("thunderdome", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    a.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_256b(v)));
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// A3) ITERATION (256B payload, full-touch variant)
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_256b_full_touch(c: &mut Criterion) {
    const N: usize = 100_000;

    let mut group = c.benchmark_group("iter_256B_full_touch");
    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = sparsity_label(sparsity);

        {
            let arena = make_bitarena(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_256b_full(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_slotmap(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_256b_full(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_dense_slotmap(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_256b_full(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let arena = make_thunderdome(N, sparsity, make_payload_256b);
            group.bench_with_input(BenchmarkId::new("thunderdome", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_256b_full(v));
                    }
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// B) ITERATION (1KiB payload)
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_1kib(c: &mut Criterion) {
    // 30k * 1KiB ≈ 30 MiB per arena — keeps peak RSS reasonable even when
    // running multiple comparative benches in one session.
    const N: usize = 30_000;

    let mut group = c.benchmark_group("iter_1KiB");
    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = sparsity_label(sparsity);

        {
            let arena = make_bitarena(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_slotmap(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_dense_slotmap(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let arena = make_thunderdome(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("thunderdome", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib(v));
                    }
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// B2) ITERATION (1KiB payload, for_each variant)
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_1kib_foreach(c: &mut Criterion) {
    const N: usize = 30_000;

    let mut group = c.benchmark_group("iter_1KiB_foreach");
    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = sparsity_label(sparsity);

        {
            let arena = make_bitarena(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    a.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_1kib(v)));
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_slotmap(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    sm.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_1kib(v)));
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_dense_slotmap(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    sm.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_1kib(v)));
                    black_box(sum)
                })
            });
        }

        {
            let arena = make_thunderdome(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("thunderdome", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    a.iter()
                        .for_each(|(_, v)| sum = sum.wrapping_add(touch_payload_1kib(v)));
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

// ══════════════════════════════════════════════════════════════════════
// B3) ITERATION (1KiB payload, full-touch variant)
// ══════════════════════════════════════════════════════════════════════

fn bench_iteration_1kib_full_touch(c: &mut Criterion) {
    const N: usize = 30_000;

    let mut group = c.benchmark_group("iter_1KiB_full_touch");
    for &sparsity in &[0.0f64, 0.5, 0.9, 0.99, 0.999] {
        let label = sparsity_label(sparsity);

        {
            let arena = make_bitarena(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("bitarena", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib_full(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_slotmap(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib_full(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let sm = make_dense_slotmap(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("dense_slotmap", &label), &sm, |b, sm| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in sm.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib_full(v));
                    }
                    black_box(sum)
                })
            });
        }

        {
            let arena = make_thunderdome(N, sparsity, make_payload_1kib);
            group.bench_with_input(BenchmarkId::new("thunderdome", &label), &arena, |b, a| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, v) in a.iter() {
                        sum = sum.wrapping_add(touch_payload_1kib_full(v));
                    }
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_iteration_256b,
    bench_iteration_256b_foreach,
    bench_iteration_256b_full_touch,
    bench_iteration_1kib,
    bench_iteration_1kib_foreach,
    bench_iteration_1kib_full_touch,
);
criterion_main!(benches);
