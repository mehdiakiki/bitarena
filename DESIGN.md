# bitarena — design notes

This document explains how `bitarena` works and what makes it fast, without requiring you to read the whole codebase.
For benchmark numbers and reproduction steps, see [BENCHMARKS.md](BENCHMARKS.md).

## 30‑second summary

- `bitarena` is a generational arena (`Arena<T>` + `Index`) intended for **fast sweeps over sparse handle tables**.
- It uses a **Struct‑of‑Arrays (SoA)** layout and tracks occupancy in a **packed bitset** (`Vec<u64>`).
- Iteration scans the bitset and only touches `T` for occupied slots, so holes are cheap.
- Insert/remove reuse slots via a LIFO free list, while generations prevent stale handles from aliasing new values.
- Hot iteration prefers `.values()` and iterator adapters like `.for_each()` (they hit a bulk `fold()` fast path).

## The problem

Most generational arenas store each slot as an “entry struct” that contains:

- a tag (“occupied vs free”),
- a generation,
- and either a value `T` or a free‑list pointer.

That layout is easy to implement, but it makes sparse iteration expensive: to figure out whether a slot is occupied, the CPU
has to load (at least part of) the entry for every slot. When the arena is holey, that means lots of wasted memory traffic.

`bitarena` splits “is this slot occupied?” away from the actual `T` storage.

## Layout (SoA + bitset)

Instead of `Vec<Entry<T>>`, `bitarena` keeps separate arrays:

```
occupancy:   [bit, bit, bit, ...]        // 1 bit per slot, packed in Vec<u64>
generations: [u32, u32, u32, ...]        // generation per slot (always initialized)
values:      [MaybeUninit<T>, ...]       // only initialized when occupancy bit is set
free_list:   [slot, slot, slot, ...]     // LIFO stack of free slots (u32)
```

This lets iteration skip empty regions by scanning the occupancy bitset in 64‑slot chunks.

### Index encoding

`Index` matches thunderdome’s 64‑bit encoding for easy migration:

- lower 32 bits: slot (`u32`)
- upper 32 bits: generation (`NonZeroU32`)

This keeps `size_of::<Index>() == 8` and also allows `Option<Index>` to stay 8 bytes.

## Core operations

### `insert`

1. Choose a slot:
   - pop from `free_list` if available (reuse),
   - otherwise use `next_fresh` (grow-on-demand).
2. Write the value into `values[slot]`.
3. Set `occupancy[slot] = 1` and bump `len`.

The important ordering is “write value, then set the bit”: a slot is never considered occupied until its `T` is fully
initialized.

### `remove`

1. Validate the handle (slot in range, occupied, generation matches).
2. Move out the value (via `ptr::read`).
3. Clear `occupancy[slot]`, increment the generation, push slot onto `free_list`, decrement `len`.

This makes old handles immediately invalid, while allowing the slot to be reused quickly.

### `get` / `get_mut`

Fast path is:

1. check occupancy bit,
2. check generation,
3. return reference into `values`.

For sparse tables, (1) and (2) often come from cache lines shared with neighboring slots, without pulling in `T`.

## Iteration design

Iteration is where `bitarena` is meant to win.

### Bitset scanning

The iterator scans `occupancy` word-by-word (`u64`):

- `word == 0`: skip 64 slots without touching `values`.
- `word == u64::MAX`: all 64 slots are occupied (dense chunk).
- otherwise: sparse chunk; find set bits with `trailing_zeros()` and clear them with `word &= word - 1`.

This means holes cost ~“one bitset word read per 64 slots”, not “probe every entry”.

### `next()` vs `fold()`

Rust `for` loops desugar to repeated `next()` calls. Iterator adapters like `.for_each()`, `.sum()`, `.collect()`, etc.
use `fold()` internally.

`bitarena` overrides `fold()` for its iterators so it can process each 64‑slot word in bulk (especially dense words).
That structure often lets LLVM generate tighter code (and sometimes auto-vectorize dense chunks). In hot loops:

- Prefer `arena.values()` if you don’t need keys.
- Prefer iterator adapters (`.for_each(...)`, `.sum()`) over hand-written `for` loops when measuring peak throughput.

## Parallel iteration (`rayon` feature)

With `rayon` enabled, `par_values` / `par_iter` / `par_keys` split the occupancy bitset by contiguous word ranges.
These APIs are **read-only by design** (no parallel mutation), so they stay simple and don’t require internal locking.

## Safety model (unsafe code)

`bitarena` uses `unsafe` for `MaybeUninit<T>` storage and for iterator hot paths. The safety story is based on a small,
explicit set of invariants.

### Invariants

- **I1**: `values[i]` is initialized **iff** `occupancy[i]` is set.
- **I2**: `generations[i]` is always initialized for every allocated slot.
- **I3**: `free_list` contains only slots where `occupancy[i]` is clear.
- **I4**: no duplicates in `free_list`.
- **I5**: `len` equals the number of set bits in `occupancy`.
- **I6**: all internal arrays share the same logical capacity.

Each public operation maintains these invariants by construction (not “best effort”). Many debug builds also include
`debug_assert!` checks for “should never happen” states.

### How we validate it

- Unit tests for API behavior and edge cases.
- A proptest “oracle test” (`tests/oracle_tests.rs`) that compares random operation sequences against a `HashMap`
  reference model.
- Miri (`cargo +nightly miri test`) to catch UB like reading uninitialized memory, double-drop, or invalid aliasing.

## Trade-offs (what you pay for the wins)

- **Not packed**: removing elements leaves holes; iteration may still scan bitset words even when few elements are alive
  (contrast with `DenseSlotMap`, which keeps a packed live array).
- **Extra metadata**: bitset + generations + free list add overhead. In particular, the free list costs ~4 bytes per free
  slot when the arena is mostly empty.
- **More complexity**: SoA + `MaybeUninit` means more unsafe and more invariants to maintain (mitigated by the tests above).

## Practical limits

- Slot index is `u32` (`u32::MAX` slots max).
- Iteration order is slot order (not insertion order) and not stable across churn.

## Build/benchmark notes

Performance measurements are sensitive to build settings. The repo uses:

- `-C target-cpu=native` via [`.cargo/config.toml`](.cargo/config.toml)
- `lto = true` and `codegen-units = 1` in `[profile.bench]`

See [BENCHMARKS.md](BENCHMARKS.md) for commands and more context.

