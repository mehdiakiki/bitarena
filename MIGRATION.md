# Migration from thunderdome

The goal is to be a low-friction upgrade path when your workload is iteration-heavy.

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

Differences / gotchas:

- Iteration order is slot order, not insertion order, and is not stable across churn.
- `no_std` is supported (disable default features); `rayon` is read-only (no parallel mutation).
- `Index` stores the slot as `u32` (max `u32::MAX` slots).

If you hit an incompatibility that blocks a drop-in migration, open an issue with a minimal repro.

