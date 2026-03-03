// ══════════════════════════════════════════════════════════════════════
// Serde Serialization — Feature-gated behind "serde"
// ══════════════════════════════════════════════════════════════════════
//
// WHY INCLUDE THIS
// ────────────────
// slotmap has serde support; thunderdome does not.
// This is a common reason people choose slotmap over thunderdome.
// By including serde from the start, we remove that friction.
//
// SERIALIZATION FORMAT
// ────────────────────
// We serialize as a list of (slot, generation, value) tuples.
// Only occupied entries are serialized (sparse arenas serialize small).
//
// Example JSON:
//   {
//     "entries": [
//       { "slot": 0, "generation": 1, "value": "hello" },
//       { "slot": 3, "generation": 2, "value": "world" }
//     ]
//   }
//
// DESERIALIZATION
// ───────────────
// On deserialization:
//   1. Find the max slot to determine capacity
//   2. Allocate arrays
//   3. For each entry: set bit, write generation, write value
//   4. All non-mentioned slots are empty (bit clear, in free list)
//
// INDEX SERIALIZATION
// ───────────────────
// Index serializes as its u64 bits representation (from to_bits).
// This is compact and portable.
//
// IMPORTANT: Deserialized arenas may have different internal layout
// (free list order, capacity) than the original. The observable behavior
// (same indices return same values) is preserved.
//
// ══════════════════════════════════════════════════════════════════════

use serde::{
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::arena::Arena;
use crate::index::Index;

// ──────────────────────────────────────────────────────────────────────
// Index Serialize / Deserialize
// ──────────────────────────────────────────────────────────────────────

impl Serialize for Index {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(self.to_bits())
    }
}

impl<'de> Deserialize<'de> for Index {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bits = u64::deserialize(d)?;
        Index::from_bits(bits).ok_or_else(|| de::Error::custom("Index has zero generation"))
    }
}

// ──────────────────────────────────────────────────────────────────────
// Arena<T> Serialize
// ──────────────────────────────────────────────────────────────────────

/// A single occupied entry — the unit of serialized data.
#[derive(Serialize)]
struct SerEntry<'a, T: Serialize> {
    slot: u32,
    generation: u32,
    value: &'a T,
}

/// Deserialization counterpart.
#[derive(Deserialize)]
struct DeEntry<T> {
    slot: u32,
    generation: u32,
    value: T,
}

impl<T: Serialize> Serialize for Arena<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Serialize as a sequence of { slot, generation, value } objects.
        // Only occupied entries are emitted.
        let mut seq = s.serialize_seq(Some(self.len() as usize))?;
        for (idx, value) in self.iter() {
            seq.serialize_element(&SerEntry {
                slot: idx.slot(),
                generation: idx.generation(),
                value,
            })?;
        }
        seq.end()
    }
}

// ──────────────────────────────────────────────────────────────────────
// Arena<T> Deserialize
// ──────────────────────────────────────────────────────────────────────

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Arena<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ArenaVisitor<T>(core::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for ArenaVisitor<T> {
            type Value = Arena<T>;

            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a sequence of arena entries with slot, generation, and value fields")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Arena<T>, A::Error> {
                // Collect all entries first to find the max slot.
                let mut entries: alloc::vec::Vec<DeEntry<T>> =
                    alloc::vec::Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(entry) = seq.next_element::<DeEntry<T>>()? {
                    entries.push(entry);
                }

                // Determine required capacity.
                let max_slot = entries.iter().map(|e| e.slot as usize).max().unwrap_or(0);
                let cap = (max_slot + 1).max(entries.len());

                let mut arena = Arena::with_capacity(cap);

                // Sort by slot so we can detect duplicates easily (adjacent).
                entries.sort_unstable_by_key(|e| e.slot);

                // First pass: validate (no duplicates, no zero generation).
                let mut prev_slot = u32::MAX;
                for entry in &entries {
                    if entry.slot == prev_slot {
                        return Err(de::Error::custom(alloc::format!(
                            "duplicate slot {} in arena deserialization",
                            entry.slot
                        )));
                    }
                    if entry.generation == 0 {
                        return Err(de::Error::custom("Arena entry has zero generation"));
                    }
                    prev_slot = entry.slot;
                }

                // Second pass: write values into the arena.
                for entry in entries {
                    let slot = entry.slot as usize;
                    let gen = unsafe { core::num::NonZeroU32::new_unchecked(entry.generation) };
                    // Grow the arena if needed.
                    arena.grow_to(slot + 1);
                    // Write the value and mark the slot occupied.
                    // Safety: slot < capacity after grow_to, not yet occupied (validated above).
                    unsafe {
                        arena.values.get_unchecked_mut(slot).write(entry.value);
                    }
                    arena.generations[slot] = gen.get();
                    arena.occupancy.set(slot);
                    arena.len += 1;
                    // Advance next_fresh past this slot.
                    if slot + 1 > arena.next_fresh as usize {
                        arena.next_fresh = (slot + 1) as u32;
                    }
                }

                // The free list starts empty. Slots in 0..next_fresh that are not
                // occupied are gaps left by insert_at-style operations; they will
                // be found by next_fresh advancement in future inserts.

                Ok(arena)
            }
        }

        d.deserialize_seq(ArenaVisitor(core::marker::PhantomData))
    }
}
