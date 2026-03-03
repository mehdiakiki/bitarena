# ══════════════════════════════════════════════════════════════════════
# Development Commands
# ══════════════════════════════════════════════════════════════════════
#
# WORKFLOW:
#   make check     — Fast feedback loop (compile + test)
#   make miri      — UB checking (run after any unsafe change)
#   make bench     — Performance benchmarks
#   make full      — Full CI pipeline (everything)
#   make release   — Pre-publish checks
#
# ══════════════════════════════════════════════════════════════════════

.PHONY: check test miri bench clippy fmt doc full release clean

# ── Fast Feedback Loop ──────────────────────────────────────────────

check: fmt clippy test
	@echo "✅ All checks passed"

test:
	cargo test
	cargo test --no-default-features  # Verify no_std builds

# ── Safety Verification ─────────────────────────────────────────────

miri:
	@echo "Running Miri (undefined behavior detection)..."
	cargo +nightly miri test
	@echo "✅ Miri found no UB"

# ── Performance ──────────────────────────────────────────────────────

bench:
	cargo bench --bench comparative
	@echo "📊 Results in target/criterion/"

# ── Code Quality ─────────────────────────────────────────────────────

clippy:
	cargo clippy --all-features -- -D warnings

fmt:
	cargo fmt --check

doc:
	cargo doc --all-features --no-deps
	@echo "📖 Docs in target/doc/"

# ── Full Pipeline (CI) ──────────────────────────────────────────────

full: fmt clippy test miri doc
	@echo "✅ Full pipeline passed"

# ── Pre-Publish ──────────────────────────────────────────────────────

release: full
	cargo publish --dry-run
	@echo "🚀 Ready to publish"

clean:
	cargo clean
