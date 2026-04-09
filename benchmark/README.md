# infermap benchmark

A 200-case accuracy benchmark for the `infermap` schema mapping engine. Runs on every qualifying PR via `.github/workflows/benchmark.yml` and posts a sticky comment with side-by-side Python and TypeScript scorecards.

See [the design spec](../docs/design/2026-04-08-infermap-accuracy-benchmark-design.md) for the full rationale.

## Run it locally

From the repo root:

```bash
make bench              # both runners + delta vs local baseline
make bench-python       # python only
make bench-ts           # ts only
make bench-self-test    # 5-case smoke test
make bench-test         # unit + fixture + parity tests for both runners
```

## Adding a real-world case

See the labeling checklist in §12 of the implementation plan at
`docs/design/2026-04-08-infermap-accuracy-benchmark-implementation.md`.

TL;DR: source must be on the allowlist (CC0, CC-BY, MIT/BSD/Apache), cap samples at 100 rows per side, label every column in `expected.json` (no "undecided" middle ground), run `python -m infermap_bench rebuild-manifest` after adding.
