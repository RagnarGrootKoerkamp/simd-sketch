# Changelog

## 0.3.0
- Feat: add `sketch` subcommand to write versioned `.ssketch` files (#7).
- Feat: support ambiguous (non-ACTG) IUPAC bases using `--filter-out-n`, where
  ambiguous k-mers are skipped, by using the `PackedNSeq` from the latest
  `packed-seq:4.1.0` and `seq-hash:0.1.0`. (This is ~33% slower than the baseline.)
- Feat: print cleaned up statistics when `RUST_LOG=info` or higher.
- Feat: add `--threads` option.
- Perf: latest `packed-seq:4.1.0` and `seq-hash:0.1.0` give up to 2x high sketch throughput.
- Perf: more `inline(always)`.
- Perf: `lto = true` and `codegen-units = 1` in release profile.
- Perf: improve internal `collect_up_to_bound` by avoiding zero-initialization.
- Perf: more precise starting value for the `factor` of the collect threshold.
- Bugfix: in parallel mode, two parallel doublings would result in x4, even
  though x2 is sufficient.

## 0.2.0
- 'initial' version
