# SimdSketch

[![crates.io](https://img.shields.io/crates/v/simd-sketch.svg)](https://crates.io/crates/simd-sketch)
[![docs.rs](https://img.shields.io/docsrs/simd-sketch.svg)](https://docs.rs/simd-sketch)

A SIMD-accelerated library to compute two types of sketches:
- Classic bottom $s$ sketch, containing the $s$ smallest distinct k-mer hashes.
- Bucket sketch, which partitions the hashes into $s$ parts and returns the smallest
  hash in each part. (Introduced as *one permutation hashing* in Li, Owen, Zhang 2012.)

See the corresponding [blog post](https://curiouscoding.nl/posts/simd-sketch/)
for background and evaluation.

Sketching takes 2 seconds for a 3Gbp human genome. This library returns 32-bit `u32`
hashes. This means that currently it may not be very suitable for sequences that are
too close to 1Gbp in length, since the bottom hash values will be relatively dense.

**Algorithm.**
For the bottom $s$ sketch, we first collect all ``sufficiently small'' hashes
into a vector. Then, that vector is sorted and deduplicated, and the smallest
$s$ values are returned. This ensures that the runtime is $O(n + s \log s)$ when
the number of duplicate k-mers is limited.

For the bucket sketch, the classic method is to partition hashes linearly, e.g.,
for $s=2$ into the bottom and top half. Then, a single value is kept per part,
and each hash is compared against the rolling minimum of its bucket.

Instead, here we make buckets by the remainder modulo $s$. This way, we can
again pre-filter for ``sufficiently small'' values, and then only scan those for
the minimum.

In both variants, we double the ``smallness'' threshold until either $s$
distinct values are found or all $s$ buckets have a value in them.

**Formulas**
For the bottom sketch, the **Jaccard similarity** `j` is computed as follows:
1. Find the smallest `s` distinct k-mer hashes in the union of two sketches.
2. Return the fraction of these k-mers that occurs in both sketches.

For the bucket sketch, we first identify all buckets that are not left empty by
both sketches. Then, we take the fraction `j0` of the remaining buckets where they
are equal. We use **b-bit sketches**, where only the bottom `b` bits of each
bucket-minimum are stored. This gives a `1/2^b` probability of hash collisions.
To fix this, we compute `j = (j0 - 1/2^b) / (1 - 1/2^b)` as the similarity
corrected for these collisions.

The **Mash distance** as reported by the CLI is computed as
`-ln(2*j / (1+j))/k`.
 This is always `>=0`, but can be as large as `inf` for disjoint sets, that have `j=0`.

**Implementation notes.**
Good performance is mostly achieved by using a branch-free implementation: all
hashes are computed using 8 parallel streams using SIMD, and appended to a vector when they
are sufficiently small to likely be part of the sketch.

The underlying streaming and hashing algorithms are described in the following [preprint](https://doi.org/10.1101/2025.01.27.634998):

- SimdMinimizers: Computing random minimizers, fast.
  Ragnar Groot Koerkamp, Igor Martayan
  bioRxiv 2025.01.27 [doi.org/10.1101/2025.01.27.634998](https://doi.org/10.1101/2025.01.27.634998)


## Usage
Please see [lib.rs](src/lib.rs) and [docs.rs](https://docs.rs/simd-sketch) for
full docs.

```rust
use packed_seq::SeqVec;

// Bottom h=10000 sketch of k=31-mers.
let k = 31;
let h = 10_000;

// Use `new_rc` for a canonical version instead.
let sketch = simd_sketch::Sketcher::new(k, h);

// Generate two random sequences of 2M characters.
let n = 2_000_000;
let seq1 = packed_seq::PackedSeqVec::random(n);
let seq2 = packed_seq::PackedSeqVec::random(n);

let sketch1: simd_sketch::BottomSketch = sketcher.bottom_sketch(seq1.as_slice());
let sketch2: simd_sketch::BottomSketch = sketcher.bottom_sketch(seq2.as_slice());

// Value between 0 and 1, estimating the fraction of shared k-mers.
let similarity = sketch1.similarity(&sketch2);

// Bucket sketch variant

let sketch1: simd_sketch::BinSketch = sketcher.sketch(seq1.as_slice());
let sketch2: simd_sketch::BinSketch = sketcher.sketch(seq2.as_slice());

// Value between 0 and 1, estimating the fraction of shared k-mers.
let similarity = sketch1.similarity(&sketch2);
```

## Command line tool

The two main subcommands are `sketch` and `triangle`.
The is also `dist` that simply computes the distance between two sequences or sketches.

### `sketch`

`simd-sketch sketch` takes a list of (compressed) fasta files as input, and
writes their sketches as `<file>.ssketch` files.

For example,

```sh
simd-sketch sketch inputs/*.fa
```

writes corresponding sketches to `inputs/*.fa.ssketch`.

See below for an explanation of the sketch parameters.

### `triangle`

Compute an all-to-all Mash distance matrix between (already sketched) fasta files.
With `--save-sketches`, missing intermediate sketches are saved to disk.

```
> simd-sketch triangle --help
Takes paths to fasta files, and outputs a Phylip distance matrix to stdout

Usage: simd-sketch triangle [OPTIONS] [PATHS]...

Arguments:
  [PATHS]...  Paths to (directories of) (gzipped) fasta files

Options:
      --alg <ALG>        Sketch algorithm to use. Defaults to bucket because of its much faster comparisons [default: bucket] [possible values: bottom, bucket]
      --fwd              When set, use forward instead of canonical k-mer hashes
  -k <K>                 k-mer size [default: 31]
  -s <S>                 Bottom-s sketch, or number of buckets [default: 10000]
  -b <B>                 For bucket-sketch, store only the lower b bits [default: 8]
      --output <OUTPUT>  Write phylip distance matrix here, or default to stdout
      --save-sketches    Save missing sketches to disk, as .ssketch files alongside the input
  -h, --help             Print help
```

Minimal example usage, printing the matrix to stdout:

```sh
simd-sketch triangle inputs/*.fa
```

Typical example usage, for specific `k` and using reverse-complement-aware hashes:

```sh
simd-sketch triangle --rc -k 21 inputs/*.fa --output matrix.phylip
```

Maximal usage with default parameters:

```sh
simd-sketch triangle --alg bucket -k 31 -s 10000 -b 8 inputs/*.fna.gz --output matrix.phylip
```

## `.ssketch` files

Sketches are stores as `.ssketch` files alongside each (gzipped) fasta file.
This is done using the [`bincode` crate](https://crates.io/crates/bincode),
whose format is described [here](https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md).

Each sketch starts with the binary format version of the current version of the
tool. The remainder is simply a direct binary encoding of the `Sketch` enum,
using fixed-width integers.

- For bottom sketches, this contains a sorted `Vec<u32>` of length `s`. Total
  size: `4s` bytes. (TODO: This is quite inefficient.)
- For bucket sketches, this contains a `Vec<u32|u16|u8>` of length `s` when storing the bottom
  `b` in `{32, 16, 8}` bits of each value. When storing only the bottom bit, a
  `Vec<u64>` of length `s/64` is used instead. Either way, the total size is
  `s * b/8` bytes.
