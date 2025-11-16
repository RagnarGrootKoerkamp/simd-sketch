#![feature(hash_set_entry, array_windows)]
//! # SimdSketch
//!
//! This library provides two types of sequence sketches:
//! - the classic bottom-`s` sketch;
//! - the newer bucket sketch, returning the smallest hash in each of `s` buckets.
//!
//! See the corresponding [blogpost](https://curiouscoding.nl/posts/simd-sketch/) for more background and an evaluation.
//!
//! ## Hash function
//! All internal hashes are 32 bits. Either a forward-only hash or
//! reverse-complement-aware (canonical) hash can be used.
//!
//! **TODO:** Current we use (canonical) ntHash. This causes some hash-collisions
//! for `k <= 16`, [which could be avoided](https://curiouscoding.nl/posts/nthash/#is-nthash-injective-on-kmers).
//!
//! ## BucketSketch
//! For classic bottom-sketch, evaluating the similarity is slow because a
//! merge-sort must be done between the two lists.
//!
//! The bucket sketch solves this by partitioning the hashes into `s` partitions.
//! Previous methods partition into ranges of size `u32::MAX/s`, but here we
//! partition by remainder mod `s` instead.
//!
//! We find the smallest hash for each remainder as the sketch.
//! To compute the similarity, we can simply use the hamming distance between
//! two sketches, which is significantly faster.
//!
//! The bucket sketch similarity has a very strong one-to-one correlation with the classic bottom-sketch.
//!
//! **TODO:** A drawback of this method is that some buckets may remain empty
//! when the input sequences are not long enough.  In that case, _densification_
//! could be applied, but this is not currently implemented. If you need this, please reach out.
//! Instead, we currently simply keep a bitvector indicating empty buckets.
//!
//! ## Jaccard similarity
//! For the bottom sketch, we conceptually estimate similarity as follows:
//! 1. Find the smallest `s` distinct k-mer hashes in the union of two sketches.
//! 2. Return the fraction of these k-mers that occurs in both sketches.
//!
//! For the bucket sketch, we simply return the fraction of parts that have
//! the same k-mer for both sequences (out of those that are not both empty).
//!
//! ## b-bit sketches
//!
//! Instead of storing the full 32-bit hashes, it is sufficient to only store the low bits of each hash.
//! In practice, `b=8` is usually fine.
//! When extra fast comparisons are needed, use `b=1` in combination with a 3 to 4x larger `s`.
//!
//! This causes around `1/2^b` matches because of collisions in the lower bits.
//! We correct for this via `j = (j0 - 1/2^b) / (1 - 1/2^b)`.
//! When the fraction of matches is less than `1/2^b`, this is negative, which we explicitly correct to `0`.
//!
//! ## Mash distance
//! We compute the mash distance as `-log( 2*j / (1+j) ) / k`.
//! This is always >=0, but can be as large as `inf` when `j=0` (as is the case for disjoint input sets).
//!
//! ## Usage
//!
//! The main entrypoint of this library is the [`Sketcher`] object.
//! Construct it in either the forward or canonical variant, and give `k` and `s`.
//! Then call either [`Sketcher::bottom_sketch`] or [`Sketcher::sketch`] on it, and use the
//! `similarity` functions on the returned [`BottomSketch`] and [`BucketSketch`] objects.
//!
//! ```
//! use packed_seq::SeqVec;
//!
//! let sketcher = simd_sketch::SketchParams {
//!     alg: simd_sketch::SketchAlg::Bucket,
//!     rc: false,  // Set to `true` for a canonical (reverse-complement-aware) hash.
//!     k: 31,      // Hash 31-mers
//!     s: 8192,    // Sample 8192 hashes
//!     b: 8,       // Store the bottom 8 bits of each hash.
//!     filter_empty: true, // Explicitly filter out empty buckets for BucketSketch.
//!     filter_out_n: false, // Set to true to ignore k-mers with `N` or other non-ACTG bases.
//! }.build();
//!
//! // Generate two random sequences of 2M characters.
//! let n = 2_000_000;
//! let seq1 = packed_seq::PackedSeqVec::random(n);
//! let seq2 = packed_seq::PackedSeqVec::random(n);
//!
//! // Sketch using given algorithm:
//! let sketch1: simd_sketch::Sketch = sketcher.sketch(seq1.as_slice());
//! let sketch2: simd_sketch::Sketch = sketcher.sketch(seq2.as_slice());
//!
//! // Value between 0 and 1, estimating the fraction of shared k-mers.
//! let j = sketch1.jaccard_similarity(&sketch2);
//! assert!(0.0 <= j && j <= 1.0);
//!
//! let d = sketch1.mash_distance(&sketch2);
//! assert!(0.0 <= d);
//! ```
//!
//! **TODO:** Currently there is no support yet for merging sketches, or for
//! sketching multiple sequences into one sketch. It's not hard, I just need to find a good API.
//! Please reach out if you're interested in this.
//!
//! **TODO:** If you would like a binary instead of a library, again, please reach out :)
//!
//! ## Implementation notes
//!
//! This library works by partitioning the input sequence into 8 chunks,
//! and processing those in parallel using SIMD.
//! This is based on the [`packed-seq`](../packed_seq/index.html) and [`seq-hash`](../seq-hash/index.html) crates
//! that were originally developed for [`simd-minimizers`](../simd_minimizers/index.html).
//!
//! For bottom sketch, the largest hash should be around `target = u32::MAX * s / n` (ignoring duplicates).
//! To ensure a branch-free algorithm, we first collect all hashes up to `bound = 1.5 * target`.
//! Then we sort the collected hashes. If there are at least `s` left after deduplicating, we return the bottom `s`.
//! Otherwise, we double the `1.5` multiplication factor and retry. This
//! factor is cached to make the sketching of multiple genomes more efficient.
//!
//! For bucket sketch, we use the same approach, and increase the factor until we find a k-mer hash in every bucket.
//! In expectation, this needs to collect a fraction around `log(n) * s / n` of hashes, rather than `s / n`.
//! In practice this doesn't matter much, as the hashing of all input k-mers is the bottleneck,
//! and the sorting of the small sample of k-mers is relatively fast.
//!
//! For bucket sketch we assign each element to its bucket via its remainder modulo `s`.
//! We compute this efficiently using [fast-mod](https://github.com/lemire/fastmod/blob/master/include/fastmod.h).
//!
//! ## Performance
//!
//! The sketching throughput of this library is around 2 seconds for a 3GB human genome
//! (once the scaling factor is large enough to avoid a second pass).
//! That's typically a few times faster than parsing a Fasta file.
//!
//! [BinDash](https://github.com/zhaoxiaofei/bindash) instead takes 180s (90x
//! more), when running on a single thread.
//!
//! Comparing sketches is relatively fast, but can become a bottleneck when there are many input sequences,
//! since the number of comparisons grows quadratically. In this case, prefer bucket sketch.
//! As an example, when sketching 5MB bacterial genomes using `s=10000`, each sketch takes 4ms.
//! Comparing two sketches takes 1.6us.
//! This starts to be the dominant factor when the number of input sequences is more than 5000.

mod intrinsics;

use std::{
    collections::{hash_set::Entry, BinaryHeap, HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use itertools::Itertools;
use log::{debug, info};
use packed_seq::{u32x8, ChunkIt, PackedNSeq, PaddedIt, Seq};
use seq_hash::KmerHasher;

/// Use the classic rotate-by-1 for backwards compatibility.
type FwdNtHasher = seq_hash::NtHasher<false, 1>;
type RcNtHasher = seq_hash::NtHasher<true, 1>;

#[derive(bincode::Encode, bincode::Decode, Debug)]
pub enum Sketch {
    BottomSketch(BottomSketch),
    BucketSketch(BucketSketch),
}

fn compute_mash_distance(j: f32, k: usize) -> f32 {
    assert!(j >= 0.0, "Jaccard similarity {j} should not be negative");
    // See eq. 4 of mash paper.
    let mash_dist = -(2. * j / (1. + j)).ln() / k as f32;
    assert!(
        mash_dist >= 0.0,
        "Bad mash distance {mash_dist} for jaccard similarity {j}"
    );
    // NOTE: Mash distance can be >1 when jaccard similarity is close to 0.
    // assert!(
    //     mash_dist <= 1.0,
    //     "Bad mash distance {mash_dist} for jaccard similarity {j}"
    // );
    // Distance 0 is computed as -log(1) and becomes -0.0.
    // This maximum fixes that.
    mash_dist.max(0.0)
}

impl Sketch {
    pub fn to_params(&self) -> SketchParams {
        match self {
            Sketch::BottomSketch(sketch) => SketchParams {
                alg: SketchAlg::Bottom,
                rc: sketch.rc,
                k: sketch.k,
                s: sketch.bottom.len(),
                b: 0,
                seed: 0,
                duplicate: sketch.duplicate,
                count: sketch.count,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false, // FIXME
            },
            Sketch::BucketSketch(sketch) => SketchParams {
                alg: SketchAlg::Bucket,
                rc: sketch.rc,
                k: sketch.k,
                s: sketch.buckets.len(),
                b: sketch.b,
                seed: 0,
                duplicate: sketch.duplicate,
                count: sketch.count,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false, // FIXME
            },
        }
    }
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        match (self, other) {
            (Sketch::BottomSketch(a), Sketch::BottomSketch(b)) => a.jaccard_similarity(b),
            (Sketch::BucketSketch(a), Sketch::BucketSketch(b)) => a.jaccard_similarity(b),
            _ => panic!("Sketches are of different types!"),
        }
    }
    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        let k = match self {
            Sketch::BottomSketch(sketch) => sketch.k,
            Sketch::BucketSketch(sketch) => sketch.k,
        };
        compute_mash_distance(j, k)
    }
}

/// Store only the bottom b bits of each input value.
#[derive(bincode::Encode, bincode::Decode, Debug)]
pub enum BitSketch {
    B32(Vec<u32>),
    B16(Vec<u16>),
    B8(Vec<u8>),
    B1(Vec<u64>),
}

impl BitSketch {
    fn new(b: usize, vals: &Vec<u32>) -> Self {
        match b {
            32 => BitSketch::B32(vals.clone()),
            16 => BitSketch::B16(vals.iter().map(|x| *x as u16).collect()),
            8 => BitSketch::B8(vals.iter().map(|x| *x as u8).collect()),
            1 => BitSketch::B1({
                assert_eq!(vals.len() % 64, 0);
                vals.chunks_exact(64)
                    .map(|xs| {
                        xs.iter()
                            .enumerate()
                            .fold(0u64, |bits, (i, x)| bits | (((x & 1) as u64) << i))
                    })
                    .collect()
            }),
            _ => panic!("Unsupported bit width. Must be 1 or 8 or 16 or 32."),
        }
    }

    fn len(&self) -> usize {
        match self {
            BitSketch::B32(v) => v.len(),
            BitSketch::B16(v) => v.len(),
            BitSketch::B8(v) => v.len(),
            BitSketch::B1(v) => 64 * v.len(),
        }
    }
}

/// A sketch containing the `s` smallest k-mer hashes.
#[derive(bincode::Encode, bincode::Decode, Debug)]
pub struct BottomSketch {
    pub rc: bool,
    pub k: usize,
    pub seed: u32,
    pub duplicate: bool,
    pub count: usize,
    pub bottom: Vec<u32>,
}

impl BottomSketch {
    /// Compute the similarity between two `BottomSketch`es.
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        let a = &self.bottom;
        let b = &other.bottom;
        assert_eq!(a.len(), b.len());
        let mut intersection_size = 0;
        let mut union_size = 0;
        let mut i = 0;
        let mut j = 0;
        while union_size < a.len() {
            intersection_size += (a[i] == b[j]) as usize;
            let di = (a[i] <= b[j]) as usize;
            let dj = (a[i] >= b[j]) as usize;
            i += di;
            j += dj;
            union_size += 1;
        }

        return intersection_size as f32 / a.len() as f32;
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        compute_mash_distance(j, self.k)
    }
}

/// A sketch containing the smallest k-mer hash for each remainder mod `s`.
#[derive(bincode::Encode, bincode::Decode, Debug)]
pub struct BucketSketch {
    pub rc: bool,
    pub k: usize,
    pub b: usize,
    pub seed: u32,
    pub duplicate: bool,
    pub count: usize,
    pub buckets: BitSketch,
    /// Bit-vector indicating empty buckets, so the similarity score can be adjusted accordingly.
    pub empty: Vec<u64>,
}

impl BucketSketch {
    /// Compute the similarity between two `BucketSketch`es.
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        assert_eq!(self.b, other.b);
        let both_empty = self.both_empty(other);
        // if both_empty > 0 {
        // debug!("Both empty: {}", both_empty);
        // }
        match (&self.buckets, &other.buckets) {
            (BitSketch::B32(a), BitSketch::B32(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B16(a), BitSketch::B16(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B8(a), BitSketch::B8(b)) => Self::inner_similarity(a, b, both_empty),
            (BitSketch::B1(a), BitSketch::B1(b)) => Self::b1_similarity(a, b, both_empty),
            _ => panic!("Bit width mismatch"),
        }
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        compute_mash_distance(j, self.k)
    }

    fn inner_similarity<T: Eq>(a: &Vec<T>, b: &Vec<T>, both_empty: usize) -> f32 {
        assert_eq!(a.len(), b.len());
        let f = 1.0
            - std::iter::zip(a, b)
                .map(|(a, b)| (a != b) as u32)
                .sum::<u32>() as f32
                / (a.len() - both_empty) as f32;
        // Correction for accidental matches.
        let bb = (1usize << (size_of::<T>() * 8)) as f32;

        // Correction for accidental matches.
        // Take a max with 0 to avoid correcting into a negative jaccard similarity
        // for uncorrelated sketches.
        (bb * f - 1.0).max(0.0) / (bb - 1.0)
    }

    fn b1_similarity(a: &Vec<u64>, b: &Vec<u64>, both_empty: usize) -> f32 {
        assert_eq!(a.len(), b.len());
        let f = 1.0
            - std::iter::zip(a, b)
                .map(|(a, b)| (*a ^ *b).count_ones())
                .sum::<u32>() as f32
                / (64 * a.len() - both_empty) as f32;

        // Correction for accidental matches.
        // Take a max with 0 to avoid correcting into a negative jaccard similarity
        // for uncorrelated sketches.
        (2. * f - 1.).max(0.0)
    }

    fn both_empty(&self, other: &Self) -> usize {
        std::iter::zip(&self.empty, &other.empty)
            .map(|(a, b)| (a & b).count_ones())
            .sum::<u32>() as usize
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub enum SketchAlg {
    Bottom,
    Bottom2,
    Bottom3,
    Bucket,
}

#[derive(clap::Args, Copy, Clone, Debug, Eq, PartialEq)]
pub struct SketchParams {
    /// Sketch algorithm to use. Defaults to bucket because of its much faster comparisons.
    #[arg(long, default_value_t = SketchAlg::Bucket)]
    #[arg(value_enum)]
    pub alg: SketchAlg,
    /// When set, use forward instead of canonical k-mer hashes.
    #[arg(
        long="fwd",
        num_args(0),
        action = clap::builder::ArgAction::Set,
        default_value = "false",
        default_missing_value = "true",
    )]
    pub rc: bool,
    /// k-mer size.
    #[arg(short, default_value_t = 31)]
    pub k: usize,
    /// Bottom-s sketch, or number of buckets.
    #[arg(short, default_value_t = 10000)]
    pub s: usize,
    /// For bucket-sketch, store only the lower b bits.
    #[arg(short, default_value_t = 8)]
    pub b: usize,
    /// Seed for the hasher.
    #[arg(long, default_value_t = 0)]
    pub seed: u32,

    /// Sketch only duplicate (non-unique) kmers.
    #[arg(long)]
    pub duplicate: bool,
    /// Sketch only kmers with at least this count.
    #[arg(long)]
    pub count: usize,
    /// When sketching read sets of coverage >1, set this for a better initial estimate for the threshold on kmer hashes.
    #[arg(short, long, default_value_t = 1)]
    pub coverage: usize,

    /// For bucket-sketch, store a bitmask of empty buckets, to increase accuracy on small genomes.
    #[arg(skip = true)]
    pub filter_empty: bool,

    #[arg(long)]
    pub filter_out_n: bool,
}

/// An object containing the sketch parameters.
///
/// Contains internal state to optimize the implementation when sketching multiple similar sequences.
pub struct Sketcher {
    params: SketchParams,
    rc_hasher: RcNtHasher,
    fwd_hasher: FwdNtHasher,
    factor: AtomicU64,
}

impl SketchParams {
    pub fn build(&self) -> Sketcher {
        let mut params = *self;
        // factor is pre-multiplied by 10 for a bit more fine-grained resolution.
        let factor;
        match params.alg {
            SketchAlg::Bottom | SketchAlg::Bottom2 | SketchAlg::Bottom3 => {
                // Clear out redundant value.
                params.b = 0;
                factor = 13; // 1.3 * 10
            }
            SketchAlg::Bucket => {
                // To fill s buckets, we need ln(s)*s elements.
                // lg(s) is already a bit larger.
                factor = params.s.ilog2() as u64 * 10; // 1.0 * 10
            }
        }
        if params.alg == SketchAlg::Bottom {}
        Sketcher {
            params,
            rc_hasher: RcNtHasher::new_with_seed(params.k, params.seed),
            fwd_hasher: FwdNtHasher::new_with_seed(params.k, params.seed),
            factor: AtomicU64::new(factor),
        }
    }

    /// Default sketcher that very fast at comparisons, but 20% slower at sketching.
    /// Use for >= 50000 seqs, and safe default when input sequences are > 500'000 characters.
    ///
    /// When sequences are < 100'000 characters, inaccuracies may occur due to empty buckets.
    pub fn default(k: usize) -> Self {
        SketchParams {
            alg: SketchAlg::Bucket,
            rc: true,
            k,
            s: 32768,
            b: 1,
            seed: 0,
            duplicate: false,
            count: 0,
            coverage: 1,
            filter_empty: true,
            filter_out_n: false,
        }
    }

    /// Default sketcher that is fast at sketching, but somewhat slower at comparisons.
    /// Use for <= 5000 seqs, or when input sequences are < 100'000 characters.
    pub fn default_fast_sketching(k: usize) -> Self {
        SketchParams {
            alg: SketchAlg::Bucket,
            rc: true,
            k,
            s: 8192,
            b: 8,
            seed: 0,
            duplicate: false,
            count: 0,
            coverage: 1,
            filter_empty: false,
            filter_out_n: false,
        }
    }
}

impl Sketcher {
    pub fn params(&self) -> &SketchParams {
        &self.params
    }

    /// Sketch a single sequence.
    pub fn sketch(&self, seq: impl Sketchable) -> Sketch {
        self.sketch_seqs(&[seq])
    }

    /// Sketch multiple sequence (fasta records) into a single sketch.
    pub fn sketch_seqs<'s>(&self, seqs: &[impl Sketchable]) -> Sketch {
        match self.params.alg {
            SketchAlg::Bottom => Sketch::BottomSketch(self.bottom_sketch(seqs)),
            SketchAlg::Bottom2 => Sketch::BottomSketch(self.bottom_sketch_2(seqs)),
            SketchAlg::Bottom3 => Sketch::BottomSketch(self.bottom_sketch_3(seqs)),
            SketchAlg::Bucket => Sketch::BucketSketch(self.bucket_sketch(seqs)),
        }
    }

    fn num_kmers<'s>(&self, seqs: &[impl Sketchable]) -> usize {
        seqs.iter()
            .map(|seq| seq.len() - self.params.k + 1)
            .sum::<usize>()
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    fn bottom_sketch<'s>(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);
        let mut out = vec![];
        loop {
            // The total number of kmers is roughly n/coverage.
            // We want s of those, so scale u32::MAX by s/(n/coverage).
            let target = u32::MAX as usize * self.params.s / (n / self.params.coverage);
            let factor = self.factor.load(Relaxed);
            let bound = (target as u128 * factor as u128 / 10 as u128).min(u32::MAX as u128) as u32;

            self.collect_up_to_bound(seqs, bound, &mut out, usize::MAX, |_| unreachable!());

            if self.params.duplicate {
                panic!("Bottom-sketching only duplicate kmers is not implemented yet.");
            }

            if bound == u32::MAX || out.len() >= self.params.s {
                out.sort_unstable();
                let old_len = out.len();
                let new_len = 1 + out
                    .array_windows()
                    .map(|[l, r]| (l != r) as usize)
                    .sum::<usize>();

                debug!("Deduplicated from {old_len} to {new_len}");
                if bound == u32::MAX || new_len >= self.params.s {
                    out.dedup();
                    out.resize(self.params.s, u32::MAX);

                    if log::log_enabled!(log::Level::Debug) {
                        let mut counts = vec![];
                        for g in out.iter().chunk_by(|x| **x).into_iter() {
                            counts.push(g.1.count() as u16);
                        }

                        self.stats(n, &out, counts);
                    }

                    return BottomSketch {
                        rc: self.params.rc,
                        k: self.params.k,
                        seed: self.params.seed,
                        duplicate: self.params.duplicate,
                        count: self.params.count,
                        bottom: out,
                    };
                }
            }

            let new_factor = factor + factor.div_ceil(4);
            let prev = self.factor.fetch_max(new_factor, Relaxed);
            debug!(
                "Found only {:>10} of {:>10} ({:>6.3}%)) Increasing factor from {factor} to {new_factor} (was already {prev})",
                out.len(),
                self.params.s,
                out.len() as f32 / self.params.s as f32,
            );
        }
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    fn bottom_sketch_2<'s>(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);
        let mut out = vec![];

        if self.params.duplicate {
            panic!("Bottom-sketching only duplicate kmers is not implemented yet.");
        }

        self.collect_up_to_bound(seqs, u32::MAX, &mut out, 2 * self.params.s, |out| {
            // TODO: Can this be made more efficient?
            let l0 = out.len();
            out.sort_unstable();
            out.dedup();
            let l1 = out.len();
            out.truncate(self.params.s);
            let l2 = out.len();
            let bound = out.get(self.params.s - 1).copied().unwrap_or(u32::MAX);
            debug!("Len before {l0} => dedup {l1} => truncate {l2}. New bound {bound}");
            bound
        });

        let l0 = out.len();
        out.sort_unstable();
        out.dedup();
        let l1 = out.len();
        debug!("Len before {l0} => after {l1}.");
        out.resize(self.params.s, u32::MAX);

        if log::log_enabled!(log::Level::Debug) {
            let mut counts = vec![];
            for g in out.iter().chunk_by(|x| **x).into_iter() {
                counts.push(g.1.count() as u16);
            }

            self.stats(n, &out, counts);
        }

        return BottomSketch {
            rc: self.params.rc,
            k: self.params.k,
            seed: self.params.seed,
            duplicate: self.params.duplicate,
            count: self.params.count,
            bottom: out,
        };
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    fn bottom_sketch_3<'s>(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        let out = new_collect(
            self.params.s,
            self.params.count,
            self.params.coverage,
            seqs.iter().map(|seq| seq.hash_kmers(&self.rc_hasher)),
        );

        return BottomSketch {
            rc: self.params.rc,
            k: self.params.k,
            seed: self.params.seed,
            duplicate: self.params.duplicate,
            count: self.params.count,
            bottom: out,
        };
    }

    /// s-buckets sketch. Splits the hashes into `s` buckets and returns the smallest hash per bucket.
    /// Buckets are determined via the remainder mod `s`.
    fn bucket_sketch<'s>(&self, seqs: &[impl Sketchable]) -> BucketSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);

        thread_local! {
            static CACHE: std::cell::RefCell<(Vec<u32>, Vec<u32>)> = std::cell::RefCell::new((vec![], vec![]));
        }

        CACHE.with_borrow_mut(|(out, buckets)| {
            out.clear();
            // let mut out = vec![];
            // let mut buckets = vec![u32::MAX; self.params.s];
            buckets.clear();
            buckets.resize(self.params.s, u32::MAX);

            loop {
                // The total number of kmers is roughly n/coverage.
                // We want s of those, so scale u32::MAX by s/(n/coverage).
                let target = u32::MAX as usize * self.params.s / (n / self.params.coverage);
                let factor = self.factor.load(Relaxed);
                let bound = (target as u128 * factor as u128 / 10 as u128).min(u32::MAX as u128) as u32;

                debug!(
                    "n {n:>10} s {} target {target:>10} factor {factor:>3} bound {bound:>10} ({:>6.3}%)",
                    self.params.s,
                    bound as f32 / u32::MAX as f32 * 100.0,
                );

                self.collect_up_to_bound(seqs, bound, out, usize::MAX, |_| unreachable!());

                let mut num_empty = 0;
                if bound == u32::MAX || out.len() >= self.params.s {
                    let m = FM32::new(self.params.s as u32);

                    let mut seen = HashSet::with_capacity(4 * self.params.s);

                    let mut counts = vec![0u16; self.params.s];
                    if !self.params.duplicate {
                        for &hash in &*out {
                            let bucket = m.fastmod(hash);
                            debug_assert!(bucket < buckets.len());
                            let min = unsafe { buckets.get_unchecked_mut(bucket) };
                            if hash < *min {
                                *min = hash;
                                unsafe { *counts.get_unchecked_mut(bucket) = 1 };
                            }
                            else if hash == *min {
                                unsafe { *counts.get_unchecked_mut(bucket) += 1 };
                            }
                            // *min = (*min).min(hash);
                        }
                    } else {
                        for &hash in &*out {
                            let bucket = m.fastmod(hash);
                            debug_assert!(bucket < buckets.len());
                            let min = unsafe { buckets.get_unchecked_mut(bucket) };

                            if hash > *min {
                                continue;
                            }
                            if hash == *min {
                                unsafe { *counts.get_unchecked_mut(bucket) += 1 };
                                continue;
                            }

                            match seen.entry(hash) {
                                Entry::Vacant(e) => {
                                    e.insert();
                                }
                                Entry::Occupied(e) => {
                                    e.remove();
                                    *min = hash;
                                    unsafe { *counts.get_unchecked_mut(bucket) = 2};
                                }
                            }
                        }
                        // debug!(
                        //     "Hashset size: {} ({:>5.2}%)",
                        //     seen.len(),
                        //     seen.len() as f32 / out.len() as f32 * 100.0
                        // );
                    }
                    for &x in &*buckets {
                        if x == u32::MAX {
                            num_empty += 1;
                        }
                    }
                    if bound == u32::MAX || num_empty == 0 {
                        if num_empty > 0 {
                            debug!("Found {num_empty} empty buckets.");
                        }
                        let empty = if num_empty > 0 && self.params.filter_empty {
                            debug!("Found {num_empty} empty buckets. Storing bitmask.");
                            buckets
                                .chunks(64)
                                .map(|xs| {
                                    xs.iter().enumerate().fold(0u64, |bits, (i, x)| {
                                        bits | (((*x == u32::MAX) as u64) << i)
                                    })
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        self.stats(n, buckets, counts);



                        // Reduce buckets mod m.
                        buckets.iter_mut().for_each(|x| *x =  m.fastdiv(*x) as u32);
                        return BucketSketch {
                            rc: self.params.rc,
                            k: self.params.k,
                            b: self.params.b,
                            seed: self.params.seed,
                            duplicate: self.params.duplicate,
                            count: self.params.count,
                            empty,
                            buckets: BitSketch::new(
                                self.params.b,
                                &buckets,
                            ),
                        };
                    }
                }

                let new_factor = factor + factor.div_ceil(4);
                let prev = self.factor.fetch_max(new_factor, Relaxed);
                debug!(
                    "Found only {:>10} of {:>10} ({:>6.3}%, {num_empty:>5} empty) Increasing factor from {factor} to {new_factor} (was already {prev})",
                    out.len(),
                    self.params.s,
                    out.len() as f32 / self.params.s as f32 * 100.,
                );
            }
        })
    }

    fn stats(&self, num_kmers: usize, hashes: &Vec<u32>, counts: Vec<u16>) {
        let num_empty = hashes
            .iter()
            .map(|x| (*x == u32::MAX) as usize)
            .sum::<usize>();
        // Statistics
        let mut cc = HashMap::new();
        for c in counts {
            *cc.entry(c).or_insert(0) += 1;
        }
        let mut cc = cc.into_iter().collect::<Vec<_>>();
        cc.sort_unstable();
        debug!("Counts: {:?}", cc);

        // for bottom sketch:
        // - the i'th smallest value is roughly i * MAX/(n+1).
        // - averaging over i in 1..=s => * (1+s)/2.
        let expected_hash =
            u32::MAX as f64 / (num_kmers as f64 + 1.0) * (1.0 + self.params.s as f64) / 2.0;
        let average_hash = hashes
            .iter()
            .filter(|x| **x != u32::MAX)
            .map(|x| *x as f64)
            .sum::<f64>()
            / (self.params.s - num_empty) as f64;
        let harmonic_hash = (self.params.s - num_empty) as f64
            / hashes
                .iter()
                .filter(|x| **x != u32::MAX)
                .map(|x| 1.0 / (*x as f64 + 1.0))
                .sum::<f64>();
        let harmonic2_hash = self.params.s as f64 * (self.params.s - num_empty) as f64
            / hashes
                .iter()
                .filter(|x| **x != u32::MAX)
                .map(|x| 1.0 / ((*x / self.params.s as u32) as f64 + 1.0))
                .sum::<f64>();
        let median_hash = {
            let mut xs = hashes
                .iter()
                .filter(|x| **x != u32::MAX)
                .copied()
                .collect::<Vec<_>>();
            xs.sort_unstable();
            // debug!("Hashes {xs:?}");
            xs[xs.len() / 2] as f64
        };
        debug!("Expected  hash {expected_hash:>11.2}");
        debug!("Average   hash {average_hash:>11.2}");
        debug!("Harmonic  hash {harmonic_hash:>11.2}");
        debug!("Harmonic2 hash {harmonic2_hash:>11.2}");
        debug!("Median    hash {median_hash:>11.2}");
        debug!("Average ratio   {:>7.4}", average_hash / expected_hash);
        debug!("Harmonic ratio  {:>7.4}", harmonic_hash / expected_hash);
        debug!("Harmonic2 ratio {:>7.4}", harmonic2_hash / expected_hash);
        debug!("Median ratio    {:>7.4}", median_hash / expected_hash);
        let average_kmers = u32::MAX as f64 / average_hash;
        let harmonic_kmers = u32::MAX as f64 / harmonic_hash;
        let harmonic2_kmers = u32::MAX as f64 / harmonic2_hash;
        let median_kmers = u32::MAX as f64 / median_hash;
        let real_kmers = num_kmers as f64 / self.params.s as f64;
        let average_coverage = real_kmers / average_kmers;
        let harmonic_coverage = real_kmers / harmonic_kmers;
        let harmonic2_coverage = real_kmers / harmonic2_kmers;
        let median_coverage = real_kmers / median_kmers;
        debug!("Average coverage   {average_coverage:>7.4}");
        debug!("Harmonic coverage  {harmonic_coverage:>7.4}");
        debug!("Harmonic2 coverage {harmonic2_coverage:>7.4}");
        debug!("Median coverage    {median_coverage:>7.4}");
    }

    /// Collect all values `<= bound`.
    #[inline(always)]
    fn collect_up_to_bound<'s>(
        &self,
        seqs: &[impl Sketchable],
        mut bound: u32,
        out: &mut Vec<u32>,
        batch_size: usize,
        mut callback: impl FnMut(&mut Vec<u32>) -> u32,
    ) {
        out.clear();
        let it = &mut 0;
        if self.params.rc {
            for &seq in seqs {
                let hashes = seq.hash_kmers(&self.rc_hasher);
                collect_impl(&mut bound, hashes, out, batch_size, &mut callback, it);
            }
        } else {
            for &seq in seqs {
                let hashes = seq.hash_kmers(&self.fwd_hasher);
                collect_impl(&mut bound, hashes, out, batch_size, &mut callback, it);
            }
        }
        debug!(
            "Collect up to {bound:>10}: {:>9} ({:>6.3}%)",
            out.len(),
            out.len() as f32 / self.num_kmers(seqs) as f32 * 100.0
        );
    }
}

/// Collect values <= bound.
///
/// Once `out` reaches `max_len`, `callback` will be called to update the bound.
#[inline(always)]
fn collect_impl(
    bound: &mut u32,
    hashes: PaddedIt<impl ChunkIt<u32x8>>,
    out: &mut Vec<u32>,
    batch_size: usize,
    callback: &mut impl FnMut(&mut Vec<u32>) -> u32,
    it: &mut usize,
) {
    let mut simd_bound = u32x8::splat(*bound);
    let mut write_idx = out.len();
    let lane_len = hashes.it.len();
    let mut idx = u32x8::from(std::array::from_fn(|i| (i * lane_len) as u32));
    let max_idx = (8 * lane_len - hashes.padding) as u32;
    let max_idx = u32x8::splat(max_idx);
    hashes.it.for_each(|hashes| {
        // hashes <= simd_bound
        // let mask = !simd_bound.cmp_gt(hashes);
        let mask = hashes.cmp_lt(simd_bound);
        // TODO: transform to a decreasing loop with >= 0 check.
        let in_bounds = idx.cmp_lt(max_idx);
        if write_idx + 8 > out.capacity() {
            out.reserve(out.capacity() + 8);
        }
        unsafe { intrinsics::append_from_mask(hashes, mask & in_bounds, out, &mut write_idx) };
        idx += u32x8::ONE;
        if write_idx >= batch_size as usize {
            debug!("CALLBACK in iteration {it} old bound {bound}");
            unsafe { out.set_len(write_idx) };
            *bound = callback(out);
            simd_bound = u32x8::splat(*bound);
            write_idx = out.len();
        }
        *it += 1;
    });

    unsafe { out.set_len(write_idx) };
}

/// Collect the `s` smallest values that occur at least `cnt` times.
///
/// Implementation:
/// 1. Keep a bound, and collect batch of ~`s` kmers below `bound`.
/// 2. Increase counts in a hashmap for all kmers in the batch.
/// 3. Kmers with the required count are added to a priority queue, and the current `bound` is updated.
///    - Should we collect these in batches too?
/// 4. (From time to time, the hashmap with counts is pruned.)
#[inline(always)]
fn new_collect(
    s: usize,
    cnt: usize,
    coverage: usize,
    hashes: impl Iterator<Item = PaddedIt<impl ChunkIt<u32x8>>>,
) -> Vec<u32> {
    let initial_bound = u32::MAX / coverage as u32;
    let mut bound = u32x8::splat(initial_bound);

    // 1. Collect values <= bound.
    let mut buf = vec![];
    let mut write_idx = buf.len();
    let batch_size = s;

    // 2. HashMap with counts.
    let mut counts = HashMap::<u32, usize>::new();

    // 3. Priority queue with smallest s elements with sufficient count.
    // Largest at the top, so they can be removed easily.
    let mut pq = BinaryHeap::<u32>::from_iter((0..s).map(|_| initial_bound));

    let mut start = std::time::Instant::now();

    let mut process_batch = |buf: &mut Vec<u32>, write_idx: &mut usize, i: usize| {
        let mut top = *pq.peek().unwrap();
        unsafe { buf.set_len(*write_idx) };
        for &hash in &*buf {
            if hash < top {
                *counts.entry(hash).or_insert(0) += 1;
                if counts[&hash] == cnt {
                    pq.pop();
                    pq.push(hash);
                    top = *pq.peek().unwrap();
                    // info!("Push {hash:>10}; new top {top:>10}");
                }
            }
        }
        let now = std::time::Instant::now();
        let duration = now.duration_since(start);
        start = now;
        info!(
            "Batch of size {write_idx:>10} after {i:>10} kmers. Top: {top:>10} = {:5.1}% hashmap size {:>9} took {duration:?}",
            top as f32 / u32::MAX as f32 * 100.0,
            counts.len()
        );
        buf.clear();

        *write_idx = 0;
        u32x8::splat(top)
    };

    let mut i = 0;
    for hashes in hashes {
        // Prevent saving out-of-bound kmers.
        let lane_len = hashes.it.len();
        let mut idx = u32x8::from(std::array::from_fn(|i| (i * lane_len) as u32));
        let max_idx = u32x8::splat((8 * lane_len - hashes.padding) as u32);

        hashes.it.for_each(|hashes| {
            i += 8;
            // hashes <= simd_bound
            // let mask = !simd_bound.cmp_gt(hashes);
            let mask = hashes.cmp_lt(bound);
            // TODO: transform to a decreasing loop with >= 0 check.
            let in_bounds = idx.cmp_lt(max_idx);
            if write_idx + 8 > buf.capacity() {
                buf.reserve(buf.capacity() + 8);
            }
            // Note that this does not increase the length of `out`.
            unsafe {
                intrinsics::append_from_mask(hashes, mask & in_bounds, &mut buf, &mut write_idx)
            };
            idx += u32x8::ONE;
            if write_idx >= batch_size as usize {
                bound = process_batch(&mut buf, &mut write_idx, i);
                i = 0;
            }
        });
    }
    process_batch(&mut buf, &mut write_idx, i);

    pq.into_vec()
}

pub trait Sketchable: Copy {
    fn len(&self) -> usize;
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>>;
}
impl Sketchable for &[u8] {
    fn len(&self) -> usize {
        Seq::len(self)
    }
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>> {
        hasher.hash_kmers_simd(self, 1)
    }
}
impl Sketchable for packed_seq::AsciiSeq<'_> {
    fn len(&self) -> usize {
        Seq::len(self)
    }
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>> {
        hasher.hash_kmers_simd(self, 1)
    }
}
impl Sketchable for packed_seq::PackedSeq<'_> {
    fn len(&self) -> usize {
        Seq::len(self)
    }
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>> {
        hasher.hash_kmers_simd(self, 1)
    }
}
impl<'s> Sketchable for PackedNSeq<'s> {
    fn len(&self) -> usize {
        Seq::len(&self.seq)
    }
    fn hash_kmers<'h, H: KmerHasher>(
        self,
        hasher: &'h H,
    ) -> PaddedIt<impl ChunkIt<u32x8> + use<'s, 'h, H>> {
        hasher.hash_valid_kmers_simd(self, 1)
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug)]
struct FM32 {
    d: u64,
    m: u64,
}
impl FM32 {
    #[inline(always)]
    fn new(d: u32) -> Self {
        Self {
            d: d as u64,
            m: u64::MAX / d as u64 + 1,
        }
    }
    #[inline(always)]
    fn fastmod(self, h: u32) -> usize {
        let lowbits = self.m.wrapping_mul(h as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
    #[inline(always)]
    fn fastdiv(self, h: u32) -> usize {
        ((self.m as u128 * h as u128) >> 64) as u32 as usize
    }
}

#[cfg(test)]
mod test {
    use std::hint::black_box;

    use super::*;
    use packed_seq::SeqVec;

    #[test]
    fn test() {
        let b = 16;

        let k = 31;
        for n in 31..100 {
            for f in [0.0, 0.01, 0.03] {
                let s = n - k + 1;
                let seq = packed_seq::PackedNSeqVec::random(n, f);
                let sketcher = crate::SketchParams {
                    alg: SketchAlg::Bottom,
                    rc: false,
                    k,
                    s,
                    b,
                    seed: 0,
                    duplicate: false,
                    count: 0,
                    coverage: 1,
                    filter_empty: false,
                    filter_out_n: true,
                }
                .build();
                let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                assert_eq!(bottom.len(), s);
                assert!(bottom.is_sorted());

                let s = s.min(10);
                let seq = packed_seq::PackedNSeqVec::random(n, f);
                let sketcher = crate::SketchParams {
                    alg: SketchAlg::Bottom,
                    rc: true,
                    k,
                    s,
                    b,
                    seed: 0,
                    duplicate: false,
                    count: 0,
                    coverage: 1,
                    filter_empty: false,
                    filter_out_n: true,
                }
                .build();
                let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                assert_eq!(bottom.len(), s);
                assert!(bottom.is_sorted());
            }
        }
    }

    #[test]
    fn rc() {
        let b = 32;
        for k in (0..10).map(|_| rand::random_range(1..=64)) {
            for n in (0..10).map(|_| rand::random_range(k..1000)) {
                for s in (0..10).map(|_| rand::random_range(0..n - k + 1)) {
                    for f in [0.0, 0.001, 0.01] {
                        let seq = packed_seq::PackedNSeqVec::random(n, f);
                        let sketcher = crate::SketchParams {
                            alg: SketchAlg::Bottom,
                            rc: true,
                            k,
                            s,
                            b,
                            seed: 0,
                            duplicate: false,
                            count: 0,
                            coverage: 1,
                            filter_empty: false,
                            filter_out_n: true,
                        }
                        .build();
                        let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                        assert_eq!(bottom.len(), s);
                        assert!(bottom.is_sorted());

                        let seq_rc = seq.as_slice().to_revcomp();

                        let bottom_rc = sketcher.bottom_sketch(&[seq_rc.as_slice()]).bottom;
                        assert_eq!(bottom, bottom_rc);
                    }
                }
            }
        }
    }

    #[test]
    fn equal_dist() {
        let s = 1000;
        let k = 10;
        let n = 300;
        let b = 8;
        let seq = packed_seq::AsciiSeqVec::random(n);

        for (alg, filter_empty) in [
            (SketchAlg::Bottom, false),
            (SketchAlg::Bucket, false),
            (SketchAlg::Bucket, true),
        ] {
            let sketcher = crate::SketchParams {
                alg,
                rc: false,
                k,
                s,
                b,
                seed: 0,
                duplicate: false,
                count: 0,
                coverage: 1,
                filter_empty,
                filter_out_n: false,
            }
            .build();
            let sketch = sketcher.sketch(seq.as_slice());
            assert_eq!(sketch.mash_distance(&sketch), 0.0);
        }
    }

    #[test]
    fn fuzz_short() {
        let s = 1024;
        let k = 10;
        for b in [1, 8, 16, 32] {
            for n in [10, 20, 40, 80, 150, 300, 500, 1000, 2000] {
                let seq1 = packed_seq::AsciiSeqVec::random(n);
                let seq2 = packed_seq::AsciiSeqVec::random(n);

                for (alg, filter_empty) in [
                    (SketchAlg::Bottom, false),
                    (SketchAlg::Bucket, false),
                    (SketchAlg::Bucket, true),
                ] {
                    let sketcher = crate::SketchParams {
                        alg,
                        rc: false,
                        k,
                        s,
                        b,
                        seed: 0,
                        duplicate: false,
                        count: 0,
                        coverage: 1,
                        filter_empty,
                        filter_out_n: false,
                    }
                    .build();
                    let s1 = sketcher.sketch(seq1.as_slice());
                    let s2 = sketcher.sketch(seq2.as_slice());
                    s1.mash_distance(&s2);
                }
            }
        }
    }

    #[test]
    fn test_collect() {
        let mut out = vec![];
        let n = black_box(8000);
        let it = (0..n).map(|x| u32x8::splat((x as u32).wrapping_mul(546786567)));
        let padded_it = PaddedIt { it, padding: 0 };
        let mut bound = black_box(u32::MAX / 10);
        collect_impl(
            &mut bound,
            padded_it,
            &mut out,
            usize::MAX,
            &mut |_| unreachable!(),
            &mut 0,
        );
        eprintln!("{out:?}");
    }
}
