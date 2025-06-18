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
//! This is based on the [`packed-seq`](../packed_seq/index.html) and [`simd-minimizers`](../simd_minimizers/index.html) crates.
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

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use log::{debug, info};
use packed_seq::{u32x8, Seq};
use simd_minimizers::private::nthash::NtHasher;

#[derive(bincode::Encode, bincode::Decode)]
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
#[derive(bincode::Encode, bincode::Decode)]
pub enum BitSketch {
    B32(Vec<u32>),
    B16(Vec<u16>),
    B8(Vec<u8>),
    B1(Vec<u64>),
}

impl BitSketch {
    fn new(b: usize, vals: Vec<u32>) -> Self {
        match b {
            32 => BitSketch::B32(vals),
            16 => BitSketch::B16(vals.into_iter().map(|x| x as u16).collect()),
            8 => BitSketch::B8(vals.into_iter().map(|x| x as u8).collect()),
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
}

/// A sketch containing the `s` smallest k-mer hashes.
#[derive(bincode::Encode, bincode::Decode)]
pub struct BottomSketch {
    rc: bool,
    k: usize,
    bottom: Vec<u32>,
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
#[derive(bincode::Encode, bincode::Decode)]
pub struct BucketSketch {
    rc: bool,
    k: usize,
    b: usize,
    buckets: BitSketch,
    /// Bit-vector indicating empty buckets, so the similarity score can be adjusted accordingly.
    empty: Vec<u64>,
}

impl BucketSketch {
    /// Compute the similarity between two `BucketSketch`es.
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        assert_eq!(self.b, other.b);
        let both_empty = self.both_empty(other);
        if both_empty > 0 {
            info!("Both empty: {}", both_empty);
        }
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

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum SketchAlg {
    Bottom,
    Bucket,
}

#[derive(clap::Args, Copy, Clone, Debug)]
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
    /// For bucket-sketch, store a bitmask of empty buckets, to increase accuracy on small genomes.
    #[arg(skip = true)]
    pub filter_empty: bool,
}

/// An object containing the sketch parameters.
///
/// Contains internal state to optimize the implementation when sketching multiple similar sequences.
pub struct Sketcher {
    params: SketchParams,
    factor: AtomicUsize,
}

impl SketchParams {
    pub fn build(&self) -> Sketcher {
        Sketcher {
            params: *self,
            factor: 2.into(),
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
            filter_empty: true,
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
            filter_empty: false,
        }
    }
}

impl Sketcher {
    /// Sketch a single sequence.
    pub fn sketch<'s, S: Seq<'s>>(&self, seq: S) -> Sketch {
        self.sketch_seqs(&[seq])
    }

    /// Sketch multiple sequence (fasta records) into a single sketch.
    pub fn sketch_seqs<'s, S: Seq<'s>>(&self, seqs: &[S]) -> Sketch {
        match self.params.alg {
            SketchAlg::Bottom => Sketch::BottomSketch(self.bottom_sketch(seqs)),
            SketchAlg::Bucket => Sketch::BucketSketch(self.bucket_sketch(seqs)),
        }
    }

    fn num_kmers<'s, S: Seq<'s>>(&self, seqs: &[S]) -> usize {
        seqs.iter()
            .map(|seq| seq.len() - self.params.k + 1)
            .sum::<usize>()
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    /// Prefer [`Sketcher::sketch`] instead, which is much faster and just as
    /// accurate when input sequences are not too short.
    fn bottom_sketch<'s, S: Seq<'s>>(&self, seqs: &[S]) -> BottomSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);
        let mut out = vec![];
        loop {
            let target = u32::MAX as usize / n * self.params.s;
            let bound =
                (target.saturating_mul(self.factor.load(SeqCst))).min(u32::MAX as usize) as u32;

            self.collect_up_to_bound(seqs, bound, &mut out);

            if bound == u32::MAX || out.len() >= self.params.s {
                out.sort_unstable();
                out.dedup();
                if bound == u32::MAX || out.len() >= self.params.s {
                    out.resize(self.params.s, u32::MAX);

                    break BottomSketch {
                        rc: self.params.rc,
                        k: self.params.k,
                        bottom: out,
                    };
                }
            }
            self.factor
                .fetch_add((self.factor.load(SeqCst) + 1) / 2, SeqCst);
            debug!("Increase factor to {}", self.factor.load(SeqCst));
        }
    }

    /// s-buckets sketch. Splits the hashes into `s` buckets and returns the smallest hash per bucket.
    /// Buckets are determined via the remainder mod `s`.
    fn bucket_sketch<'s, S: Seq<'s>>(&self, seqs: &[S]) -> BucketSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);
        let mut out = vec![];
        let mut buckets = vec![u32::MAX; self.params.s];
        loop {
            let target = u32::MAX as usize / n * self.params.s;
            let bound =
                (target.saturating_mul(self.factor.load(SeqCst))).min(u32::MAX as usize) as u32;

            self.collect_up_to_bound(seqs, bound, &mut out);

            if bound == u32::MAX || out.len() >= self.params.s {
                let m = FM32::new(self.params.s as u32);
                for &hash in &out {
                    let bucket = m.fastmod(hash);
                    buckets[bucket] = buckets[bucket].min(hash);
                }
                let mut empty = 0;
                for &x in &buckets {
                    if x == u32::MAX {
                        empty += 1;
                    }
                }
                if bound == u32::MAX || empty == 0 {
                    if empty > 0 {
                        info!("Found {empty} empty buckets.");
                    }
                    let empty = if empty > 0 && self.params.filter_empty {
                        info!("Found {empty} empty buckets. Storing bitmask.");
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

                    break BucketSketch {
                        rc: self.params.rc,
                        k: self.params.k,
                        b: self.params.b,
                        empty,
                        buckets: BitSketch::new(
                            self.params.b,
                            buckets.into_iter().map(|x| m.fastdiv(x) as u32).collect(),
                        ),
                    };
                }
            }
            self.factor
                .fetch_add((self.factor.load(SeqCst) + 1) / 2, SeqCst);
            debug!("Increase factor to {}", self.factor.load(SeqCst));
        }
    }

    fn collect_up_to_bound<'s, S: Seq<'s>>(&self, seqs: &[S], bound: u32, out: &mut Vec<u32>) {
        if self.params.rc {
            collect_up_to_bound_generic::<true, S>(seqs, self.params.k, bound, out);
        } else {
            collect_up_to_bound_generic::<false, S>(seqs, self.params.k, bound, out);
        }
    }
}

fn collect_up_to_bound_generic<'s, const RC: bool, S: Seq<'s>>(
    seqs: &[S],
    k: usize,
    bound: u32,
    out: &mut Vec<u32>,
) {
    let simd_bound = u32x8::splat(bound);
    out.clear();

    for &seq in seqs {
        let (hashes_head, hashes_tail) =
            simd_minimizers::private::nthash::nthash_seq_simd::<RC, S, NtHasher>(seq, k, 1);

        let mut write_idx = out.len();
        for hashes in hashes_head {
            let mask = hashes.cmp_lt(simd_bound);
            if write_idx + 8 >= out.len() {
                out.resize(write_idx * 3 / 2 + 8, 0);
            }
            unsafe { intrinsics::append_from_mask(hashes, mask, out, &mut write_idx) };
        }

        out.resize(write_idx, 0);

        for hash in hashes_tail {
            if hash <= bound {
                out.push(hash);
            }
        }
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
    fn new(d: u32) -> Self {
        Self {
            d: d as u64,
            m: u64::MAX / d as u64 + 1,
        }
    }
    fn fastmod(self, h: u32) -> usize {
        let lowbits = self.m.wrapping_mul(h as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
    fn fastdiv(self, h: u32) -> usize {
        ((self.m as u128 * h as u128) >> 64) as u32 as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use packed_seq::SeqVec;

    #[test]
    fn test() {
        use packed_seq::SeqVec;
        let b = 16;

        let k = 31;
        for n in 31..100 {
            let s = n - k + 1;
            let seq = packed_seq::PackedSeqVec::random(n);
            let sketcher = crate::SketchParams {
                alg: SketchAlg::Bottom,
                rc: false,
                k,
                s,
                b,
                filter_empty: false,
            }
            .build();
            let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
            assert_eq!(bottom.len(), s);
            assert!(bottom.is_sorted());

            let s = s.min(10);
            let seq = packed_seq::PackedSeqVec::random(n);
            let sketcher = crate::SketchParams {
                alg: SketchAlg::Bottom,
                rc: true,
                k,
                s,
                b,
                filter_empty: false,
            }
            .build();
            let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
            assert_eq!(bottom.len(), s);
            assert!(bottom.is_sorted());
        }
    }

    #[test]
    fn rc() {
        use packed_seq::SeqVec;

        let b = 32;
        for k in (0..10).map(|_| rand::random_range(1..100)) {
            for n in (0..10).map(|_| rand::random_range(k..1000)) {
                for s in (0..10).map(|_| rand::random_range(0..n - k + 1)) {
                    let seq = packed_seq::AsciiSeqVec::random(n);
                    let sketcher = crate::SketchParams {
                        alg: SketchAlg::Bottom,
                        rc: true,
                        k,
                        s,
                        b,
                        filter_empty: false,
                    }
                    .build();
                    let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                    assert_eq!(bottom.len(), s);
                    assert!(bottom.is_sorted());

                    let seq_rc = packed_seq::AsciiSeqVec::from_ascii(
                        &seq.seq
                            .iter()
                            .rev()
                            .map(|c| packed_seq::complement_char(*c))
                            .collect::<Vec<_>>(),
                    );

                    let bottom_rc = sketcher.bottom_sketch(&[seq_rc.as_slice()]).bottom;
                    assert_eq!(bottom, bottom_rc);
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
                filter_empty,
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
                        filter_empty,
                    }
                    .build();
                    let s1 = sketcher.sketch(seq1.as_slice());
                    let s2 = sketcher.sketch(seq2.as_slice());
                    s1.mash_distance(&s2);
                }
            }
        }
    }
}
