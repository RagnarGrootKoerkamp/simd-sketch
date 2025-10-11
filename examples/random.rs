use packed_seq::{PackedSeqVec, SeqVec};

fn main() {
    let k = 31;
    let n = 4_000_000;

    let s = 32768;
    let b = 8;

    let mut bottom = 0.0;
    let mut bucket = 0.0;
    for _ in 0..10 {
        let seq1 = PackedSeqVec::random(n);
        let seq2 = PackedSeqVec::random(n);

        let bottom_sketcher = simd_sketch::SketchParams {
            alg: simd_sketch::SketchAlg::Bottom,
            k,
            s,
            b,
            rc: true,
            duplicate: false,
            coverage: 1,
            filter_empty: true,
            filter_out_n: false,
        }
        .build();
        let bucket_sketcher = simd_sketch::SketchParams {
            alg: simd_sketch::SketchAlg::Bucket,
            k,
            s,
            b,
            rc: true,
            duplicate: false,
            coverage: 1,
            filter_empty: true,
            filter_out_n: false,
        }
        .build();
        let s1 = bottom_sketcher.sketch(seq1.as_slice());
        let s2 = bottom_sketcher.sketch(seq2.as_slice());
        bottom += s1.jaccard_similarity(&s2);
        let s1 = bucket_sketcher.sketch(seq1.as_slice());
        let s2 = bucket_sketcher.sketch(seq2.as_slice());
        bucket += s1.jaccard_similarity(&s2);
    }
    println!("AVG Bottom: {}", bottom / 10.0);
    println!("AVG Bucket: {}", bucket / 10.0);
    // if b == 8 {
    //     for (i, v) in counts.iter().enumerate() {
    //         println!("{i:>3} {v:>3}");
    //     }
    // }
}
