use std::path::PathBuf;

use clap::Parser;
use itertools::Itertools;
use log::{info, trace};
use packed_seq::{PackedSeqVec, SeqVec};
use simd_sketch::SketchParams;
use std::io::Write;

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    #[command(flatten)]
    params: SketchParams,

    paths: Vec<PathBuf>,

    #[arg(long)]
    stats: Option<PathBuf>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let paths = collect_paths(&args.paths);
    let q = paths.len();

    let k = args.params.k;
    let s = args.params.s;
    let b = args.params.b;

    let sketcher = SketchParams {
        alg: args.params.alg,
        rc: true,
        k,
        s,
        b,
        filter_empty: true,
        filter_out_n: false,
    }
    .build();

    let mut sketches = vec![];
    let start = std::time::Instant::now();

    for path in paths {
        trace!("Sketching {path:?}");
        let mut reader = needletail::parse_fastx_file(path).unwrap();
        let start = std::time::Instant::now();
        let mut seqs = vec![];
        while let Some(r) = reader.next() {
            seqs.push(PackedSeqVec::from_ascii(&r.unwrap().seq()));
        }
        trace!("Reading & filtering took {:?}", start.elapsed());
        let start = std::time::Instant::now();
        let seqs = seqs.iter().map(|s| s.as_slice()).collect_vec();
        sketches.push(sketcher.sketch_seqs(&seqs));
        trace!("sketching itself took {:?}", start.elapsed());
    }
    let t_sketch = start.elapsed();
    info!(
        "Sketching {q} seqs took {t_sketch:?} ({:?} avg)",
        t_sketch / q as u32
    );

    let start = std::time::Instant::now();
    let dists = sketches
        .iter()
        .tuple_combinations()
        .map(|(s1, s2)| s1.jaccard_similarity(s2))
        .collect_vec();
    let t_dist = start.elapsed();
    let cnt = q * (q - 1) / 2;
    info!(
        "Computing {cnt} dists took {t_dist:?} ({:?} avg)",
        t_dist / cnt.max(1) as u32
    );
    info!(
        "Params {:?}",
        Args {
            paths: vec![],
            ..args.clone()
        }
    );

    if let Some(stats) = &args.stats {
        let mut writer = std::fs::File::options()
            .create(true)
            .append(true)
            .write(true)
            .open(stats)
            .unwrap();
        writeln!(
            writer,
            "SimdSketch {:?} {q} {k} {s} {b} {} {}",
            args.params.alg,
            t_sketch.as_secs_f32(),
            t_dist.as_secs_f32()
        )
        .unwrap();
    }

    for dist in dists {
        println!("{dist}");
    }
}

fn collect_paths(paths: &Vec<PathBuf>) -> Vec<PathBuf> {
    let mut res = vec![];
    for path in paths {
        if path.is_dir() {
            res.extend(path.read_dir().unwrap().map(|entry| entry.unwrap().path()));
        } else {
            res.push(path.clone());
        }
    }
    res.sort();
    res
}
