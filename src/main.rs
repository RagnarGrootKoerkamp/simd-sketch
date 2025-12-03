use std::{
    fs::File,
    io::Seek,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use clap::Parser;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::info;
use packed_seq::{PackedNSeqVec, PackedSeqVec, SeqVec};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use simd_sketch::SketchParams;

/// Compute the sketch distance between two fasta files.
#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// TODO: Support for writing sketches to disk.
#[derive(clap::Subcommand)]
enum Command {
    /// Takes paths to fasta files, and writes .ssketch files.
    Sketch {
        #[command(flatten)]
        params: SketchParams,
        /// Paths to (directories of) (gzipped) fasta files.
        paths: Vec<PathBuf>,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        #[arg(long)]
        no_save: bool,
    },
    /// Compute the distance between two sequences.
    Dist {
        #[command(flatten)]
        params: SketchParams,
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Takes paths to fasta files, and outputs a Phylip distance matrix to stdout.
    Triangle {
        #[command(flatten)]
        params: SketchParams,
        /// Paths to (directories of) (gzipped) fasta files or .ssketch files.
        /// If <path>.ssketch exists, it is automatically used.
        paths: Vec<PathBuf>,
        /// Write phylip distance matrix here, or default to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Save missing sketches to disk, as .ssketch files alongside the input.
        #[arg(long)]
        save_sketches: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
}

const BINCODE_CONFIG: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> = bincode::config::standard().with_fixed_int_encoding();
const EXTENSION: &str = "ssketch";
const SKETCH_VERSION: usize = 1;

#[derive(bincode::Encode, bincode::Decode)]
pub struct VersionedSketch {
    /// This version of simd-sketch only supports encoding version 1.
    /// This is encoded first, so that it can (hopefully) still be recovered in case decoding fails.
    version: usize,
    /// The sketch itself.
    sketch: simd_sketch::Sketch,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    // Initialize thread pool.
    let (Command::Sketch { threads, .. }
    | Command::Dist { threads, .. }
    | Command::Triangle { threads, .. }) = &args.command;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(*threads)
            .build_global()
            .unwrap();
    }

    let (params, paths) = match &args.command {
        Command::Dist {
            params,
            path_a,
            path_b,
            ..
        } => (params, vec![path_a.clone(), path_b.clone()]),
        Command::Sketch { params, paths, .. } | Command::Triangle { params, paths, .. } => {
            (params, collect_paths(&paths))
        }
    };

    let save_sketches = match &args.command {
        Command::Sketch { no_save, .. } => !no_save,
        Command::Dist { .. } => false,
        Command::Triangle { save_sketches, .. } => *save_sketches,
    };

    let q = paths.len();

    let sketcher = params.build();
    let params = sketcher.params();

    let style = indicatif::ProgressStyle::with_template(
        "{msg:.bold} [{elapsed_precise:.cyan}] {bar} {pos}/{len} ({percent:>3}%)",
    )
    .unwrap()
    .progress_chars("##-");

    let start = std::time::Instant::now();

    let num_sketched = AtomicUsize::new(0);
    let num_read = AtomicUsize::new(0);
    let num_written = AtomicUsize::new(0);
    let total_bytes = AtomicUsize::new(0);

    let sketches: Vec<_> = paths
        .par_iter()
        .progress_with_style(style.clone())
        .with_message("Sketching")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|path| {
            let read_sketch = |path| {
                num_read.fetch_add(1, Relaxed);
                let mut file = File::open(path).unwrap();
                // Read the first integer to check the version.
                let version: usize =
                    bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                if version != SKETCH_VERSION {
                    panic!("Unsupported sketch version: {version}. Only version {SKETCH_VERSION} is supported.");
                }
                file.seek(std::io::SeekFrom::Start(0)).unwrap();
                let VersionedSketch {
                    version,
                    sketch,
                } = bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                assert_eq!(version, SKETCH_VERSION);

                let mut sketch_params = sketch.to_params();
                sketch_params.filter_empty = params.filter_empty;
                if *params != sketch_params {
                    panic!(
                        "Sketch parameters do not match:\nCommand line: {params:?}\nOn disk:      {sketch_params:?}",
                    );
                }

                return sketch;
            };

            // Input path is a .ssketch file.
            if path.extension().is_some_and(|ext| ext == EXTENSION) {
                return read_sketch(path);
            }

            // Input path is a .fa, and the .fa.ssketch file exists.
            let ssketch_path = path.with_extension(EXTENSION);
            if ssketch_path.exists() {
                return read_sketch(&ssketch_path);
            }

            let mut reader = needletail::parse_fastx_file(&path).unwrap();

            let mut sketch;
            if params.filter_out_n {
                let mut ranges = vec![];
                let mut seq = PackedNSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let range = seq.push_ascii(&r.unwrap().seq());
                    size += range.len();
                    ranges.push(range);
                }
                total_bytes.fetch_add(size, Relaxed);
                let slices = ranges.into_iter().map(|r| seq.slice(r)).collect_vec();
                sketch = sketcher.sketch_seqs(&slices);
            } else {
                let mut ranges = vec![];
                let mut seq = PackedSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let range = seq.push_ascii(&r.unwrap().seq());
                    size += range.len();
                    ranges.push(range);
                }
                total_bytes.fetch_add(size, Relaxed);
                let slices = ranges.into_iter().map(|r| seq.slice(r)).collect_vec();
                sketch = sketcher.sketch_seqs(&slices);
            }
            num_sketched.fetch_add(1, Relaxed);

            if save_sketches {
                num_written.fetch_add(1, Relaxed);
                let versioned_sketch = VersionedSketch {
                    version: SKETCH_VERSION,
                    sketch,
                };
                bincode::encode_into_std_write(
                    &versioned_sketch,
                    &mut File::create(ssketch_path).unwrap(),
                    BINCODE_CONFIG,
                )
                .unwrap();
                sketch = versioned_sketch.sketch;
            }

            sketch
        })
        .collect();
    let t_sketch = start.elapsed();

    info!(
        "Sketching {q} seqs took {t_sketch:?} ({:?} avg, {} MiB/s)",
        t_sketch / q as u32,
        total_bytes.into_inner() as f32 / t_sketch.as_secs_f32() / (1 << 20) as f32
    );
    let num_read = num_read.into_inner();
    let num_sketched = num_sketched.into_inner();
    let num_written = num_written.into_inner();
    if num_read > 0 {
        info!("Read {num_read} sketches from disk.");
    }
    if num_sketched > 0 {
        info!("Newly sketched {num_sketched} files.");
    }
    if num_written > 0 {
        info!("Wrote {num_written} sketches to disk.");
    }

    if matches!(args.command, Command::Sketch { .. }) {
        // If we are sketching, we are done.
        return;
    }

    let num_pairs = q * (q - 1) / 2;
    let mut pairs = Vec::with_capacity(num_pairs);
    for i in 0..q {
        for j in 0..i {
            pairs.push((i, j));
        }
    }
    let start = std::time::Instant::now();
    let dists: Vec<_> = pairs
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("Distances")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|(i, j)| sketches[i].mash_distance(&sketches[j]))
        .collect();
    let t_dist = start.elapsed();

    let cnt = q * (q - 1) / 2;
    info!(
        "Computing {cnt} dists took {t_dist:?} ({:?} avg)",
        t_dist / cnt.max(1) as u32
    );

    match &args.command {
        Command::Sketch { .. } => {
            unreachable!();
        }
        Command::Dist { .. } => {
            println!("Distance: {:.4}", dists[0]);
            return;
        }
        Command::Triangle { output, .. } => {
            use std::io::Write;

            // Output Phylip triangle format.
            let mut out = Vec::new();
            writeln!(out, "{q}").unwrap();
            let mut d = dists.iter();
            for i in 0..q {
                write!(out, "{}", paths[i].to_str().unwrap()).unwrap();
                for _ in 0..i {
                    write!(out, "\t{:.7}", d.next().unwrap()).unwrap();
                }
                writeln!(out).unwrap();
            }

            match output {
                Some(output) => std::fs::write(output, out).unwrap(),
                None => println!("{}", str::from_utf8(&out).unwrap()),
            }
        }
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

    let extensions = [
        "fa", "fasta", "fq", "fastq", "gz", "fasta.gz", "fq.gz", "fastq.gz",
    ];
    res.retain(|p| extensions.iter().any(|e| p.extension().unwrap() == *e));
    res
}
