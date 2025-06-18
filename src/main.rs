#![feature(path_add_extension)]
use std::{fs::File, path::PathBuf};

use clap::Parser;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::info;
use packed_seq::{PackedSeqVec, SeqVec};
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
    },
    /// Compute the distance between two sequences.
    Dist {
        #[command(flatten)]
        params: SketchParams,
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
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
    },
}

const BINCODE_CONFIG: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> = bincode::config::standard().with_fixed_int_encoding();
const EXTENSION: &str = "ssketch";

fn main() {
    env_logger::init();

    let args = Args::parse();

    let (params, paths) = match &args.command {
        Command::Dist {
            params,
            path_a,
            path_b,
        } => (params, vec![path_a.clone(), path_b.clone()]),
        Command::Sketch { params, paths } | Command::Triangle { params, paths, .. } => {
            (params, collect_paths(&paths))
        }
    };

    let save_sketches = match &args.command {
        Command::Sketch { .. } => true,
        Command::Dist { .. } => false,
        Command::Triangle { save_sketches, .. } => *save_sketches,
    };

    let q = paths.len();

    let sketcher = params.build();

    let style = indicatif::ProgressStyle::with_template(
        "{msg:.bold} [{elapsed_precise:.cyan}] {bar} {pos}/{len} ({percent:>3}%)",
    )
    .unwrap()
    .progress_chars("##-");

    let start = std::time::Instant::now();

    let sketches: Vec<_> = paths
        .par_iter()
        .progress_with_style(style.clone())
        .with_message("Sketching")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|path| {
            if path.extension().is_some_and(|ext| ext == EXTENSION) {
                return bincode::decode_from_std_read(
                    &mut File::open(path).unwrap(),
                    BINCODE_CONFIG,
                )
                .unwrap();
            }

            let ssketch_path = {
                // TODO: Use path.add_extension(EXTENSION) when stable.
                // https://doc.rust-lang.org/std/path/struct.Path.html#method.with_added_extension
                let filename = path.file_name().unwrap();
                let new_filename = format!("{}.{}", filename.to_str().unwrap(), EXTENSION);
                path.with_file_name(new_filename)
            };
            if ssketch_path.exists() {
                return bincode::decode_from_std_read(
                    &mut File::open(ssketch_path).unwrap(),
                    BINCODE_CONFIG,
                )
                .unwrap();
            }

            let mut seqs = vec![];
            let mut reader = needletail::parse_fastx_file(&path).unwrap();
            while let Some(r) = reader.next() {
                seqs.push(PackedSeqVec::from_ascii(&r.unwrap().seq()));
            }
            let slices = seqs.iter().map(|s| s.as_slice()).collect_vec();
            let sketch = sketcher.sketch_seqs(&slices);

            if save_sketches {
                bincode::encode_into_std_write(
                    &sketch,
                    &mut File::create(ssketch_path).unwrap(),
                    BINCODE_CONFIG,
                )
                .unwrap();
            }

            sketch
        })
        .collect();
    let t_sketch = start.elapsed();

    info!(
        "Sketching {q} seqs took {t_sketch:?} ({:?} avg)",
        t_sketch / q as u32
    );

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
    res
}
