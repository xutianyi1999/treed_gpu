#![doc(hidden)]

use std::io;
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log::LevelFilter;

fn logger_init() {
    let pattern = if cfg!(debug_assertions) {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {f}:{L} - {m}{n}"
    } else {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {t} - {m}{n}"
    };

    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();

    let config = log4rs::Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(
            Root::builder()
                .appender("stdout")
                .build(LevelFilter::from_str(
                    &std::env::var("BUILD_TREE_LOG").unwrap_or_else(|_| String::from("INFO")),
                ).unwrap()),
        ).unwrap();

    log4rs::init_config(config).unwrap();
}

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    unsealed_path: PathBuf,

    #[arg(short, long, default_value = "treed_file" )]
    output_treed: PathBuf
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let out_path = args.output_treed;
    let in_path = args.unsealed_path;

    logger_init();

    let md = std::fs::metadata(&in_path)?;

    let mut buf = vec![0u8; md.len() as usize * 2 - 32];
    let _root = treed_gpu::build_treed(&in_path, &out_path, &mut buf, 4 * 1024 * 1024 * 1024)?;
    Ok(())
}