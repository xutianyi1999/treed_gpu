use std::path::PathBuf;
use std::str::FromStr;

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

fn main() {
    let mut args = std::env::args();
    args.next();

    let in_path = args.next().expect("must need in file path");

    let out_path = PathBuf::from(format!("{}_treed", in_path));
    let in_path = PathBuf::from(in_path);

    logger_init();

    let md = std::fs::metadata(&in_path).unwrap();
    treed_gpu::build_treed(&in_path, &out_path, &mut vec![0u8; md.len() as usize * 2 - 32]).unwrap();
}