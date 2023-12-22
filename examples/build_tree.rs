use std::alloc::Layout;
use std::path::PathBuf;
use std::slice;
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

fn alloc_buff(size: usize) -> &'static mut [u8] {
    let layout = Layout::array::<u8>(size).expect("failed to allocate label buffer");
    let ptr = hugepage_rs::alloc(layout);

    if ptr.is_null() {
        panic!("unable to allocate huge page");
    }

    unsafe {
        slice::from_raw_parts_mut(ptr, size)
    }
}

fn dealloc_buff(ptr: &mut [u8]) {
    let layout = Layout::array::<u8>(ptr.len()).expect("failed to deallocate label buffer");
    hugepage_rs::dealloc(ptr.as_mut_ptr(), layout);
}

fn main() {
    let mut args = std::env::args();
    args.next();

    let in_path = args.next().expect("must need in file path");

    let out_path = PathBuf::from(format!("{}_treed", in_path));
    let in_path = PathBuf::from(in_path);

    logger_init();

    let md = std::fs::metadata(&in_path).unwrap();

    let buf = alloc_buff(md.len() as usize * 2 - 32);
    treed_gpu::build_treed(&in_path, &out_path, buf, 4 * 1024 * 1024 * 1024).unwrap();
    dealloc_buff(buf)
}