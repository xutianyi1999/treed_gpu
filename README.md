treed_gpu
===

Build TreeD with GPU.

Supports simultaneous calculations on multiple GPUs and is faster than CPU multi-thread construction.

[![Latest version](https://img.shields.io/crates/v/treed_gpu.svg)](https://crates.io/crates/treed_gpu)
[![Documentation](https://docs.rs/treed_gpu/badge.svg)](https://docs.rs/treed_gpu)
![License](https://img.shields.io/crates/l/log.svg)

### Usage

- CUDA Toolchain
- Rust Nightly Toolchain

link `treed_gpu` crate

Cargo.toml
```toml
[dependencies]
treed_gpu = "*"
```

main.rs
```rust
let unsealed_file = "unsealed";
let treed_file = "treed";
let unsealed_size = std::fs::metadata(unsealed_file)?.len();
let mut buf = vec![0u8; unsealed_size as usize * 2 - 32];

// use 4GB GPU memory
let tree_root = treed_gput::build_treed(
  unsealed_file,
  treed_file,
  &mut buf,
  4 * 1024 * 1024 * 1024
);
```
[Complete code](https://github.com/gh-efforts/treed_gpu/blob/master/examples/build_tree.rs)


### Build as executable
```shell
export RUST_TOOLCHAIN=nightly
cargo build --release --example build_tree
cd target/release/examples
./build_tree <unsealed-path>
```

build_tree -h

```shell
Usage: build_tree.exe [OPTIONS] --unsealed-path <UNSEALED_PATH>

Options:
  -u, --unsealed-path <UNSEALED_PATH>
  -o, --output-treed <OUTPUT_TREED>    [default: treed_file]
  -h, --help                           Print help
```
