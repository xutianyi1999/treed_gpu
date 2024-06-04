treed_gpu
===

Build TreeD with GPU

### Usage
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
let tree_root = treed_gput::build_treed(unsealed_file, treed_file, &mut buf, 4 * 1024 * 1024 * 1024);
```
[Example](https://github.com/gh-efforts/treed_gpu/blob/master/examples/build_tree.rs)
