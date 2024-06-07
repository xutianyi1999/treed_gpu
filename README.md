treed_gpu
===

Build TreeD with GPU.

Supports simultaneous calculations on multiple GPUs and is faster than CPU multi-thread construction.

[![Latest version](https://img.shields.io/crates/v/treed_gpu.svg)](https://crates.io/crates/treed_gpu)
[![Documentation](https://docs.rs/treed_gpu/badge.svg)](https://docs.rs/treed_gpu)
![License](https://img.shields.io/crates/l/log.svg)

### About TreeD
TreeD is a tree-like data structure used in Filecoin’s PoRep. 
It organizes data in a hierarchical manner, 
facilitating efficient and secure verification of data replication.

TreeD is build over the nodes of the unencoded sector 𝐷.

*Purpose:*

The primary purpose of TreeD is to ensure data integrity and tamper-proof storage.
It achieves this by constructing a Merkle tree that enables efficient and secure proof generation and verification.

*Data Segmentation:*

The original data is divided into smaller chunks. 
Each chunk will become a leaf node in the Merkle tree

*Hash Calculation:*

Node data chunk is hashed using a cryptographic hash function (SHA-256). 
This results in a list of hash values corresponding to the leaf nodes.

*Tree Construction:*

The leaf nodes (hashes of the data chunks) are paired and combined to compute their parent node’s hash. 
This process continues iteratively, layer by layer, until a single root hash is obtained. 
This root hash represents the entire dataset uniquely.

*Example Process:*

- Suppose we have four data chunks: D1,D2,D3,D4
- Each data size is 32 bytes
- Pairwise combine and hash the leaf nodes: H12 = H(D1, D2), H34 = H(D3, D4)
- Compute the root hash:Root = H(H12, H34)
- When validating data, just check the hashes of the root node and associated paths

The detailed technical specifications of TreeD and its role in Filecoin's PoRep can be found in the [Filecoin Specification](https://spec.filecoin.io/#section-algorithms.sdr.replication).

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

#### Merged into rust-fil-proofs
[rust-fil-proffs/filecoin-proofs/Cargo.toml](https://github.com/filecoin-project/rust-fil-proofs/blob/51c7a79d9058a45988290b4da7217bec05750cc8/filecoin-proofs/Cargo.toml)

Add a line

```toml
[dependencies]
treed_gpu = "*"
```

[rust-fil-proofs/filecoin-proofs/src/api/seal.rs](https://github.com/filecoin-project/rust-fil-proofs/blob/51c7a79d9058a45988290b4da7217bec05750cc8/filecoin-proofs/src/api/seal.rs#L162-L175)

Replace it with the following code

```rust
// Only 32G or 64G Sector use treed_gpu
 if porep_config.sector_size.0 == 32 * 1024 * 1024 * 1024 ||
    porep_config.sector_size.0 == 64 * 1024 * 1024 * 1024 {
    let input_file = &out_path;
    let out_file = StoreConfig::data_path(&config.path, &config.id);

    let treed_size = porep_config.sector_size.0 as usize * 2 - 32;

    let cuda_memory_size = 4 * 1024 * 1024 * 1024;
    let buff = vec![0u8; treed_size];
    let root = treed_gpu::build_treed(input_file.as_ref(), &out_file, buff, cuda_memory_size)?;
    let root = <Tree::Hasher as Hasher>::Domain::from_slice(&root);

    config.size = Some(treed_size / 32);
    let comm_d_root: Fr = root.into();
    let comm_d = commitment_from_fr(comm_d_root);

    Ok((config, comm_d))
} else {
    let data_tree = create_base_merkle_tree::<BinaryMerkleTree<DefaultPieceHasher>>(
        Some(config.clone()),
        base_tree_leafs,
        &data,
    )?;
    drop(data);

    config.size = Some(data_tree.len());
    let comm_d_root: Fr = data_tree.root().into();
    let comm_d = commitment_from_fr(comm_d_root);

    drop(data_tree);

    Ok((config, comm_d))
}
```

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
