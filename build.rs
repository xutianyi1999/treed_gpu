fn main() {
    println!("cargo:rerun-if-changed=src/cuda/treed.cu");

    #[cfg(not(doc))]
    cc::Build::new()
        .cuda(true)
        .include("src/cuda")
        .file("src/cuda/treed.cu")
        .compile("treed");
}