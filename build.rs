fn main() {
    cc::Build::new()
        .cuda(true)
        .include("src/cuda")
        .file("src/cuda/treed.cu")
        .compile("treed");
}