use half::bf16;
use std::time::SystemTime;

#[derive(Clone, Copy)]
pub struct SparseMatmulContext<'a> {
    pub n: usize,
    pub k: usize,
    pub m: usize,
    pub sparse_weights: &'a [bf16],
    pub sparse_indices: &'a [u32],
    pub decoder_weights: &'a [bf16],
}

impl SparseMatmulContext<'_> {
    pub fn from_vectors<'a>(
        n: usize,
        k: usize,
        m: usize,
        sparse_weights: &'a Vec<bf16>,
        sparse_indices: &'a Vec<u32>,
        decoder_weights: &'a Vec<bf16>,
    ) -> SparseMatmulContext<'a> {
        SparseMatmulContext {
            n,
            k,
            m,
            sparse_weights: sparse_weights.as_slice(),
            sparse_indices: sparse_indices.as_slice(),
            decoder_weights: decoder_weights.as_slice(),
        }
    }
}

pub fn benchmark<'a, T>(
    fun: fn(SparseMatmulContext) -> T,
    ctx: SparseMatmulContext<'a>,
    fun_name: &str,
) {
    println!("Benchmarking {}", fun_name);
    let start = SystemTime::now();
    fun(ctx);
    let elapsed = start.elapsed().unwrap();
    println!("Elapsed time for {}: {:?}", fun_name, elapsed);
}
