use half::bf16;
use std::time::SystemTime;

#[derive(Clone)]
pub struct SparseMatmulContext {
    pub n: usize,
    pub k: usize,
    pub l: usize,
    pub m: usize,
    pub sparse_weights: Box<[bf16]>,
    pub sparse_indices: Box<[u32]>,
    pub decoder_weights: Box<[bf16]>,
}

impl SparseMatmulContext {
    pub fn from_vec(
        n: usize,
        k: usize,
        l: usize,
        m: usize,
        sparse_weights: Vec<bf16>,
        sparse_indices: Vec<u32>,
        decoder_weights: Vec<bf16>,
    ) -> SparseMatmulContext {
        SparseMatmulContext {
            n,
            k,
            l,
            m,
            sparse_weights: sparse_weights.into_boxed_slice(),
            sparse_indices: sparse_indices.into_boxed_slice(),
            decoder_weights: decoder_weights.into_boxed_slice(),
        }
    }
}

pub fn benchmark<T>(
    fun: fn(&SparseMatmulContext) -> T,
    ctx: &SparseMatmulContext,
    fun_name: &str,
) -> T {
    println!("Benchmarking {}", fun_name);
    let start = SystemTime::now();
    let result = fun(ctx);
    let elapsed = start.elapsed().unwrap();
    println!("Elapsed time for {}: {:?}", fun_name, elapsed);
    result
}
