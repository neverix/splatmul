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
    pub fn new<'a, X: AsRef<[bf16]>, Y: AsRef<[u32]>, Z: AsRef<[bf16]>>(
        n: usize,
        k: usize,
        m: usize,
        sparse_weights: &'a X,
        sparse_indices: &'a Y,
        decoder_weights: &'a Z,
    ) -> SparseMatmulContext<'a> {
        SparseMatmulContext {
            n,
            k,
            m,
            sparse_weights: sparse_weights.as_ref(),
            sparse_indices: sparse_indices.as_ref(),
            decoder_weights: decoder_weights.as_ref(),
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
