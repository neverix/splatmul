use half::bf16;
use std::time::SystemTime;

#[derive(Clone, Copy)]
pub struct SparseMatmulContext<'a> {
    pub n: usize,
    pub k: usize,
    pub l: usize,
    pub m: usize,
    pub sparse_weights: &'a [bf16],
    pub sparse_indices: &'a [u32],
    pub decoder_weights: &'a [bf16],
}

impl SparseMatmulContext<'_> {
    pub fn from_vec<'a>(
        n: usize,
        k: usize,
        l: usize,
        m: usize,
        sparse_weights: &'a Vec<bf16>,
        sparse_indices: &'a Vec<u32>,
        decoder_weights: &'a Vec<bf16>,
    ) -> SparseMatmulContext<'a> {
        SparseMatmulContext {
            n,
            k,
            l,
            m,
            sparse_weights: sparse_weights.as_slice(),
            sparse_indices: sparse_indices.as_slice(),
            decoder_weights: decoder_weights.as_slice(),
        }
    }
}

pub fn benchmark<T>(
    fun: fn(SparseMatmulContext) -> T,
    ctx: SparseMatmulContext,
    fun_name: &str,
) -> T {
    println!("Benchmarking {}", fun_name);
    let start = SystemTime::now();
    let result = fun(ctx);
    let elapsed = start.elapsed().unwrap();
    println!("Elapsed time for {}: {:?}", fun_name, elapsed);
    result
}

#[macro_export]
macro_rules! time_fn {
    ($e: expr) => {{
        let start = std::time::Instant::now();
        let result = $e;
        let duration = start.elapsed();
        println!("Time: {:?}", duration);
        result
    }}
}

#[macro_export]
macro_rules! make_progress {
    () => {{
        use indicatif::style;
        style::ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} [{elapsed_precise} {eta_precise}]").unwrap()
    }}
}
