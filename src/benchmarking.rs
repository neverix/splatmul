use crate::constants::{K, L, M, N};
use half::bf16;
use std::time::SystemTime;

#[derive(Clone, Copy)]
pub struct SparseMatmulContext<'a> {
    pub sparse_weights: &'a [bf16; N * K],
    pub sparse_indices: &'a [u32; N * K],
    pub decoder_weights: &'a [bf16; L * M],
}

pub fn benchmark<'a, T>(
    fun: fn(SparseMatmulContext) -> T,
    sparse_weights: &'a Vec<bf16>,
    sparse_indices: &'a Vec<u32>,
    decoder_weights: &'a Vec<bf16>,
    fun_name: &str,
) {
    println!("Benchmarking {}", fun_name);
    let start = SystemTime::now();
    fun(SparseMatmulContext {
        sparse_weights: sparse_weights.as_slice().try_into().unwrap(),
        sparse_indices: sparse_indices.as_slice().try_into().unwrap(),
        decoder_weights: decoder_weights.as_slice().try_into().unwrap(),
    });
    let elapsed = start.elapsed().unwrap();
    println!("Elapsed time for {}: {:?}", fun_name, elapsed);
}
