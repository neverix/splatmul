#![feature(portable_simd)]
#![feature(new_uninit)]
use splatmul::benchmarking::benchmark;
use splatmul::generate::{generate_indices, generate_weights};

use splatmul::attempts::unsafe_alloc_parallel_sparse_matmul;
use splatmul::attempts::{alloc_lower_bound, alloc_uninit_sync, limit_parallel_sparse_matmul};
use splatmul::attempts::{naive_parallel_sparse_matmul, ugly_parallel_sparse_matmul};

fn main() {
    let n = 1 << 20;
    let k = 64;
    let l = 1 << 20;
    let m = 1 << 14;
    let sparse_weights = generate_weights(n * k, 50.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices(n * k, l as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);
    let scale = 1.0 / (m as f32).sqrt();
    let decoder_weights = generate_weights(l * m, scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);

    let ctx = splatmul::benchmarking::SparseMatmulContext::from_vectors(
        n,
        k,
        m,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
    );
    benchmark(
        unsafe_alloc_parallel_sparse_matmul,
        ctx,
        "unsafe_alloc_parallel_sparse_matmul",
    ); // 17.332739998s
    benchmark(alloc_uninit_sync, ctx, "alloc_uninit_sync"); // 169ns
    benchmark(alloc_lower_bound, ctx, "alloc_lower_bound"); // 3.777104249s
    benchmark(
        limit_parallel_sparse_matmul,
        ctx,
        "limit_parallel_sparse_matmul",
    ); // 14.725697018s
    benchmark(
        ugly_parallel_sparse_matmul,
        ctx,
        "naive_parallel_sparse_matmul",
    ); // 17.930454782s
    benchmark(
        naive_parallel_sparse_matmul,
        ctx,
        "naiver_parallel_sparse_matmul",
    ); // 34.811447838s
}
