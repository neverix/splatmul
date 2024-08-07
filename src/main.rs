#![feature(portable_simd)]
#![feature(new_uninit)]
use splatmul::benchmarking::benchmark;
use splatmul::constants::{K, L, M, N};
use splatmul::generate::{generate_indices, generate_weights};

use splatmul::attempts::unsafe_alloc_parallel_sparse_matmul;
use splatmul::attempts::{alloc_lower_bound, alloc_uninit_sync, limit_parallel_sparse_matmul};
use splatmul::attempts::{naive_parallel_sparse_matmul, ugly_parallel_sparse_matmul};

fn main() {
    let sparse_weights = generate_weights::<{ N * K }>(50.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices::<{ N * K }>(L as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);
    let scale = 1.0 / (M as f32).sqrt();
    let decoder_weights = generate_weights::<{ L * M }>(scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);

    benchmark(
        unsafe_alloc_parallel_sparse_matmul,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
        "unsafe_alloc_parallel_sparse_matmul",
    ); // 15.4s
    benchmark(
        alloc_uninit_sync,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
        "alloc_uninit_sync",
    ); // 130ns
    benchmark(
        alloc_lower_bound,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
        "alloc_lower_bound",
    ); // 7.8s
    benchmark(
        limit_parallel_sparse_matmul,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
        "limit_parallel_sparse_matmul",
    ); // 13.7s
    benchmark(
        ugly_parallel_sparse_matmul,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
        "naive_parallel_sparse_matmul",
    ); // 16.7s
    benchmark(
        naive_parallel_sparse_matmul,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
        "naiver_parallel_sparse_matmul",
    ); // 30.8s
}
