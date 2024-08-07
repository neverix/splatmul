#![feature(portable_simd)]
#![feature(new_uninit)]
use splatmul::benchmarking::{benchmark, SparseMatmulContext};
use splatmul::generate::{generate_indices, generate_weights};

use splatmul::attempts::{alloc_lower_bound, alloc_uninit_sync, limit_parallel_sparse_matmul};
use splatmul::attempts::{naive_parallel_sparse_matmul, ugly_parallel_sparse_matmul};
use splatmul::attempts::{simd_parallel_sparse_matmul, unsafe_alloc_parallel_sparse_matmul};

const N: usize = 1 << 20;
const K: usize = 64;
const L: usize = 1 << 20;
const M: usize = 1 << 14;

fn main() {
    let sparse_weights = generate_weights(N * K, 50.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices(N * K, L as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);
    let scale = 1.0 / (M as f32).sqrt();
    let decoder_weights = generate_weights(L * M, scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);

    let ctx = SparseMatmulContext::from_vec(
        N,
        K,
        L,
        M,
        &sparse_weights,
        &sparse_indices,
        &decoder_weights,
    );
    // benchmark(
    //     simd_parallel_sparse_matmul,
    //     ctx,
    //     "simd_parallel_sparse_matmul",
    // ); // 20.33s
    benchmark(
        unsafe_alloc_parallel_sparse_matmul,
        ctx,
        "unsafe_alloc_parallel_sparse_matmul",
    ); // 17.332739998s
       // benchmark(alloc_uninit_sync, ctx, "alloc_uninit_sync"); // 169ns
       // benchmark(alloc_lower_bound, ctx, "alloc_lower_bound"); // 3.777104249s
       // benchmark(
       //     limit_parallel_sparse_matmul,
       //     ctx,
       //     "limit_parallel_sparse_matmul",
       // ); // 14.725697018s
       // benchmark(
       //     ugly_parallel_sparse_matmul,
       //     ctx,
       //     "naive_parallel_sparse_matmul",
       // ); // 17.930454782s
       // benchmark(
       //     naive_parallel_sparse_matmul,
       //     ctx,
       //     "naiver_parallel_sparse_matmul",
       // ); // 34.811447838s
}
