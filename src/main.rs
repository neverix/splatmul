#![feature(portable_simd)]
#![feature(new_uninit)]
use half::bf16;
use indicatif::{style, ParallelProgressIterator, ProgressIterator};
use rayon::prelude::*;
use splatmul::benchmarking::{benchmark, SparseMatmulContext};
use splatmul::backward::{backward, BackwardPassContext};
use splatmul::time_fn;
use splatmul::generate::{generate_data, generate_indices, generate_weights};
use ndarray::{Array, ArrayView, Dim};

use splatmul::attempts::{alloc_lower_bound, alloc_uninit_sync, limit_parallel_sparse_matmul};
use splatmul::attempts::{naive_parallel_sparse_matmul, ugly_parallel_sparse_matmul};
use splatmul::attempts::{simd_parallel_sparse_matmul, unsafe_alloc_parallel_sparse_matmul};
use splatmul::types::{DecoderGradientType, WeightGradientType};

const N: usize = 1 << 20;
const K: usize = 64;
const L: usize = 1 << 20;
const M: usize = 1 << 14;
// const N: usize = 1 << 15;
// const K: usize = 64;
// const L: usize = 1 << 12;
// const M: usize = 1 << 14;


fn main() {
    let sparse_weights = generate_weights(N * K, 40.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices(N * K, L as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);

    let scale = 1.0 / (M as f32).sqrt();
    let mut decoder_weights = generate_weights(L * M, scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);
    let mut encoder_weights = generate_weights(L * M, scale);
    println!("First encoder weights: {:?}", &encoder_weights[0..32]);

    let input_data = generate_data(N * M);
    println!("First input data: {:?}", &input_data[0..32]);

    let forward_result_i8 = {
        let ctx = SparseMatmulContext::from_vec(
            N,
            K,
            L,
            M,
            &sparse_weights,
            &sparse_indices,
            &decoder_weights,
        );
        let forward_result = benchmark(unsafe_alloc_parallel_sparse_matmul, ctx, "unsafe_alloc_parallel_sparse_matmul");
        println!("First forward result embeds: {:?}", &forward_result[0..32]);
        println!("Benchmarking to int8...");
        time_fn!(forward_result.par_iter().map(|&x| (x.to_f32() * 127.5f32).clamp(-128., 127.) as i8).collect::<Vec<i8>>())
    };
    println!("First forward result embeds (int8): {:?}", &forward_result_i8[0..32]);

    let backward_ctx = BackwardPassContext {
        n: N,
        k: K,
        l: L,
        m: M,
        input_embeds: &input_data,
        target_embeds: &input_data,
        output_embeds: &forward_result_i8,
        sparse_indices: &sparse_indices,
        sparse_weights: &sparse_weights,
        decoder_weights: &mut decoder_weights,
        encoder_weights: &mut encoder_weights,
    };
    backward(&backward_ctx);
}
