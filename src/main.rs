#![feature(portable_simd)]
#![feature(new_uninit)]
use std::convert::identity;

use half::bf16;
use indicatif::ProgressIterator;
use ndarray::{Array, ArrayView, Dim};
use rayon::prelude::*;
use splatmul::adam::AdamState;
use splatmul::attempts::classic::beautiful_parallel_sparse_matmul;
use splatmul::backward::{backward, BackwardPassContext};
use splatmul::benchmarking::{benchmark, SparseMatmulContext};
use splatmul::generate::{generate_data, generate_indices, generate_orthogonal, generate_weights};
use splatmul::{make_progress, time_fn};

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
const M_CHUNK: usize = 1 << 7;

fn main() {
    let sparse_weights = generate_weights(N * K, 40.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices(N * K, L as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);

    let scale = 1.0 / (M as f32).sqrt();
    // let mut encoder_weights = {
    //     let encoder_weights_transpose = generate_orthogonal(M, N, scale);
    //     (0..N*M).map(|i| encoder_weights_transpose[i % M * N + i / M]).collect::<Vec<bf16>>()
    // };
    let mut encoder_weights = generate_weights(N * M, scale);
    println!("First encoder weights: {:?}", &encoder_weights[0..32]);
    let mut decoder_weights = encoder_weights
        .par_iter()
        .map(|&x| x)
        .collect::<Vec<bf16>>();
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);

    let mut adam = time_fn!(AdamState::new(1e-3, 0.9, 0.999, 1e-8, N * M, 16), "Adam initialization...");
    time_fn!({
        let grads = generate_weights(N * M, scale);
        adam.update(grads.as_slice(), decoder_weights.as_mut_slice(), identity);
    }, "Adam update...");
    return;

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
        // benchmark(ugly_parallel_sparse_matmul, ctx, "ugly_parallel_sparse_matmul");
        benchmark(
            beautiful_parallel_sparse_matmul,
            ctx,
            "beautiful_parallel_sparse_matmul",
        );
        let forward_result = benchmark(
            unsafe_alloc_parallel_sparse_matmul,
            ctx,
            "unsafe_alloc_parallel_sparse_matmul",
        );
        println!("First forward result embeds: {:?}", &forward_result[0..32]);
        println!("Benchmarking to int8...");
        time_fn!(forward_result
            .par_iter()
            .map(|&x| (x.to_f32() * 127.5f32).clamp(-128., 127.) as i8)
            .collect::<Vec<i8>>())
    };
    println!(
        "First forward result embeds (int8): {:?}",
        &forward_result_i8[0..32]
    );

    // let backward_ctx = BackwardPassContext {
    //     n: N,
    //     k: K,
    //     l: L,
    //     m: M,
    //     input_embeds: &input_data,
    //     target_embeds: &input_data,
    //     output_embeds: &forward_result_i8,
    //     sparse_indices: &sparse_indices,
    //     sparse_weights: &sparse_weights,
    //     decoder_weights: &mut decoder_weights,
    //     encoder_weights: &mut encoder_weights,
    // };
    // backward::<M_CHUNK>(&backward_ctx);
}
