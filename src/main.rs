#![feature(portable_simd)]
#![feature(new_uninit)]
use std::mem::MaybeUninit;

use half::bf16;
use indicatif::{style, ParallelProgressIterator, ProgressIterator};
use rayon::prelude::*;
use splatmul::benchmarking::{benchmark, BackwardPassContext, SparseMatmulContext};
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

macro_rules! time_fn {
    ($e: expr) => {{
        let start = std::time::Instant::now();
        let result = $e;
        let duration = start.elapsed();
        println!("Time: {:?}", duration);
        result
    }}
}

fn compute_output_gradient_slice(ctx: &BackwardPassContext, n: usize, m_start: usize, m_end: usize) -> Array<f32, Dim<[usize; 1]>> {
    let target_embed_row = &ctx.target_embeds[n * ctx.m..(n + 1) * ctx.m][m_start..m_end];
    let output_embed_row = &ctx.output_embeds[n * ctx.m..(n + 1) * ctx.m][m_start..m_end];
    
    let target_embed_row_nd = ArrayView::from_shape((m_end - m_start,), target_embed_row).unwrap();
    let output_embed_row_nd = ArrayView::from_shape((m_end - m_start,), output_embed_row).unwrap();

    let target_f32 = target_embed_row_nd.mapv(|x| (x as f32) / 127.5);
    let output_f32 = output_embed_row_nd.mapv(|x| (x as f32) / 127.5);
    target_f32 - output_f32
}

fn compute_grads(ctx: &BackwardPassContext) -> Vec<DecoderGradientType> {
    let mut v = Vec::<f32>::with_capacity(ctx.n * ctx.k);
    v.spare_capacity_mut()
        .par_chunks_mut(ctx.k)
        .progress_with_style(style::ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} [{elapsed_precise}]").unwrap())
        .enumerate()
        .for_each(|(n, outputs)| {
            let k_grads: Vec<MaybeUninit<f32>> = (0..ctx.k).into_iter().map(|k| {
                let index = ctx.sparse_indices[n * ctx.k + k];
                let decoder_row = &ctx.decoder_weights[index as usize * ctx.m..(index as usize + 1) * ctx.m];

                let decoder_row_nd = ArrayView::from_shape((ctx.m,), decoder_row).unwrap();
                let decoder_row_f32 = decoder_row_nd.mapv(|x| x.to_f32());
                let output_gradient = compute_output_gradient_slice(ctx, n, 0, ctx.m);

                let gradient = output_gradient.dot(&decoder_row_f32);
                MaybeUninit::new(gradient)
            }).collect::<Vec<MaybeUninit<DecoderGradientType>>>();
            outputs.copy_from_slice(&k_grads);
        });
    unsafe { v.set_len(ctx.n * ctx.k) };
    v
}

const M_CHUNK: usize = 1 << 6;
// const M_CHUNK: usize = 1 << 2;

fn weight_grads(ctx: &BackwardPassContext, decoder_grads: &[DecoderGradientType]) -> Vec<WeightGradientType> {
    let lm = ctx.l * ctx.m;
    let mut v = Vec::<WeightGradientType>::with_capacity(lm * 2);
    let decoder_offset = lm;
    v.spare_capacity_mut()
        // .par_chunks_mut(ctx.l * M_CHUNK)
        .chunks_mut(ctx.l * M_CHUNK)
        .progress_with_style(style::ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} [{elapsed_precise}]").unwrap())
        .enumerate()
        .for_each(|(m_start, outputs)| {
            let is_decoder = m_start >= decoder_offset;
            let real_m_start = if is_decoder { m_start - decoder_offset } else { m_start };
            for l in (0..ctx.l).progress_with_style(style::ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} [{elapsed_precise}]").unwrap()) {
                let mut grad_accum = Array::from_elem((M_CHUNK,), 0f32);
                for n in 0..ctx.n {
                    let big_elem = if is_decoder {
                        compute_output_gradient_slice(ctx, n, real_m_start, real_m_start + M_CHUNK)
                    } else {
                        let input_embeds_i8 = &ctx.input_embeds[n * ctx.m..(n + 1) * ctx.m][real_m_start..real_m_start + M_CHUNK];
                        let input_embeds_f32 = ArrayView::from_shape((M_CHUNK,), input_embeds_i8).unwrap().mapv(|x| (x as f32) / 127.5);
                        input_embeds_f32
                    };
                    for k in 0..ctx.k {
                        let small_elem = if is_decoder { 
                            ctx.sparse_weights[n * ctx.k + k].to_f32()
                         } else {
                            decoder_grads[n * ctx.k + k] as f32
                         };
                         grad_accum += &(&(&big_elem * small_elem) / (ctx.n as f32));
                    }
                }
                let grad_slice = grad_accum.as_slice().unwrap();
                let grad_slice_bf16_uninit = grad_slice.into_iter().map(|&x| MaybeUninit::new(bf16::from_f32(x))).collect::<Vec<MaybeUninit<bf16>>>();
                outputs[l * ctx.m + m_start .. l * ctx.m + m_start + M_CHUNK].copy_from_slice(grad_slice_bf16_uninit.as_slice());
            }
        });
    unsafe { v.set_len(lm * 2) };
    v
}

fn backward(ctx: &BackwardPassContext) {
    println!("Benchmarking compute_grads...");
    let decoder_grads = time_fn!(compute_grads(&ctx));
    println!("First decoder grads: {:?}", &decoder_grads[0..32]);
    println!("Benchmarking weight_grads...");
    let weight_grads = time_fn!(weight_grads(&ctx, &decoder_grads));
    println!("First encoder weight grads: {:?}", &weight_grads[0..32]);
    println!("First decoder weight grads: {:?}", &weight_grads[ctx.m..ctx.m + 32]);
}

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

    // let ctx = SparseMatmulContext::from_vec(
    //     N,
    //     K,
    //     L,
    //     M,
    //     &sparse_weights,
    //     &sparse_indices,
    //     &decoder_weights,
    // );
    // benchmark(
    //     simd_parallel_sparse_matmul,
    //     ctx,
    //     "simd_parallel_sparse_matmul",
    // ); // 20.33s
    // benchmark(
    //     unsafe_alloc_parallel_sparse_matmul,
    //     ctx,
    //     "unsafe_alloc_parallel_sparse_matmul",
    // ); // 17.332739998s
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
