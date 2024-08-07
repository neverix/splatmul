#![feature(portable_simd)]
#![feature(new_uninit)]
use std::time::SystemTime;
use std::simd::prelude::*;

use ndarray::{Array1, ArrayView};
use rand_distr::Uniform;
use rand::Rng;

use rayon::prelude::*;
use half::bf16;


const N: usize = 1 << 20;
const K: usize = 64;
const L: usize = 1 << 20;
const M: usize = 1 << 14;


fn generate_weights<const S: usize>(scale: f32) -> Vec<bf16> {
    let distribution = Uniform::new(-scale, scale);
    (0..S).into_par_iter().map(|_| {
        let mut rng = rand::thread_rng();
        let value = rng.sample(distribution);
        bf16::from_f32(value)
    }).collect()
}

fn generate_indices<const S: usize>(max_value: u32) -> Vec<u32> {
    let distribution = Uniform::new(0, max_value);
    (0..S).into_par_iter().map(|_| {
        let mut rng = rand::thread_rng();
        rng.sample(distribution)
    }).collect()
}

#[derive(Clone, Copy)]
struct SparseMatmulContext<'a> {
    sparse_weights: &'a [bf16; N*K],
    sparse_indices: &'a [u32; N*K],
    decoder_weights: &'a [bf16; L*M],
}

fn benchmark<'a, T>(fun: fn(SparseMatmulContext) -> T, sparse_weights: &'a Vec<bf16>, sparse_indices: &'a Vec<u32>, decoder_weights: &'a Vec<bf16>, fun_name: &str) {
    println!("Benchmarking {}", fun_name);
    let start = SystemTime::now();
    let result = fun(SparseMatmulContext {
        sparse_weights: sparse_weights.as_slice().try_into().unwrap(),
        sparse_indices: sparse_indices.as_slice().try_into().unwrap(),
        decoder_weights: decoder_weights.as_slice().try_into().unwrap()
    });
    let elapsed = start.elapsed().unwrap();
    println!("Elapsed time for {}: {:?}", fun_name, elapsed);
}

fn naive_parallel_sparse_matmul(
    ctx: SparseMatmulContext,
) -> Vec<Array1<bf16>> {
    (0..N).into_par_iter().map(|n| {
        let mut accum = Array1::from_elem((M,), 0f32);
        for k in 0..K {
            let weight = ctx.sparse_weights[n * K + k].to_f32();
            let index = ctx.sparse_indices[n * K + k];
            let decoder_row = (&ctx.decoder_weights[index as usize * M..(index as usize + 1) * M]).iter().map(|x| (*x).to_f32()).collect::<Vec<f32>>();
            accum += &(ArrayView::from_shape((M,), decoder_row.as_slice()).unwrap().to_owned() * weight);
        }
        accum.map(|x| bf16::from_f32(*x))
    }).collect()
}

fn naiver_parallel_sparse_matmul(
    ctx: SparseMatmulContext,
) -> Vec<Vec<bf16>> {
    (0..N).into_par_iter().map(|n| {
        (0..M).into_iter().map(|m| {
            let mut accum = 0f32;
            for k in 0..K {
                let weight = ctx.sparse_weights[n * K + k].to_f32();
                let index = ctx.sparse_indices[n * K + k];
                let decoder_weight = ctx.decoder_weights[index as usize * M + m].to_f32();
                accum += decoder_weight * weight;
            }
            bf16::from_f32(accum)
        }).collect()
    }).collect()
}

fn limit_parallel_sparse_matmul(
    ctx: SparseMatmulContext,
) -> Vec<Vec<bf16>> {
    (0..N).into_par_iter().map(|n| {
        let mut accum = vec![0f32; M];
        for k in 0..K {
            let weight = ctx.sparse_weights[n * K + k].to_f32();
            let index = ctx.sparse_indices[n * K + k];
            let decoder_row = &ctx.decoder_weights[index as usize * M..(index as usize + 1) * M];
            // fake computation
            let mut acc = 0f32;
            for m in 0..M {
                acc += decoder_row[m].to_f32() * weight;
            }
            accum[k] += acc;
        }
        accum.iter().map(|x| bf16::from_f32(*x)).collect()
    }).collect()
}

fn alloc_lower_bound(
    _ctx: SparseMatmulContext
) -> Vec<Vec<bf16>> {
    (0..N).into_par_iter().map(|_| vec![bf16::default(); M]).collect()
}

fn alloc_uninit_sync(
    _ctx: SparseMatmulContext,
) -> Box<[bf16; N*M]> {
    unsafe {
        Box::<[bf16; N*M]>::new_uninit().assume_init()
    }
}

fn simd_bf16_to_f32(bf16: u16x64) -> f32x64 {
    let is_nan = mask32x64::from((bf16 & u16x64::splat(0x7FFFu16)).simd_gt(u16x64::splat(0x7F80u16)));
    let bf16_u32 = bf16.cast::<u32>();
    let is_nan_result = bf16_u32 | u32x64::splat(0x0040u32);

    f32x64::from_bits(is_nan.select(is_nan_result, bf16_u32) << 16)
}

fn simd_f32_to_bf16(f32: f32x64) -> u16x64 {
    let x = f32.to_bits();

    let is_nan = mask16x64::from((x & u32x64::splat(0x7FFF_FFFFu32)).simd_gt(u32x64::splat(0x7F80_0000u32)));
    let nan_result = ((x >> 16) | u32x64::splat(0x0040u32)).cast::<u16>();

    let round_bit = u32x64::splat(1 << 15);
    let zero = u32x64::splat(0);
    let round_mask = (x & round_bit).simd_ne(zero) & (x & (round_bit + round_bit << 1 - 1)).simd_ne(zero);
    let rounded_result = round_mask.select(x >> 16, x >> 16 + 1).cast::<u16>();
    is_nan.select(nan_result, rounded_result)
}

fn transmute_bf16_to_u16(bf16: &[bf16]) -> &[u16] {
    unsafe { std::mem::transmute::<&[bf16], &[u16]>(bf16)}
}

fn transmute_u16_to_bf16(u16: &[u16]) -> &[bf16] {
    unsafe { std::mem::transmute::<&[u16], &[bf16]>(u16)}
}

fn unsafe_alloc_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Box<[bf16; N*M]> {
    let mut big_box = alloc_uninit_sync(ctx);
    big_box.par_chunks_mut(M).enumerate().for_each(|(n, outputs)| {
        for m_start in (0..M).step_by(64) {
            let mut simd_accum = f32x64::splat(0.0);
            for k in 0..K {
                let weight = ctx.sparse_weights[n * K + k].to_f32();
                let index = ctx.sparse_indices[n * K + k];
                let decoder_row_slice = &ctx.decoder_weights[index as usize * M..(index as usize + 1) * M][m_start..m_start+64];
                let decoder_row_slice_u16 = transmute_bf16_to_u16(&decoder_row_slice);
                let simd_decoder_row_slice = u16x64::from_slice(&decoder_row_slice_u16);
                let simd_weight = f32x64::splat(weight);
                simd_accum += simd_bf16_to_f32(simd_decoder_row_slice) * simd_weight;
            }
            let simd_result = simd_f32_to_bf16(simd_accum);
            outputs[m_start..m_start+64].copy_from_slice(transmute_u16_to_bf16(&simd_result.to_array()));
        }
    });
    big_box
}

// fn transmute_u16_to_bf16_owned<const S: usize>(u16: [u16; S]) -> [bf16; S] {
//     u16.map(|x| unsafe { std::mem::transmute::<u16, bf16>(x) })
// }

// fn simd_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Vec<[bf16; 64]>> {
//     (0..N).into_par_iter().map(|n| {
//         (0..M).into_par_iter().step_by(64).map(|m_start| {
//             let mut simd_accum = f32x64::splat(0.0);
//             for k in 0..K {
//                 let weight = ctx.sparse_weights[n * K + k].to_f32();
//                 let index = ctx.sparse_indices[n * K + k];
//                 let decoder_row_slice = &ctx.decoder_weights[index as usize * M..(index as usize + 1) * M][m_start..m_start+64];
//                 let decoder_row_slice_u16 = transmute_bf16_to_u16(&decoder_row_slice);
//                 let simd_decoder_row_slice = u16x64::from_slice(&decoder_row_slice_u16);
//                 let simd_weight = f32x64::splat(weight);
//                 simd_accum += simd_bf16_to_f32(simd_decoder_row_slice) * simd_weight;
//             }
//             let simd_result = simd_f32_to_bf16(simd_accum);
//             transmute_u16_to_bf16_owned(simd_result.to_array())
//         }).collect()
//     }).collect()
// }

fn main() {
    let sparse_weights = generate_weights::<{N*K}>(50.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices::<{N*K}>(L as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);
    let scale = 1.0 / (M as f32).sqrt();
    let decoder_weights = generate_weights::<{L*M}>(scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);

    benchmark(unsafe_alloc_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "unsafe_alloc_parallel_sparse_matmul");  // 15.3s
    // benchmark(alloc_uninit_sync, &sparse_weights, &sparse_indices, &decoder_weights, "alloc_uninit_sync");  // 130ns
    // benchmark(alloc_lower_bound, &sparse_weights, &sparse_indices, &decoder_weights, "alloc_lower_bound");  // 7.8s
    // benchmark(limit_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "limit_parallel_sparse_matmul");  // 15.3s
    // benchmark(naive_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "naive_parallel_sparse_matmul");  // 17.9s
    // benchmark(naiver_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "naiver_parallel_sparse_matmul");  // 32.6s
}
