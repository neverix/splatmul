use crate::attempts::alloc_uninit_sync;
use crate::benchmarking::SparseMatmulContext;
use crate::constants::{K, M, N};
use crate::conversions::{simd_bf16_to_f32, simd_f32_to_bf16};
use half::bf16;
use rayon::prelude::*;
use std::simd::prelude::*;

fn transmute_bf16_to_u16(bf16: &[bf16]) -> &[u16] {
    unsafe { std::mem::transmute::<&[bf16], &[u16]>(bf16) }
}

fn transmute_u16_to_bf16(u16: &[u16]) -> &[bf16] {
    unsafe { std::mem::transmute::<&[u16], &[bf16]>(u16) }
}

pub fn unsafe_alloc_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Box<[bf16; N * M]> {
    let mut big_box = alloc_uninit_sync(ctx);
    big_box
        .par_chunks_mut(M)
        .enumerate()
        .for_each(|(n, outputs)| {
            for m_start in (0..M).step_by(64) {
                let mut simd_accum = f32x64::splat(0.0);
                for k in 0..K {
                    let weight = ctx.sparse_weights[n * K + k].to_f32();
                    let index = ctx.sparse_indices[n * K + k];
                    let decoder_row_slice = &ctx.decoder_weights
                        [index as usize * M..(index as usize + 1) * M][m_start..m_start + 64];
                    let decoder_row_slice_u16 = transmute_bf16_to_u16(&decoder_row_slice);
                    let simd_decoder_row_slice = u16x64::from_slice(&decoder_row_slice_u16);
                    let simd_weight = f32x64::splat(weight);
                    simd_accum += simd_bf16_to_f32(simd_decoder_row_slice) * simd_weight;
                }
                let simd_result = simd_f32_to_bf16(simd_accum);
                outputs[m_start..m_start + 64]
                    .copy_from_slice(transmute_u16_to_bf16(&simd_result.to_array()));
            }
        });
    big_box
}

fn transmute_u16_to_bf16_owned<const S: usize>(u16: [u16; S]) -> [bf16; S] {
    u16.map(|x| unsafe { std::mem::transmute::<u16, bf16>(x) })
}

pub fn simd_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Vec<[bf16; 64]>> {
    (0..N)
        .into_par_iter()
        .map(|n| {
            (0..M)
                .into_par_iter()
                .step_by(64)
                .map(|m_start| {
                    let mut simd_accum = f32x64::splat(0.0);
                    for k in 0..K {
                        let weight = ctx.sparse_weights[n * K + k].to_f32();
                        let index = ctx.sparse_indices[n * K + k];
                        let decoder_row_slice = &ctx.decoder_weights
                            [index as usize * M..(index as usize + 1) * M]
                            [m_start..m_start + 64];
                        let decoder_row_slice_u16 = transmute_bf16_to_u16(&decoder_row_slice);
                        let simd_decoder_row_slice = u16x64::from_slice(&decoder_row_slice_u16);
                        let simd_weight = f32x64::splat(weight);
                        simd_accum += simd_bf16_to_f32(simd_decoder_row_slice) * simd_weight;
                    }
                    let simd_result = simd_f32_to_bf16(simd_accum);
                    transmute_u16_to_bf16_owned(simd_result.to_array())
                })
                .collect()
        })
        .collect()
}
