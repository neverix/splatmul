use crate::benchmarking::SparseMatmulContext;
use crate::conversions::{simd_bf16_to_f32, simd_f32_to_bf16};
use half::bf16;
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::simd::prelude::*;

fn transmute_bf16_to_u16(bf16: &[bf16]) -> &[u16] {
    unsafe { std::mem::transmute::<&[bf16], &[u16]>(bf16) }
}

pub fn unsafe_alloc_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<bf16> {
    let mut v = Vec::<bf16>::with_capacity(ctx.n * ctx.m);
    v.spare_capacity_mut()
        .par_chunks_mut(ctx.m)
        .enumerate()
        .for_each(|(n, outputs)| {
            for m_start in (0..ctx.m).step_by(64) {
                let mut simd_accum = f32x64::splat(0.0);
                for k in 0..ctx.k {
                    let weight = ctx.sparse_weights[n * ctx.k + k].to_f32();
                    let index = ctx.sparse_indices[n * ctx.k + k];
                    let decoder_row_slice = &ctx.decoder_weights
                        [index as usize * ctx.m..(index as usize + 1) * ctx.m]
                        [m_start..m_start + 64];
                    let decoder_row_slice_u16 = transmute_bf16_to_u16(&decoder_row_slice);
                    let simd_decoder_row_slice = u16x64::from_slice(&decoder_row_slice_u16);
                    let simd_weight = f32x64::splat(weight);
                    simd_accum += simd_bf16_to_f32(simd_decoder_row_slice) * simd_weight;
                }
                let simd_result = simd_f32_to_bf16(simd_accum);
                let array_form = simd_result.to_array();
                let maybe_uninit_array =
                    transmute_u16_to_bf16_owned(array_form).map(|x| MaybeUninit::new(x));
                outputs[m_start..m_start + 64].copy_from_slice(&maybe_uninit_array);
            }
        });
    unsafe { v.set_len(ctx.n * ctx.m) };
    v
}

fn transmute_u16_to_bf16_owned<const S: usize>(u16: [u16; S]) -> [bf16; S] {
    u16.map(|x| unsafe { std::mem::transmute::<u16, bf16>(x) })
}

pub fn simd_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Vec<[bf16; 64]>> {
    (0..ctx.n)
        .into_par_iter()
        .map(|n| {
            (0..ctx.m)
                .into_par_iter()
                .step_by(64)
                .map(|m_start| {
                    let mut simd_accum = f32x64::splat(0.0);
                    for k in 0..ctx.k {
                        let weight = ctx.sparse_weights[n * ctx.k + k].to_f32();
                        let index = ctx.sparse_indices[n * ctx.k + k];
                        let decoder_row_slice = &ctx.decoder_weights
                            [index as usize * ctx.m..(index as usize + 1) * ctx.m]
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
