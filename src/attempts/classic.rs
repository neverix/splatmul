use std::mem::MaybeUninit;

use crate::{benchmarking::SparseMatmulContext, make_progress};
use half::{bf16, slice::HalfFloatSliceExt};
use ndarray::{Array1, ArrayView};
use rayon::prelude::*;

pub fn ugly_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Array1<bf16>> {
    (0..ctx.n)
        .into_par_iter()
        .map(|n| {
            let mut accum = Array1::from_elem((ctx.m,), 0f32);
            for k in 0..ctx.k {
                let weight = ctx.sparse_weights[n * ctx.k + k].to_f32();
                let index = ctx.sparse_indices[n * ctx.k + k];
                let decoder_row = (&ctx.decoder_weights
                    [index as usize * ctx.m..(index as usize + 1) * ctx.m])
                    .iter()
                    .map(|x| (*x).to_f32())
                    .collect::<Vec<f32>>();
                accum += &(ArrayView::from_shape((ctx.m,), decoder_row.as_slice())
                    .unwrap()
                    .to_owned()
                    * weight);
            }
            accum.map(|x| bf16::from_f32(*x))
        })
        .collect()
}

pub fn beautiful_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<i8> {
    let mut v = Vec::<i8>::with_capacity(ctx.n * ctx.m);
    make_progress!(v.spare_capacity_mut()
        .par_chunks_mut(ctx.m))
        .enumerate()
        .for_each(|(n, outputs)| {
            let mut accum = Array1::from_elem((ctx.m,), 0f32);
            for k in 0..ctx.k { 
                let weight = ctx.sparse_weights[n * ctx.k + k].to_f32();
                let index = ctx.sparse_indices[n * ctx.k + k];
                let decoder_row = (&ctx.decoder_weights
                    [index as usize * ctx.m..(index as usize + 1) * ctx.m]).to_f32_vec();
                accum += &(ArrayView::from_shape((ctx.m,), decoder_row.as_slice())
                    .unwrap()
                    .to_owned()
                    * weight);
            }
            for m in 0..ctx.m {
                outputs[m] = MaybeUninit::new((accum[m] * 127.5f32).clamp(-128., 127.) as i8);
            }
        });
    unsafe { v.set_len(ctx.n * ctx.m) };
    v
}

pub fn naive_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Vec<bf16>> {
    (0..ctx.n)
        .into_par_iter()
        .map(|n| {
            (0..ctx.m)
                .into_iter()
                .map(|m| {
                    let mut accum = 0f32;
                    for k in 0..ctx.k {
                        let weight = ctx.sparse_weights[n * ctx.k + k].to_f32();
                        let index = ctx.sparse_indices[n * ctx.k + k];
                        let decoder_weight =
                            ctx.decoder_weights[index as usize * ctx.m + m].to_f32();
                        accum += decoder_weight * weight;
                    }
                    bf16::from_f32(accum)
                })
                .collect()
        })
        .collect()
}
