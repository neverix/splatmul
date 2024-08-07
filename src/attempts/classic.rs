use crate::benchmarking::SparseMatmulContext;
use crate::constants::{K, M, N};
use half::bf16;
use ndarray::{Array1, ArrayView};
use rayon::prelude::*;

pub fn ugly_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Array1<bf16>> {
    (0..N)
        .into_par_iter()
        .map(|n| {
            let mut accum = Array1::from_elem((M,), 0f32);
            for k in 0..K {
                let weight = ctx.sparse_weights[n * K + k].to_f32();
                let index = ctx.sparse_indices[n * K + k];
                let decoder_row = (&ctx.decoder_weights
                    [index as usize * M..(index as usize + 1) * M])
                    .iter()
                    .map(|x| (*x).to_f32())
                    .collect::<Vec<f32>>();
                accum += &(ArrayView::from_shape((M,), decoder_row.as_slice())
                    .unwrap()
                    .to_owned()
                    * weight);
            }
            accum.map(|x| bf16::from_f32(*x))
        })
        .collect()
}

pub fn naive_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Vec<bf16>> {
    (0..N)
        .into_par_iter()
        .map(|n| {
            (0..M)
                .into_iter()
                .map(|m| {
                    let mut accum = 0f32;
                    for k in 0..K {
                        let weight = ctx.sparse_weights[n * K + k].to_f32();
                        let index = ctx.sparse_indices[n * K + k];
                        let decoder_weight = ctx.decoder_weights[index as usize * M + m].to_f32();
                        accum += decoder_weight * weight;
                    }
                    bf16::from_f32(accum)
                })
                .collect()
        })
        .collect()
}
