use crate::benchmarking::SparseMatmulContext;
use crate::constants::{K, M, N};
use half::bf16;
use rayon::prelude::*;

pub fn limit_parallel_sparse_matmul(ctx: SparseMatmulContext) -> Vec<Vec<bf16>> {
    (0..N)
        .into_par_iter()
        .map(|n| {
            let mut accum = vec![0f32; M];
            for k in 0..K {
                let weight = ctx.sparse_weights[n * K + k].to_f32();
                let index = ctx.sparse_indices[n * K + k];
                let decoder_row =
                    &ctx.decoder_weights[index as usize * M..(index as usize + 1) * M];
                // fake computation
                let mut acc = 0f32;
                for m in 0..M {
                    acc += decoder_row[m].to_f32() * weight;
                }
                accum[k] += acc;
            }
            accum.iter().map(|x| bf16::from_f32(*x)).collect()
        })
        .collect()
}

pub fn alloc_lower_bound(_ctx: SparseMatmulContext) -> Vec<Vec<bf16>> {
    (0..N)
        .into_par_iter()
        .map(|_| vec![bf16::default(); M])
        .collect()
}

pub fn alloc_uninit_sync(_ctx: SparseMatmulContext) -> Box<[bf16; N * M]> {
    unsafe { Box::<[bf16; N * M]>::new_uninit().assume_init() }
}
