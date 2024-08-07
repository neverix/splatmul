use crate::benchmarking::SparseMatmulContext;
use half::bf16;
use rayon::prelude::*;

pub fn limit_parallel_sparse_matmul(ctx: &SparseMatmulContext) -> Vec<Vec<bf16>> {
    (0..ctx.n)
        .into_par_iter()
        .map(|n| {
            let mut accum = vec![0f32; ctx.m];
            for k in 0..ctx.k {
                let weight = ctx.sparse_weights[n * ctx.k + k].to_f32();
                let index = ctx.sparse_indices[n * ctx.k + k];
                let decoder_row =
                    &ctx.decoder_weights[index as usize * ctx.m..(index as usize + 1) * ctx.m];
                // fake computation
                let mut acc = 0f32;
                for m in 0..ctx.m {
                    acc += decoder_row[m].to_f32() * weight;
                }
                accum[k] += acc;
            }
            accum.iter().map(|x| bf16::from_f32(*x)).collect()
        })
        .collect()
}

pub fn alloc_lower_bound(ctx: &SparseMatmulContext) -> Vec<Vec<bf16>> {
    (0..ctx.n)
        .into_par_iter()
        .map(|_| vec![bf16::default(); ctx.m])
        .collect()
}

pub fn alloc_uninit_sync(ctx: &SparseMatmulContext) -> Box<[bf16]> {
    Vec::<bf16>::with_capacity(ctx.n * ctx.m).into_boxed_slice()
}
