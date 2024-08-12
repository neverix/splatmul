use std::mem::MaybeUninit;

use half::bf16;
use indicatif::{style, ParallelProgressIterator};
use rayon::prelude::*;
use crate::{make_progress, time_fn};
use ndarray::{s, Array, ArrayView, Dim};

use crate::types::{DecoderGradientType, WeightGradientType};


pub struct BackwardPassContext<'a> {
    pub n: usize,
    pub k: usize,
    pub l: usize,
    pub m: usize,
    pub input_embeds: &'a [i8],
    pub target_embeds: &'a [i8],
    pub sparse_weights: &'a [bf16],
    pub sparse_indices: &'a [u32],
    pub output_embeds: &'a [i8],
    pub decoder_weights: &'a [bf16],
    pub encoder_weights: &'a [bf16],
}

type OutputGradientType = Vec<Array<bf16, Dim<[usize; 1]>>>;

fn compute_output_gradient(ctx: &BackwardPassContext) -> OutputGradientType {
    (0..ctx.n).into_par_iter().map(|n| {
        let target_embed_row = &ctx.target_embeds[n * ctx.m..(n + 1) * ctx.m];
        let output_embed_row = &ctx.output_embeds[n * ctx.m..(n + 1) * ctx.m];
        
        let target_embed_row_nd = ArrayView::from_shape((ctx.m,), target_embed_row).unwrap();
        let output_embed_row_nd = ArrayView::from_shape((ctx.m,), output_embed_row).unwrap();

        let target_f32 = target_embed_row_nd.mapv(|x| (x as f32) / 127.5);
        let output_f32 = output_embed_row_nd.mapv(|x| (x as f32) / 127.5);
        (target_f32 - output_f32).mapv(bf16::from_f32)
    }).collect()
}

fn compute_grads(ctx: &BackwardPassContext, output_grads: &OutputGradientType) -> Vec<DecoderGradientType> {
    let mut v = Vec::<f32>::with_capacity(ctx.n * ctx.k);
    make_progress!(v.spare_capacity_mut()
        .par_chunks_mut(ctx.k)
        .enumerate())
        .for_each(|(n, outputs)| {
            let k_grads: Vec<MaybeUninit<f32>> = (0..ctx.k).into_iter().map(|k| {
                let index = ctx.sparse_indices[n * ctx.k + k];
                let decoder_row = &ctx.decoder_weights[index as usize * ctx.m..(index as usize + 1) * ctx.m];

                let decoder_row_nd = ArrayView::from_shape((ctx.m,), decoder_row).unwrap();
                let decoder_row_f32 = decoder_row_nd.mapv(|x| x.to_f32());
                let output_gradient = output_grads[n].mapv(|x| x.to_f32());

                let gradient = output_gradient.dot(&decoder_row_f32);
                MaybeUninit::new(gradient)
            }).collect::<Vec<MaybeUninit<DecoderGradientType>>>();
            outputs.copy_from_slice(&k_grads);
        });
    unsafe { v.set_len(ctx.n * ctx.k) };
    v
}

fn weight_grads_fast<const M_CHUNK: usize>(ctx: &BackwardPassContext, out_grads: &OutputGradientType, decoder_grads: &[DecoderGradientType]) -> (Vec<WeightGradientType>, Vec<WeightGradientType>) {
    let lm = ctx.l * ctx.m;
    let mut output_grads = (0..lm * 2).into_par_iter().map(|_| bf16::ZERO).collect::<Vec<WeightGradientType>>();
    make_progress!(output_grads
        .par_chunks_mut(M_CHUNK * ctx.l)
        .enumerate())
        .for_each(|(sl_start, outputs)| {
            let m_start = (sl_start * M_CHUNK) % (ctx.m * 2);
            let is_decoder = m_start >= ctx.m;
            let real_m_start = if is_decoder { m_start - ctx.m } else { m_start };
            for n in 0..ctx.n {
                let mut big_elem: Box<Option<Array<f32, Dim<[usize; 1]>>>> = Box::new(None);
                for k in 0..ctx.k {
                    let l = ctx.sparse_indices[n * ctx.k + k] as usize;

                    let is_none = (&big_elem).is_none();
                    if is_none {
                        big_elem = Box::new(Some(if is_decoder {
                            let grad_row = &out_grads[n];
                            let grad_chunk = grad_row.slice(s![real_m_start..real_m_start + M_CHUNK]);
                            grad_chunk.map(|&x| x.to_f32())
                        } else {
                            let input_embeds_i8 = &ctx.input_embeds[n * ctx.m + real_m_start..n * ctx.m + real_m_start + M_CHUNK];
                            let input_embeds_f32 = ArrayView::from_shape((M_CHUNK,), input_embeds_i8).unwrap().mapv(|x| (x as f32) / 127.5);
                            input_embeds_f32
                        }));
                    }
                    let small_elem = if is_decoder { 
                        ctx.sparse_weights[n * ctx.k + k].to_f32()
                    } else {
                        decoder_grads[n * ctx.k + k] as f32
                    };
                    let elem = big_elem.clone().unwrap() * small_elem;
                    let grad_addition = &(&elem / (ctx.n as f32));
                    
                    let current_grad = outputs[l..l + M_CHUNK].iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
                    let current_grad_array = ArrayView::from_shape((M_CHUNK,), &current_grad).unwrap();
                    let grad_new = grad_addition + &current_grad_array;
                    let grad_slice = grad_new.as_slice().unwrap();
                    outputs[l..l + M_CHUNK].copy_from_slice(grad_slice.into_iter().map(|&x| bf16::from_f32(x)).collect::<Vec<bf16>>().as_slice());
                }
            }
        });
    let encoder_grads = (0..lm).into_par_iter().map(|i| {
        let l = i / ctx.m;
        let m = i % ctx.m;
        let m_chunk_idx = m / M_CHUNK;
        let m_in_chunk = m % M_CHUNK;
        output_grads[m_chunk_idx * M_CHUNK * ctx.l + l * M_CHUNK + m_in_chunk]
    }).collect();
    let decoder_grads = (0..lm).into_par_iter().map(|i| {
        let l = i / ctx.m;
        let m = i % ctx.m;
        let m_chunk_idx = m / M_CHUNK;
        let m_in_chunk = m % M_CHUNK;
        output_grads[lm + m_chunk_idx * M_CHUNK * ctx.l + l * M_CHUNK + m_in_chunk]
    }).collect();
    (encoder_grads, decoder_grads)
}

pub fn backward(ctx: &BackwardPassContext) -> (Vec<WeightGradientType>, Vec<WeightGradientType>) {
    let output_grads = time_fn!(compute_output_gradient(&ctx), "Benchmarking compute_output_gradient...");
    let decoder_grads = time_fn!(compute_grads(&ctx, &output_grads), "Benchmarking compute_grads...");
    let (encoder_grads, decoder_grads) = time_fn!(weight_grads_fast::<{1 << 7}>(&ctx, &output_grads, &decoder_grads), "Benchmarking weight_grads_fast...");
    (encoder_grads, decoder_grads)
}
