use std::mem::MaybeUninit;

use crate::{make_progress, time_fn};
use half::prelude::*;
use ndarray::{Array, Array1, ArrayView, Dim};
use rayon::prelude::*;
use pyo3::prelude::*;
use ouroboros::self_referencing;

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

#[pyclass]
#[self_referencing]
pub struct BackwardOutputs {
    owner: Vec<WeightGradientType>,
    #[borrows(owner)]
    encoder_grads: &'this [WeightGradientType],
    #[borrows(owner)]
    decoder_grads: &'this [WeightGradientType],
}

fn get_output_grad_slice(ctx: &BackwardPassContext, n: usize, m_start: usize, m_end: usize) -> Vec<f32> {
    let target_embed_row = &ctx.target_embeds[n * ctx.m..(n + 1) * ctx.m][m_start..m_end];
    let output_embed_row = &ctx.output_embeds[n * ctx.m..(n + 1) * ctx.m][m_start..m_end];

    let target_embed_row_nd = ArrayView::from_shape((ctx.m,), target_embed_row).unwrap();
    let output_embed_row_nd = ArrayView::from_shape((ctx.m,), output_embed_row).unwrap();

    let target_f32 = target_embed_row_nd.mapv(|x| (x as f32) / 127.5);
    let output_f32 = output_embed_row_nd.mapv(|x| (x as f32) / 127.5);
    (target_f32 - output_f32).to_vec()
}

fn compute_grads(
    ctx: &BackwardPassContext,
    // output_grads: &OutputGradientType,
) -> Vec<DecoderGradientType> {
    let mut v = Vec::<f32>::with_capacity(ctx.n * ctx.k);
    make_progress!(v.spare_capacity_mut().par_chunks_mut(ctx.k).enumerate()).for_each(
        |(n, outputs)| {
            let k_grads: Vec<MaybeUninit<f32>> = (0..ctx.k)
                .into_iter()
                .map(|k| {
                    let index = ctx.sparse_indices[n * ctx.k + k];
                    let decoder_row =
                        &ctx.decoder_weights[index as usize * ctx.m..(index as usize + 1) * ctx.m];

                    let decoder_row_nd = ArrayView::from_shape((ctx.m,), decoder_row).unwrap();
                    let decoder_row_f32 = decoder_row_nd.mapv(|x| x.to_f32());
                    // let output_gradient = Array1::from_shape_vec((ctx.m,), output_grads[n].as_slice().to_f32_vec()).unwrap();
                    let output_gradient = Array1::from_shape_vec(
                        (ctx.m,),
                        get_output_grad_slice(ctx, n, 0, ctx.m),
                    ).unwrap();

                    let gradient = output_gradient.dot(&decoder_row_f32);
                    MaybeUninit::new(gradient)
                })
                .collect::<Vec<MaybeUninit<DecoderGradientType>>>();
            outputs.copy_from_slice(&k_grads);
        },
    );
    unsafe { v.set_len(ctx.n * ctx.k) };
    v
}

fn weight_grads_fast<const M_CHUNK: usize>(
    ctx: &BackwardPassContext,
    // out_grads: &OutputGradientType,
    decoder_grads: &[DecoderGradientType],
) -> BackwardOutputs {
    let lm = ctx.l * ctx.m;
    let mut output_grads = (0..lm * 2)
        .into_par_iter()
        .map(|_| bf16::ZERO)
        .collect::<Vec<WeightGradientType>>();

    make_progress!(output_grads.par_chunks_mut(M_CHUNK * ctx.l).enumerate()).for_each(
        |(sl_start, outputs)| {
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
                            // let grad_row = &out_grads[n];
                            // let grad_chunk =
                            //     &grad_row[real_m_start..real_m_start + M_CHUNK];
                            // Array1::from_shape_vec((M_CHUNK,), grad_chunk.to_f32_vec()).unwrap()
                            Array1::from_shape_vec((M_CHUNK,), get_output_grad_slice(ctx, n, real_m_start, real_m_start + M_CHUNK)).unwrap()
                        } else {
                            let input_embeds_i8 = &ctx.input_embeds
                                [n * ctx.m + real_m_start..n * ctx.m + real_m_start + M_CHUNK];
                            let input_embeds_f32 =
                                ArrayView::from_shape((M_CHUNK,), input_embeds_i8)
                                    .unwrap()
                                    .mapv(|x| (x as f32) / 127.5);
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

                    let current_grad = outputs[l * M_CHUNK..l * M_CHUNK + M_CHUNK]
                        .iter()
                        .map(|x| x.to_f32())
                        .collect::<Vec<f32>>();
                    let current_grad_array =
                        ArrayView::from_shape((M_CHUNK,), &current_grad).unwrap();
                    let grad_new = grad_addition + &current_grad_array;
                    let grad_slice = grad_new.as_slice().unwrap();
                    outputs[l * M_CHUNK..l * M_CHUNK + M_CHUNK].copy_from_slice(
                        grad_slice
                            .into_iter()
                            .map(|&x| bf16::from_f32(x))
                            .collect::<Vec<bf16>>()
                            .as_slice(),
                    );
                }
            }
        },
    );
    BackwardOutputsBuilder {
        owner: output_grads,
        encoder_grads_builder: |owner: &Vec<WeightGradientType>| &owner[0..lm],
        decoder_grads_builder: |owner: &Vec<WeightGradientType>| &owner[lm..lm * 2],
    }.build()

}

pub fn backward<const M_CHUNK: usize>(ctx: &BackwardPassContext) -> BackwardOutputs {
    // println!("output grads");
    // let output_grads = time_fn!(
    //     compute_output_gradient(&ctx),
    //     "Benchmarking compute_output_gradient..."
    // );
    println!("decoder grads");
    let decoder_grads = time_fn!(
        compute_grads(
            &ctx,
            // &output_grads
        ),
        "Benchmarking compute_grads..."
    );
    println!("encoder grads");
    let grads = time_fn!(
        weight_grads_fast::<M_CHUNK>(
            &ctx,
            // &output_grads,
            &decoder_grads),
        "Benchmarking weight_grads_fast..."
    );
    println!("done");
    grads
}
