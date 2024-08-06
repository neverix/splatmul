#![feature(array_chunks)]
#![feature(portable_simd)]
use rand_distr::Uniform;
use rand::Rng;

use rayon::prelude::*;
use half::bf16;


const N: usize = 1 << 20;
const K: usize = 64;
const M: usize = 1 << 14;
const RANDOM_CHUNK: usize = 1 << 14;

#[inline]
fn generate_weights<const S: usize>(generate_into: &mut [bf16; S], scale: f32) {
    let distribution = Uniform::new(-scale, scale);
    generate_into.as_parallel_slice_mut().par_chunks_mut(RANDOM_CHUNK).for_each(|x| {
        let mut rng = rand::thread_rng();

        x.iter_mut().for_each(|y| {
            *y = bf16::from_f32(rng.sample(distribution));
        });
    });
}

#[inline]
fn generate_indices<const S: usize>(generate_into: &mut [u32; S], max_value: u32) {
    let distribution = Uniform::new(0, max_value);
    generate_into.as_parallel_slice_mut().par_chunks_mut(RANDOM_CHUNK).for_each(|x| {
        let mut rng = rand::thread_rng();

        x.iter_mut().for_each(|y| {
            *y = rng.sample(distribution);
        });
    });
}


fn main() {
    let mut sparse_weights: Box<[bf16; N * K]> = vec![bf16::default(); N * K].into_boxed_slice().try_into().unwrap();
    generate_weights(&mut sparse_weights, 50.0);

    let mut sparse_indices: Box<[u32; N * K]> = vec![0; N * K].into_boxed_slice().try_into().unwrap();
    generate_indices(&mut sparse_indices, M as u32);

    let mut decoder_weights: Box<[bf16; N * M]> = vec![bf16::default(); N * M].into_boxed_slice().try_into().unwrap();
    let scale = 1.0 / (M as f32).sqrt();
    generate_weights(&mut decoder_weights, scale);
}
