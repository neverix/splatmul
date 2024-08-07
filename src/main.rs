#![feature(array_chunks)]
#![feature(portable_simd)]
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


fn main() {
    let sparse_weights = generate_weights::<{N*K}>(50.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);
    let sparse_indices = generate_indices::<{N*K}>(L as u32);
    println!("First indices: {:?}", &sparse_indices[0..32]);
    let scale = 1.0 / (M as f32).sqrt();
    let decoder_weights = generate_weights::<{L*M}>(scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);
}
