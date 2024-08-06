#![feature(array_chunks)]
#![feature(portable_simd)]
use rand::Rng;
use std::simd::u32x64;

use rayon::prelude::*;
use half::bf16;


const N: usize = 1 << 20;
const K: usize = 64;
const M: usize = 1 << 14;


#[inline]
fn generate_floats<const S: usize>(generate_into: &mut [bf16; S]) {
    generate_into.as_parallel_slice_mut().par_chunks_mut(16384).for_each(|x| {
        let mut rng = rand::thread_rng();
        let mut xoroshiro_state: u32x64 = u32x64::from_array([0; 64].map(|_| rng.gen()));

        let mask_gen = 0b100011_1111111111u32;
        let mask = u32x64::splat(mask_gen);
        let add_gen = 0b101100u32 << 10;
        let add = u32x64::splat(add_gen);

        x.array_chunks_mut::<64>().for_each(|y| {
            xoroshiro_state ^= xoroshiro_state << 13;
            xoroshiro_state ^= xoroshiro_state >> 17;
            xoroshiro_state ^= xoroshiro_state << 5;
            let generated_bf16s = ((xoroshiro_state & mask) | add).to_array();

            y.iter_mut().enumerate().for_each(|(i, z)| {
                *z = bf16::from_bits(generated_bf16s[i] as u16);
            });
        });
    });
}

// #[inline]
// fn generate_floats<const S: usize>(generate_into: &mut [bf16; S]) {
//     generate_into.as_parallel_slice_mut().par_chunks_mut(16384).for_each(|x| {
//         let mut rng = rand::thread_rng();

//         x.iter_mut().for_each(|y| {
//             *y = bf16::from_f32(rng.gen());
//         });
//     });
// }

// #[inline]
// fn generate_floats<const S: usize>(generate_into: &mut [bf16; S]) {
//     let mut rng = rand::thread_rng();
//     generate_into.iter_mut().for_each(|y| {
//         *y = bf16::from_f32(rng.gen());
//     });
// }


fn main() {
    let mut sparse_weights: Box<[bf16; N * K]> = vec![bf16::from_f32(0.0); N * K].into_boxed_slice().try_into().unwrap();
    generate_floats(&mut sparse_weights);
    println!("Generated floats: {:?}", &sparse_weights[0..10]);
}
