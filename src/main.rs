#![feature(array_chunks)]
#![feature(portable_simd)]
use rand::Rng;
use std::{arch::x86_64::__m512i, ops::BitAnd, simd::{f32x16, f32x64, num::SimdFloat, u16x32, u16x64, u32x32, u32x64}};

use rayon::prelude::*;
use half::bf16;


const N: usize = 1 << 20;
const K: usize = 64;
const M: usize = 1 << 14;
const RANDOM_CHUNK: usize = 1 << 14;

struct XoroshiroSIMD {
    xoroshiro_state: u32x64,
}

impl XoroshiroSIMD {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let xoroshiro_state: u32x64 = u32x64::from_array([0; 64].map(|_| rng.gen()));
        Self { xoroshiro_state }
    }

    #[inline]
    fn next(&mut self) {
        self.xoroshiro_state ^= self.xoroshiro_state << 13;
        self.xoroshiro_state ^= self.xoroshiro_state >> 17;
        self.xoroshiro_state ^= self.xoroshiro_state << 5;
    }

    #[inline]
    fn write_bf16(&self, into: &mut [bf16; 64], modulus: f32) {
        let state_u32_23 = (self.xoroshiro_state & u32x64::splat(0x007F_FFFF)) | u32x64::splat((((1 << 7) - 1) << 23) | 1);
        let generated_f32s = f32x64::from_bits(state_u32_23);
        let scaled_f32s = f32x64::splat(-modulus * 3.) + (generated_f32s * f32x64::splat(modulus * 2.));
        let again_u32 = scaled_f32s.to_bits();
        let mantissa = again_u32 & u32x64::splat(0x007F_FFFF);
        let rest = again_u32 >> 23;
        let bf16 = (rest << 10) | (mantissa >> 13);

        let (will_be_high, will_be_low) = u32x32::deinterleave(bf16.resize::<32>(0), bf16.rotate_elements_left::<32>().resize::<32>(0));
        let combined_bf16 = will_be_high << 32 | will_be_low;

        let first_half = combined_bf16.resize::<16>(0);
        let second_half = combined_bf16.rotate_elements_left::<16>().resize::<16>(0);

        let into_u32 = unsafe { std::mem::transmute::<&mut[bf16; 64], &mut [u32; 32]>(into) };
        first_half.copy_to_slice(&mut into_u32[0..16]);
        second_half.copy_to_slice(&mut into_u32[16..32]);
    }

    #[inline]
    fn write_u32(&self, into: &mut [u32; 64], modulus: u32) {
        let modulo = self.xoroshiro_state % u32x64::splat(modulus);
        modulo.copy_to_slice(into);
    }
}

#[inline]
fn generate_weights<const S: usize>(generate_into: &mut [bf16; S], scale: f32) {
    generate_into.as_parallel_slice_mut().par_chunks_mut(RANDOM_CHUNK).for_each(|x| {
        let mut rng = XoroshiroSIMD::new();

        x.array_chunks_mut::<64>().for_each(|y| {
            rng.next();
            rng.write_bf16(y, scale);
        });
    });
}

#[inline]
fn generate_indices<const S: usize>(generate_into: &mut [u32; S], max_value: u32) {
    generate_into.as_parallel_slice_mut().par_chunks_mut(RANDOM_CHUNK).for_each(|x| {
        let mut rng = XoroshiroSIMD::new();

        x.array_chunks_mut::<64>().for_each(|y| {
            rng.next();
            rng.write_u32(y, max_value);
        });
    });
}


fn main() {
    let mut sparse_weights: Box<[bf16; N * K]> = vec![bf16::default(); N * K].into_boxed_slice().try_into().unwrap();
    generate_weights(&mut sparse_weights, 50.0);
    println!("First weights: {:?}", &sparse_weights[0..32]);

    let mut sparse_indices: Box<[u32; N * K]> = vec![0; N * K].into_boxed_slice().try_into().unwrap();
    generate_indices(&mut sparse_indices, M as u32);

    let mut decoder_weights: Box<[bf16; N * M]> = vec![bf16::default(); N * M].into_boxed_slice().try_into().unwrap();
    let scale = 1.0 / (M as f32).sqrt();
    generate_weights(&mut decoder_weights, scale);
    println!("First decoder weights: {:?}", &decoder_weights[0..32]);
}
