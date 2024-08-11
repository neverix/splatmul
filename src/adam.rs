use std::{cmp::min, simd::{cmp::SimdPartialOrd, f32x16, f32x64, num::SimdFloat, LaneCount, Simd, SimdElement, SupportedLaneCount}, time::SystemTime};

use half::{bf16, slice::HalfFloatSliceExt, vec};
use indicatif::ParallelProgressIterator;
use ndarray::{Array1, ArrayView, ArrayView1, ArrayViewMut1, ArrayViewMut2, Axis};
use pyo3::prelude::*;
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use staticsort::staticsort;

use crate::{generate::generate_weights, make_progress};


const fn generate_dtq_lut(SIGNED: bool, divisor: f32) -> [f32; 256] {
    let mut numbers = [0.0; 256];
    let mut sign = if SIGNED { -1.0 } else { 1.0 };
    let max_digits = if SIGNED { 7 } else { 8 };
    while sign <= 1.0 {
        let mut indicator = 0;
        let mut indicator_multiplier = 1f32;
        while indicator < max_digits {
            let digits = max_digits - indicator - 1;
            let mut number: usize;
            if SIGNED && sign == -1.0 {
                number = 128;
            } else {
                number = 0;
            }

            // number += [0] * indicator
            let mut add_value = if SIGNED { 128 } else { 256 };
            let mut indicator_zeros = 0;
            // add one at the position indicated by indicator
            while indicator_zeros <= indicator {
                add_value /= 2;
                indicator_zeros += 1;
            }

            // number += [1]
            number += add_value;
            add_value /= 2;

            let mut remainder: usize = 0;
            while remainder < if add_value > 0 { add_value * 2 } else { 1 } {
                let mut remainder_number = number;
                let mut remainder_add_value = add_value;

                let mut remainder_mod: usize = remainder;
                while remainder_add_value > 0 {
                    if remainder_mod >= remainder_add_value{
                        remainder_number += remainder_add_value;
                        remainder_mod -= remainder_add_value;
                    }
                    remainder_add_value /= 2;
                }

                let value: f32 = sign * indicator_multiplier * (if digits > 0 { (remainder + 1) as f32 } else { 0.0 }) / (2_i32.pow(digits as u32) as f32);
                numbers[remainder_number] = value;
                remainder += 1;
            }
            indicator += 1;
            indicator_multiplier /= divisor;
        }
        // can we have linspace?
        // we have linspace at home
        // linspace at home:
        sign += 2.0;
    };
    numbers
}
const DTQ_SIGNED_UNSORTED: [f32; 256] = generate_dtq_lut(true, 10f32);
pub const DTQ_SIGNED_LUT: [f32; 256] = staticsort!(f32, 0, 255, DTQ_SIGNED_UNSORTED);
const DTQ_UNSIGNED_UNSORTED: [f32; 256] = generate_dtq_lut(false, 10f32);
pub const DTQ_UNSIGNED_LUT: [f32; 256] = staticsort!(f32, 0, 255, DTQ_UNSIGNED_UNSORTED);


#[test]
fn test_dtq_signed_lut() {
    let mut i = 0;
    for value in DTQ_SIGNED_LUT.iter() {
        println!("s{}: {}", i, value);
        i += 1;
    }
}

#[test]
fn test_dtq_unsigned_lut() {
    let mut i = 0;
    for value in DTQ_UNSIGNED_LUT.iter() {
        println!("u{}: {}", i, value);
        i += 1;
    }
}

#[inline]
fn f32_to_dtq<const SIGNED: bool>(value: f32) -> u8 {
    assert!(value.is_finite(), "value must be finite");
    let lut = if SIGNED { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
    let binary_search_result = lut.partition_point(|&x| x < value);
    if binary_search_result == 0 {
        return 0;
    }
    let anchor = min(binary_search_result, 255);
    // function rounds down!
    let candidates = [anchor - 1, anchor];
    let mut best_candidate = 0;
    let mut best_distance = f32::MAX;
    for &candidate in candidates.iter() {
        let distance = (value - lut[candidate as usize]).abs();
        if distance < best_distance {
            best_distance = distance;
            best_candidate = candidate;
        }
    }
    best_candidate as u8
}

fn simd_abs<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> where LaneCount<N>: SupportedLaneCount {
    let zero = Simd::<f32, N>::splat(0.0);
    let mask = value.simd_lt(zero);
    mask.select(-value, value)
}

fn f32_to_dtq_simd<const N: usize, const SIGNED: bool>(value: f32) -> u8 where LaneCount<N>: SupportedLaneCount {
    let lut = if SIGNED { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
    let mut best_distance = f32::MAX;
    let mut best_simd = Simd::<f32, N>::splat(0.0);
    let mut best_candidate = 0;
    let value = Simd::<f32, N>::splat(value);
    for i in (0..256).step_by(N) {
        let lut_simd = Simd::<f32, N>::from_slice(&lut[i..i+N]);
        let distance = simd_abs(value - lut_simd);
        let min_dist = distance.reduce_min();
        if min_dist < best_distance {
            best_distance = min_dist;
            best_simd = distance;
            best_candidate = i;
        }
    }
    let best_distance_simd = Simd::<f32, N>::splat(best_distance);
    let best_candidate = (best_candidate + best_simd.simd_gt(best_distance_simd).to_bitmask().trailing_ones() as usize) as u8;
    best_candidate
}

#[cfg(test)]
fn f32_to_dtq_slow<const SIGNED: bool>(value: f32) -> u8 {
    assert!(value.is_finite(), "value must be finite");
    let lut = if SIGNED { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
    let mut best_distance = f32::MAX;
    let mut best_candidate = 0;
    for (i, lut_value) in lut.iter().enumerate() {
        let distance = (value - lut_value).abs();
        if distance < best_distance {
            best_distance = distance;
            best_candidate = i;
        }
    }
    best_candidate as u8
}

#[inline]
fn dtq_to_f32<const SIGNED: bool>(value: u8) -> f32 {
    let lut = if SIGNED { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
    lut[value as usize]
}

#[test]
fn test_dtq_conv_signed() {
    let maximum = 1024;
    for value_range in 0..(maximum+1) {
        let value = (value_range as f32 - ((maximum / 2) as f32)) / ((maximum / 2) as f32);
        let dtq = f32_to_dtq::<true>(value);
        let undtq = dtq_to_f32::<true>(dtq);
        assert!((value - undtq).abs() < 1e-2, "value: {}, dtq: {}, undtq: {}", value, dtq, undtq);
        let dtq_optimal = f32_to_dtq_slow::<true>(value);
        assert_eq!(dtq, dtq_optimal, "value: {}, dtq: {}, dtq_optimal: {}", value, dtq, dtq_optimal);
        let dtq_simd = f32_to_dtq_simd::<16, true>(value);
        assert_eq!(dtq, dtq_simd, "value: {}, dtq: {}, dtq_simd: {}", value, dtq, dtq_simd);
    }
}

#[test]
fn test_dtq_conv_unsigned() {
    let maximum = 1024;
    for value_range in 0..(maximum+1) {
        let value = (value_range as f32) / (maximum as f32);
        let dtq = f32_to_dtq::<false>(value);
        let undtq = dtq_to_f32::<false>(dtq);
        assert!((value - undtq).abs() < 1e-2, "value: {}, dtq: {}, undtq: {}", value, dtq, undtq);
        let dtq_optimal = f32_to_dtq_slow::<false>(value);
        assert_eq!(dtq, dtq_optimal, "value: {}, dtq: {}, dtq_optimal: {}", value, dtq, dtq_optimal);
        let dtq_simd = f32_to_dtq_simd::<16, false>(value);
        assert_eq!(dtq, dtq_simd, "value: {}, dtq: {}, dtq_simd: {}", value, dtq, dtq_simd);
    }
}


pub struct BlockScaled<const SIGNED: bool> {
    scales: Vec<f32>,
    block: Vec<u8>,
    block_size: usize,
}

const CHUNK_CHUNK: usize = 64;
const ADAM_FLOOR: f32 = 1e-8;

impl<const SIGNED: bool> BlockScaled<SIGNED> {
    pub fn from_elem(length: usize, block_size: usize, value: f32) -> Self {
        assert!(length % (block_size * CHUNK_CHUNK) == 0, "length must be a multiple of block_size");
        assert!(length >= (block_size * CHUNK_CHUNK), "length must be greater than block_size");
        let num_blocks = length / block_size;
        let scale = 1f32.max(value);
        // let scales = vec![scale; num_blocks];
        let scales = (0..num_blocks).into_par_iter().map(|_| scale).collect::<Vec<f32>>();
        let value_dtq = f32_to_dtq::<SIGNED>(value / scale);
        // let block = vec![value_dtq; length];
        let block = (0..length).into_par_iter().map(|_| value_dtq).collect::<Vec<u8>>();
        BlockScaled {
            scales,
            block,
            block_size,
        }
    }

    #[inline]
    pub fn par_for_each(&mut self, visitor: impl Fn(&mut [u8], &mut [f32], usize) + Sync) {
        self.block.as_mut_slice().par_chunks_mut(self.block_size * CHUNK_CHUNK).zip(self.scales.par_chunks_mut(CHUNK_CHUNK)).enumerate().into_par_iter().progress_with_style(make_progress!()).for_each(|(i, (block_chunk, scale))| {
            visitor(block_chunk, scale, i * self.block_size * CHUNK_CHUNK);
        });
    }

    #[inline]
    pub fn par_f32_for_each(&mut self, visitor: impl Fn(&mut [f32], usize) + Sync) {
        let block_size = self.block_size;
        self.par_for_each(|block_chunk, scale, i| {
            let chunk_array = ArrayViewMut2::from_shape((CHUNK_CHUNK, block_size), block_chunk).unwrap();
            let scale_array = ArrayViewMut2::from_shape((CHUNK_CHUNK, 1), scale).unwrap();
            let chunk_array_f32 = chunk_array.mapv(|x| dtq_to_f32::<SIGNED>(x));
            let mut scaled_chunk_array_f32 = chunk_array_f32 * &scale_array;
            scaled_chunk_array_f32.axis_iter_mut(ndarray::Axis(0)).enumerate().for_each(|(j, mut array)| {
                visitor(array.as_slice_mut().unwrap(), i + j * block_size);
            });
            let new_scale = scaled_chunk_array_f32.abs().map_axis(Axis(1), |x| ADAM_FLOOR.max(*x.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()));
            scaled_chunk_array_f32 /= &new_scale.to_shape([CHUNK_CHUNK, 1]).unwrap().to_owned();
            let chunk_dtq = scaled_chunk_array_f32.mapv(|x| f32_to_dtq::<SIGNED>(x));
            block_chunk.copy_from_slice(&chunk_dtq.as_slice().unwrap());
            scale.copy_from_slice(&new_scale.as_slice().unwrap());
        });
    }

    #[inline]
    pub fn par_array_for_each(&mut self, visitor: impl Fn(&mut ArrayViewMut1<f32>, usize) + Sync) {
        let block_size = self.block_size;
        self.par_f32_for_each(|f32_chunk, i| {
            let mut array = ArrayViewMut1::from_shape((block_size,), f32_chunk).unwrap();
            visitor(&mut array, i);
        });
    }

    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = f32> + '_ {
        (0..self.block.len()).into_par_iter().map(|i| dtq_to_f32::<SIGNED>(self.block[i]) * self.scales[i / self.block_size])
    }
}

#[pyclass]
pub struct AdamState {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    block_size: usize,
    pub m: Option<BlockScaled<true>>,
    pub v: BlockScaled<false>,
    pub t: u64,
}

impl AdamState {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, length: usize, block_size: usize) -> Self {
        let use_momentum = beta1 > 0.0;
        let m = match use_momentum {
            true => Some(BlockScaled::<true>::from_elem(length, block_size, 0f32)),
            false => None,
        };
        let v = BlockScaled::<false>::from_elem(length, block_size, 1f32);
        let t = 0;
        AdamState {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            block_size,
            m,
            v,
            t,
        }
    }

    pub fn update(&mut self, gradients: &[bf16], parameters: &mut [bf16]) {
        // update i
        self.t += 1;
        // update m (if present)
        if let Some(ref mut m) = self.m {
            m.par_array_for_each(|m_chunk, i| {
                *m_chunk *= self.beta1;
                let mut dst = vec![0f32; self.block_size];
                (&gradients[i..i+self.block_size]).convert_to_f32_slice(dst.as_mut_slice());
                let mut grad_array = Array1::from_shape_vec((self.block_size,), dst).unwrap();
                grad_array *= 1.0 - self.beta1;
                *m_chunk += &grad_array;
            });
        }
        // update v
        self.v.par_array_for_each(|v_chunk, i| {
            *v_chunk *= self.beta2;
            let mut dst = vec![0f32; self.block_size];
            (&gradients[i..i+self.block_size]).convert_to_f32_slice(dst.as_mut_slice());
            let mut grad_array = Array1::from_shape_vec((self.block_size,), dst).unwrap();
            grad_array *= &grad_array.clone();
            grad_array *= 1.0 - self.beta2;
            *v_chunk += &grad_array;
        });
        // update parameters
        if let Some(ref m) = self.m {
            parameters.par_iter_mut().zip(m.par_iter()).zip(self.v.par_iter()).for_each(|((parameter, m), v)| {
                let m_hat = m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v / (1.0 - self.beta2.powi(self.t as i32));
                let update = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                *parameter = bf16::from_f32(parameter.to_f32() - update);
            });
        } else {
            parameters.par_iter_mut().zip(gradients.into_par_iter()).zip(self.v.par_iter()).for_each(|((parameter, &gradient), v)| {
                let gradient = gradient.to_f32();
                let v_hat = v / (1.0 - self.beta2.powi(self.t as i32));
                let update = self.learning_rate * gradient / (v_hat.sqrt() + self.epsilon);
                *parameter = bf16::from_f32(parameter.to_f32() - update);
            });
        }
    }
}

#[test]
fn test_adam_momentum_state() {
    let length = 1 << 12;
    let mut state = AdamState::new(1e-1, 0.9, 0.9, 1e-10, length, 64);
    let mut parameters = vec![bf16::ZERO; length];
    for i in 0..1001 {
        let gradients = parameters.par_iter().enumerate().map(|(i, x)| bf16::from_f32(x.to_f32() - (i % 14) as f32)).collect::<Vec<bf16>>();
        state.update(&gradients, &mut parameters);
        if i % 100 == 0 {
            println!("Step {}; first 32 parameters: {:?}", i, &parameters[0..32]);
        }
    }
}
