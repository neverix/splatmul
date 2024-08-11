use std::cmp::min;

use half::bf16;
use ndarray::{Array1, ArrayView, ArrayView1, ArrayViewMut1};
use pyo3::prelude::*;
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use staticsort::staticsort;

use crate::generate::generate_weights;


const fn generate_dtq_lut(signed: bool, divisor: f32) -> [f32; 256] {
    let mut numbers = [0.0; 256];
    let mut sign = if signed { -1.0 } else { 1.0 };
    let max_digits = if signed { 7 } else { 8 };
    while sign <= 1.0 {
        let mut indicator = 0;
        let mut indicator_multiplier = 1f32;
        while indicator < max_digits {
            let digits = max_digits - indicator - 1;
            let mut number: usize;
            if signed && sign == -1.0 {
                number = 128;
            } else {
                number = 0;
            }

            // number += [0] * indicator
            let mut add_value = if signed { 128 } else { 256 };
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
fn f32_to_dtq(value: f32, signed: bool) -> u8 {
    assert!(value.is_finite(), "value must be finite");
    let lut = if signed { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
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

#[cfg(test)]
fn f32_to_dtq_slow(value: f32, signed: bool) -> u8 {
    assert!(value.is_finite(), "value must be finite");
    let lut = if signed { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
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
fn dtq_to_f32(value: u8, signed: bool) -> f32 {
    let lut = if signed { DTQ_SIGNED_LUT } else { DTQ_UNSIGNED_LUT };
    lut[value as usize]
}

#[test]
fn test_dtq_conv_signed() {
    let maximum = 256;
    for value_range in 0..(maximum+1) {
        let value = (value_range as f32 - ((maximum / 2) as f32)) / ((maximum / 2) as f32);
        let dtq = f32_to_dtq(value, true);
        let undtq = dtq_to_f32(dtq, true);
        assert!((value - undtq).abs() < 1e-2, "value: {}, dtq: {}, undtq: {}", value, dtq, undtq);
        let dtq_optimal = f32_to_dtq_slow(value, true);
        assert_eq!(dtq, dtq_optimal, "value: {}, dtq: {}, dtq_optimal: {}", value, dtq, dtq_optimal);
    }
}

#[test]
fn test_dtq_conv_unsigned() {
    let maximum = 256;
    for value_range in 0..(maximum+1) {
        let value = (value_range as f32) / (maximum as f32);
        let dtq = f32_to_dtq(value, false);
        let undtq = dtq_to_f32(dtq, false);
        assert!((value - undtq).abs() < 1e-2, "value: {}, dtq: {}, undtq: {}", value, dtq, undtq);
        let dtq_optimal = f32_to_dtq_slow(value, false);
        assert_eq!(dtq, dtq_optimal, "value: {}, dtq: {}, dtq_optimal: {}", value, dtq, dtq_optimal);
    }
}

// #[test]
// fn test_dtq


#[pyclass]
#[derive(Clone)]
pub struct BlockScaled {
    scales: Vec<f32>,
    block: Vec<u8>,
    block_size: usize,
    signed: bool,
}

impl BlockScaled {
    pub fn from_elem(length: usize, block_size: usize, value: f32, signed: bool) -> Self {
        assert!(length % block_size == 0, "length must be a multiple of block_size");
        let num_blocks = length / block_size;
        let scale = 1f32.max(value);
        let scales = vec![scale; num_blocks];
        let value_dtq = f32_to_dtq(value / scale, signed);
        let block = vec![value_dtq; length];
        BlockScaled {
            scales,
            block,
            block_size,
            signed
        }
    }

    pub fn par_for_each(&mut self, visitor: impl Fn(&mut [u8], &mut f32, usize) + Sync) {
        self.block.as_mut_slice().par_chunks_mut(self.block_size).zip(self.scales.par_iter_mut()).enumerate().for_each(|(i, (block_chunk, scale))| {
            visitor(block_chunk, scale, i * self.block_size);
        });
    }

    pub fn par_f32_for_each(&mut self, visitor: impl Fn(&mut [f32], usize) + Sync) {
        let is_signed = self.signed;
        self.par_for_each(|block_chunk, scale, i| {
            let mut f32_chunk = block_chunk.iter().map(|&x| dtq_to_f32(x, is_signed) * (*scale)).collect::<Vec<f32>>();
            visitor(f32_chunk.as_mut_slice(), i);
            let new_scale = f32_chunk.iter().map(|&x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            block_chunk.copy_from_slice(&f32_chunk.iter().map(|&x| f32_to_dtq(x / new_scale, is_signed)).collect::<Vec<u8>>());
        });
    }

    pub fn par_array_for_each(&mut self, visitor: impl Fn(&mut ArrayViewMut1<f32>, usize) + Sync) {
        let block_size = self.block_size;
        self.par_f32_for_each(|f32_chunk, i| {
            let mut array = ArrayViewMut1::from_shape((block_size,), f32_chunk).unwrap();
            visitor(&mut array, i);
        });
    }

    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = f32> + '_ {
        (0..self.block.len()).into_par_iter().map(|i| dtq_to_f32(self.block[i], self.signed) * self.scales[i / self.block_size])
    }
}

#[pyclass]
pub struct AdamState {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    block_size: usize,
    pub m: Option<BlockScaled>,
    pub v: BlockScaled,
    pub t: u64,
}

impl AdamState {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, length: usize, block_size: usize) -> Self {
        let use_momentum = beta1 > 0.0;
        let m = match use_momentum {
            true => Some(BlockScaled::from_elem(length, block_size, 0f32, true)),
            false => None,
        };
        let v = BlockScaled::from_elem(length, block_size, 1f32, false);
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
        println!("update m");
        // update m (if present)
        if let Some(ref mut m) = self.m {
            m.par_array_for_each(|m_chunk, i| {
                *m_chunk *= self.beta1;
                let mut grad_array = ArrayView1::from_shape((self.block_size,), &gradients[i..i+self.block_size]).unwrap().mapv(|x| x.to_f32());
                grad_array *= 1.0 - self.beta1;
                *m_chunk += &grad_array;
            });
        }
        println!("update v");
        // update v
        self.v.par_array_for_each(|v_chunk, i| {
            v_chunk.mapv_inplace(|x| x * self.beta2);
            let grad_array = ArrayView1::from_shape((self.block_size,), &gradients[i..i+self.block_size]).unwrap().mapv(|x| x.to_f32().powi(2) * (1.0 - self.beta2));
            *v_chunk += &grad_array;
        });
        println!("update parameters");
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
    let length = 1 << 10;
    let mut state = AdamState::new(1e-2, 0.9, 0.9, 1e-10, length, 64);
    let mut parameters = vec![bf16::ZERO; 1 << 10];
    for i in 0..100 {
        let gradients = generate_weights(length, 0.1);
        state.update(&gradients, &mut parameters);
        println!("Step {}; first 32 parameters: {:?}", i, &parameters[0..32]);
    }
}
