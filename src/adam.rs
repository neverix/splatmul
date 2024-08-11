use std::cmp::min;

use pyo3::prelude::*;
use staticsort::staticsort;


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
        let scales = vec![value; num_blocks];
        let value_dtq = f32_to_dtq(value, signed);
        let block = vec![f32_to_dtq(1.0, signed); length];
        BlockScaled {
            scales,
            block,
            block_size,
            signed
        }
    }
}

#[pyclass]
pub struct AdamState {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub block_size: usize,
    pub m: BlockScaled,
    pub v: BlockScaled,
    pub t: u64,
}

// impl AdamState {
//     pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, block_size: usize) -> Self {
//         let m = BlockScaled::new(block_size);
//         let v = BlockScaled::new(block_size);
//         let t = 0;
//         AdamState {
//             learning_rate,
//             beta1,
//             beta2,
//             epsilon,
//             m,
//             v,
//             t,
//         }
//     }
// }
