use half::bf16;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand_distr::Uniform;
use rayon::prelude::*;

use crate::make_progress;

pub fn generate_weights(size: usize, scale: f32) -> Vec<bf16> {
    let distribution = Uniform::new(-scale, scale);
    (0..size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let value = rng.sample(distribution);
            bf16::from_f32(value)
        })
        .collect()
}

pub fn generate_indices(size: usize, max_value: u32) -> Vec<u32> {
    let distribution = Uniform::new(0, max_value);
    (0..size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            rng.sample(distribution)
        })
        .collect()
}

pub fn generate_data(size: usize) -> Vec<i8> {
    let distribution = Uniform::new(-128, 127);
    (0..size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            rng.sample(distribution)
        })
        .collect()
}

pub fn generate_orthogonal(n: usize, m: usize, scale: f32) -> Vec<bf16> {
    // random, non-orthogonal weights
    let mut weights = generate_weights(n * m, scale);
    let mut norms = Vec::<f32>::with_capacity(n);
    let owned_array_index = |i, weights: &Vec<bf16>| {
        ArrayView1::from_shape([m], &weights[i * m..(i + 1) * m])
            .unwrap()
            .map(|x| x.to_f32())
    };
    let arr_0 = owned_array_index(0, &weights);
    norms.push(arr_0.dot(&arr_0));
    for i in make_progress!(1..n) {
        let mut arr_current = owned_array_index(i, &weights);
        let og_norm = arr_current.dot(&arr_current);
        norms.push(og_norm);

        let to_sub = (0..i)
            .into_par_iter()
            .map(|j| {
                let mut arr_prev = owned_array_index(j, &weights);
                let dot = arr_current.dot(&arr_prev) / norms[j];
                arr_prev *= dot;
                arr_prev
            })
            .reduce(|| Array1::from_elem((m,), 0f32), |a, b| a + b);
        arr_current -= &to_sub;
        arr_current *= og_norm.sqrt() / arr_current.dot(&arr_current).sqrt();
        weights[i * m..(i + 1) * m]
            .copy_from_slice(&arr_current.mapv(|x| bf16::from_f32(x)).as_slice().unwrap());
    }
    weights
}

#[test]
fn test_orthogonal_init() {
    let n = 1 << 10;
    let m = 1 << 10;
    let scale = 1.0 / (m as f32).sqrt();
    let weights = generate_orthogonal(n, m, scale);
    assert_eq!(weights.len(), n * m);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let arr_i = ArrayView1::from_shape([m], &weights[i * m..(i + 1) * m])
                .unwrap()
                .map(|x| x.to_f32());
            let arr_j = ArrayView1::from_shape([m], &weights[j * m..(j + 1) * m])
                .unwrap()
                .map(|x| x.to_f32());
            let arr_i_norm = arr_i.dot(&arr_i);
            let arr_j_norm = arr_j.dot(&arr_j);
            let arr_ij_norm = arr_i.dot(&arr_j);
            assert!(
                // 🥲
                arr_ij_norm.abs() < 0.5,
                "i: {}, j: {}, dot: {}, i_norm: {}, j_norm: {}, i_j_sqrt: {}",
                i,
                j,
                arr_ij_norm,
                arr_i_norm,
                arr_j_norm,
                (arr_i_norm * arr_j_norm).abs().sqrt()
            );
        }
    }
}

#[test]
fn test_bf16_in_range() {
    for e in -10..30 {
        let scale = 10f32.powi(e);
        let weights = generate_weights(1 << 10, scale);
        assert_eq!(weights.len(), 1 << 10);
        assert!(
            weights
                .iter()
                .all(|&x| x.to_f32().abs() <= scale + scale * 1e-2),
            "scale: {}, power: {}",
            scale,
            e
        );
    }
}

#[test]
fn test_indices_in_range() {
    for max_value in (1..1 << 31).into_iter().step_by(1 << 20) {
        let indices = generate_indices(1 << 10, max_value);
        assert_eq!(indices.len(), 1 << 10);
        assert!(
            indices.iter().all(|&x| x < max_value),
            "max_value: {}",
            max_value
        );
    }
}

#[test]
fn test_input_correct_size() {
    let weights = generate_data(1 << 10);
    assert_eq!(weights.len(), 1 << 10);
}
