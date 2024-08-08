use half::bf16;
use rand::Rng;
use rand_distr::Uniform;
use rayon::prelude::*;

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
