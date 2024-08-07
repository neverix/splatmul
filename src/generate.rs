use half::bf16;
use rand::Rng;
use rand_distr::Uniform;
use rayon::prelude::*;

pub fn generate_weights<const S: usize>(scale: f32) -> Vec<bf16> {
    let distribution = Uniform::new(-scale, scale);
    (0..S)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let value = rng.sample(distribution);
            bf16::from_f32(value)
        })
        .collect()
}

pub fn generate_indices<const S: usize>(max_value: u32) -> Vec<u32> {
    let distribution = Uniform::new(0, max_value);
    (0..S)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            rng.sample(distribution)
        })
        .collect()
}
