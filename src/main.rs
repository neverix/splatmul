#![feature(portable_simd)]
use std::time::SystemTime;

use ndarray::{Array1, ArrayView};
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

fn benchmark<T>(fun: fn(&[bf16; N*K], &[u32; N*K], &[bf16; L*M]) -> T, sparse_weights: &Vec<bf16>, sparse_indices: &Vec<u32>, decoder_weights: &Vec<bf16>, fun_name: &str) {
    println!("Benchmarking {}", fun_name);
    let start = SystemTime::now();
    let result = fun(sparse_weights.as_slice().try_into().unwrap(), sparse_indices.as_slice().try_into().unwrap(), decoder_weights.as_slice().try_into().unwrap());
    let elapsed = start.elapsed().unwrap();
    println!("Elapsed time for {}: {:?}", fun_name, elapsed);
}

fn naive_parallel_sparse_matmul(
    sparse_weights: &[bf16; N*K],
    sparse_indices: &[u32; N*K],
    decoder_weights: &[bf16; L*M],
) -> Vec<Array1<bf16>> {
    (0..N).into_par_iter().map(|n| {
        let mut accum = Array1::from_elem((M,), 0f32);
        for k in 0..K {
            let weight = sparse_weights[n * K + k].to_f32();
            let index = sparse_indices[n * K + k];
            let decoder_row = (&decoder_weights[index as usize * M..(index as usize + 1) * M]).iter().map(|x| (*x).to_f32()).collect::<Vec<f32>>();
            accum += &(ArrayView::from_shape((M,), decoder_row.as_slice()).unwrap().to_owned() * weight);
        }
        accum.map(|x| bf16::from_f32(*x))
    }).collect()
}

fn naiver_parallel_sparse_matmul(
    sparse_weights: &[bf16; N*K],
    sparse_indices: &[u32; N*K],
    decoder_weights: &[bf16; L*M],
) -> Vec<Vec<bf16>> {
    (0..N).into_par_iter().map(|n| {
        (0..M).into_iter().map(|m| {
            let mut accum = 0f32;
            for k in 0..K {
                let weight = sparse_weights[n * K + k].to_f32();
                let index = sparse_indices[n * K + k];
                let decoder_weight = decoder_weights[index as usize * M + m].to_f32();
                accum += decoder_weight * weight;
            }
            bf16::from_f32(accum)
        }).collect()
    }).collect()
}

fn limit_parallel_sparse_matmul(
    sparse_weights: &[bf16; N*K],
    sparse_indices: &[u32; N*K],
    decoder_weights: &[bf16; L*M],
) -> Vec<Vec<bf16>> {
    (0..N).into_par_iter().map(|n| {
        let mut accum = vec![0f32; M];
        for k in 0..K {
            let weight = sparse_weights[n * K + k].to_f32();
            let index = sparse_indices[n * K + k];
            let decoder_row = &decoder_weights[index as usize * M..(index as usize + 1) * M];
            // fake computation
            let mut acc = 0f32;
            for m in 0..M {
                acc += decoder_row[m].to_f32() * weight;
            }
            accum[k] += acc;
        }
        accum.iter().map(|x| bf16::from_f32(*x)).collect()
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

    // benchmark(limit_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "limit_parallel_sparse_matmul");  // 15.3s
    // benchmark(naive_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "naive_parallel_sparse_matmul");  // 17.9s
    // benchmark(naiver_parallel_sparse_matmul, &sparse_weights, &sparse_indices, &decoder_weights, "naiver_parallel_sparse_matmul");  // 32.6s
}
