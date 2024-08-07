use rand::{Rng, seq::SliceRandom};
use rand_distr::Uniform;
use splatmul::attempts::unsafe_alloc_parallel_sparse_matmul;
use splatmul::attempts::naive_parallel_sparse_matmul;
use splatmul::benchmarking::{benchmark, SparseMatmulContext};
use splatmul::generate::{generate_indices, generate_weights};

fn setup_mid_matmul() -> SparseMatmulContext {
    let n = 1 << 15;
    let k = 48;
    let l = 1 << 15;
    let m = 1 << 12;
    let sparse_weights = generate_weights(n * k, 50.0);
    assert!(sparse_weights
        .choose_multiple(&mut rand::thread_rng(), 128)
        .all(|&x| x.to_f32().abs() <= 50.0));
    let sparse_indices = generate_indices(n * k, l as u32);
    assert!(sparse_indices
        .choose_multiple(&mut rand::thread_rng(), 128)
        .all(|&x| x < l as u32));
    let scale = 1.0 / (m as f32).sqrt();
    let decoder_weights = generate_weights(l * m, scale);
    assert!(decoder_weights
        .choose_multiple(&mut rand::thread_rng(), 128)
        .all(|&x| x.to_f32().abs() <= scale));

    let ctx =
        SparseMatmulContext::from_vec(n, k, l, m, sparse_weights, sparse_indices, decoder_weights);

    return ctx;
}

#[test]
fn test_unsafe_alloc_mid_matmul() {
    let ctx = setup_mid_matmul();
    let result = benchmark(
        unsafe_alloc_parallel_sparse_matmul,
        &ctx,
        "unsafe_alloc_parallel_sparse_matmul",
    );
    let baseline = naive_parallel_sparse_matmul(&ctx);
    let distribution = Uniform::new(0, ctx.n * ctx.m);
    for _ in 0..128 {
        let i = rand::thread_rng().sample(distribution);
        let (n, m) = (i / ctx.m, i % ctx.m);
        let diff = (baseline[n][m].to_f32() - result[i].to_f32()).abs();
        assert!(diff < 1e-3, "diff: {}, result: {}, baseline: {}", diff, result[i].to_f32(), baseline[n][m].to_f32());
    }
}
