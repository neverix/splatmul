pub mod allocation;
pub mod classic;
pub mod simd;

pub use allocation::{alloc_lower_bound, alloc_uninit_sync, limit_parallel_sparse_matmul};
pub use classic::{naive_parallel_sparse_matmul, ugly_parallel_sparse_matmul};
pub use simd::{simd_parallel_sparse_matmul, unsafe_alloc_parallel_sparse_matmul};
