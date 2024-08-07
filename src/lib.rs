#![feature(portable_simd)]
#![feature(new_uninit)]
pub mod attempts;
pub mod benchmarking;
pub mod conversions;
pub mod generate;

use benchmarking::SparseMatmulContext;
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;

// #[pymodule]
// fn splatmul<'py>(m: Bound<'py, PyModule>) -> PyResult<()> {
//     fn splatmul_ndarray(
//         sparse_weights: ArrayView2<'_, u16>,
//         sparse_indices: ArrayView2<'_, u32>,
//         decoder_weights: ArrayView2<'_, u16>,
//     ) -> Array2<u16> {
//         let ctx = SparseMatmulContext::from_ndarray(sparse_weights, sparse_indices, decoder_weights);
//     }

//     Ok(())
// }
