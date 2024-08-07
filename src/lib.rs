#![feature(portable_simd)]
#![feature(new_uninit)]
pub mod attempts;
pub mod benchmarking;
pub mod conversions;
pub mod generate;

use attempts::unsafe_alloc_parallel_sparse_matmul;
use benchmarking::SparseMatmulContext;
use half::bf16;
use numpy::ndarray::Array2;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn transmute_u16_to_bf16(u16: &[u16]) -> &[bf16] {
    unsafe { std::mem::transmute::<&[u16], &[bf16]>(u16) }
}

#[pymodule]
fn splatmul<'py>(m: Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "splatmul")]
    fn splatmul_py<'py>(
        py: Python<'py>,
        sparse_weights: PyReadonlyArrayDyn<'py, u16>,
        sparse_indices: PyReadonlyArrayDyn<'py, u32>,
        decoder_weights: PyReadonlyArrayDyn<'py, u16>,
    ) -> Bound<'py, PyArrayDyn<u16>> {
        assert!(
            sparse_weights.as_array().is_standard_layout(),
            "sparse_weights is not standard layout"
        );
        assert!(
            sparse_indices.as_array().is_standard_layout(),
            "sparse_indices is not standard layout"
        );
        assert!(
            decoder_weights.as_array().is_standard_layout(),
            "decoder_weights is not standard layout"
        );
        assert_eq!(
            sparse_weights.as_array().shape(),
            sparse_indices.as_array().shape(),
            "sparse_weights and sparse_indices have different shapes"
        );
        let n = sparse_weights.as_array().shape()[0];
        let k = sparse_weights.as_array().shape()[1];
        let l = sparse_indices.as_array().shape()[1];
        let m = decoder_weights.as_array().shape()[1];
        let result = {
            let sparse_weights_slice = sparse_weights.as_slice().unwrap();
            let sparse_indices_slice = sparse_indices.as_slice().unwrap();
            let decoder_weights_slice = decoder_weights.as_slice().unwrap();
            let ctx = SparseMatmulContext {
                n: n as usize,
                k: k as usize,
                l: l as usize,
                m: m as usize,
                sparse_weights: transmute_u16_to_bf16(sparse_weights_slice),
                sparse_indices: sparse_indices_slice,
                decoder_weights: transmute_u16_to_bf16(decoder_weights_slice),
            };
            unsafe_alloc_parallel_sparse_matmul(ctx)
                .into_iter()
                .map(|x| x.to_bits())
                .collect()
        };
        PyArrayDyn::from_owned_array_bound(
            py,
            Array2::from_shape_vec([n, m], result).unwrap().into_dyn(),
        )
    }

    Ok(())
}
