// for DTQ
#![feature(const_trait_impl)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_mut_refs)]

#![feature(portable_simd)]
#![feature(new_uninit)]

pub mod attempts;
pub mod benchmarking;
pub mod conversions;
pub mod generate;
pub mod types;
pub mod backward;
pub mod adam;

use attempts::classic::beautiful_parallel_sparse_matmul;
use backward::{backward, BackwardPassContext};
use benchmarking::SparseMatmulContext;
use half::bf16;
use numpy::ndarray::Array2;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn transmute_u16_to_bf16(u16: &[u16]) -> &[bf16] {
    unsafe { std::mem::transmute::<&[u16], &[bf16]>(u16) }
}

macro_rules! assert_std_layout {
    ($x: ident) => {
        assert!($x.as_array().is_standard_layout(), concat!(stringify!($x), " is not standard layout"));
    };
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
    ) -> Bound<'py, PyArrayDyn<i8>> {
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
        let l = decoder_weights.as_array().shape()[0];
        let m = decoder_weights.as_array().shape()[1];
        let sparse_weights_slice = sparse_weights.as_slice().unwrap();
        let sparse_indices_slice = sparse_indices.as_slice().unwrap();
        let decoder_weights_slice = decoder_weights.as_slice().unwrap();
        let result = Python::allow_threads(py, move || {
            let ctx = SparseMatmulContext {
                n: n as usize,
                k: k as usize,
                l: l as usize,
                m: m as usize,
                sparse_weights: transmute_u16_to_bf16(sparse_weights_slice),
                sparse_indices: sparse_indices_slice,
                decoder_weights: transmute_u16_to_bf16(decoder_weights_slice),
            };
            beautiful_parallel_sparse_matmul(ctx)
        });
        PyArrayDyn::from_owned_array_bound(
            py,
            Array2::from_shape_vec([n, m], result).unwrap().into_dyn(),
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "matsplat")]
    fn matsplat_py<'py>(
        py: Python<'py>,
        input_embeds: PyReadonlyArrayDyn<'py, i8>,
        target_embeds: PyReadonlyArrayDyn<'py, i8>,
        encoder_weights: PyReadonlyArrayDyn<'py, u16>,
        decoder_weights: PyReadonlyArrayDyn<'py, u16>,
        sparse_weights: PyReadonlyArrayDyn<'py, u16>,
        sparse_indices: PyReadonlyArrayDyn<'py, u32>,
        output_embeds: PyReadonlyArrayDyn<'py, i8>,
    ) -> Bound<'py, PyTuple> {
        assert_std_layout!(sparse_weights);
        assert_std_layout!(sparse_indices);
        assert_std_layout!(decoder_weights);
        assert_std_layout!(encoder_weights);
        assert_std_layout!(input_embeds);
        assert_std_layout!(output_embeds);
        assert_eq!(
            sparse_weights.as_array().shape(),
            sparse_indices.as_array().shape(),
            "sparse_weights and sparse_indices have different shapes"
        );
        let n = sparse_weights.as_array().shape()[0];
        let k = sparse_weights.as_array().shape()[1];
        let l = decoder_weights.as_array().shape()[0];
        let m = decoder_weights.as_array().shape()[1];
        let sparse_weights_slice = sparse_weights.as_slice().unwrap();
        let sparse_indices_slice = sparse_indices.as_slice().unwrap();
        let decoder_weights_slice = decoder_weights.as_slice().unwrap();
        let encoder_weights_slice = encoder_weights.as_slice().unwrap();
        let input_embeds_slice = input_embeds.as_slice().unwrap();
        let output_embeds_slice = output_embeds.as_slice().unwrap();
        let target_embeds_slice = target_embeds.as_slice().unwrap();
        let (encoder_grad, decoder_grad) = Python::allow_threads(py, move || {
            let ctx = BackwardPassContext {
                n: n as usize,
                k: k as usize,
                l: l as usize,
                m: m as usize,
                input_embeds: input_embeds_slice,
                output_embeds: output_embeds_slice,
                target_embeds: target_embeds_slice,
                encoder_weights: transmute_u16_to_bf16(encoder_weights_slice),
                decoder_weights: transmute_u16_to_bf16(decoder_weights_slice),
                sparse_weights: transmute_u16_to_bf16(sparse_weights_slice),
                sparse_indices: sparse_indices_slice,
            };
            backward(&ctx)
        });
        let encoder_grad_nd = Array2::from_shape_vec([l, m], encoder_grad.into_iter().map(|x| x.to_bits()).collect()).unwrap().into_dyn();
        let decoder_grad_nd = Array2::from_shape_vec([l, m], decoder_grad.into_iter().map(|x| x.to_bits()).collect()).unwrap().into_dyn();
        PyTuple::new_bound(
            py,
            vec![PyArrayDyn::from_owned_array_bound(py, encoder_grad_nd),
                 PyArrayDyn::from_owned_array_bound(py, decoder_grad_nd)],
        )
    }

    Ok(())
}
