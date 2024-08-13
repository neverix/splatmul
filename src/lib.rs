// for DTQ
#![feature(const_trait_impl)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_mut_refs)]
#![feature(portable_simd)]
#![feature(new_uninit)]

pub mod adam;
pub mod attempts;
pub mod backward;
pub mod benchmarking;
pub mod conversions;
pub mod generate;
pub mod types;

use adam::AdamState;
use attempts::classic::beautiful_parallel_sparse_matmul;
use backward::{backward, BackwardPassContext};
use benchmarking::SparseMatmulContext;
use half::bf16;
use numpy::ndarray::Array2;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyInt;
use pyo3::types::PyFloat;
use pyo3::types::PyTuple;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

fn transmute_u16_to_bf16(u16: &[u16]) -> &[bf16] {
    unsafe { std::mem::transmute::<&[u16], &[bf16]>(u16) }
}

macro_rules! assert_std_layout {
    ($x: ident) => {
        assert!(
            $x.as_array().is_standard_layout(),
            concat!(stringify!($x), " is not standard layout")
        );
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
        let encoder_grad_nd = Array2::from_shape_vec(
            [l, m],
            encoder_grad.into_iter().map(|x| x.to_bits()).collect(),
        )
        .unwrap()
        .into_dyn();
        let decoder_grad_nd = Array2::from_shape_vec(
            [l, m],
            decoder_grad.into_iter().map(|x| x.to_bits()).collect(),
        )
        .unwrap()
        .into_dyn();
        PyTuple::new_bound(
            py,
            vec![
                PyArrayDyn::from_owned_array_bound(py, encoder_grad_nd),
                PyArrayDyn::from_owned_array_bound(py, decoder_grad_nd),
            ],
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "matmat")]
    fn matmat_py<'py>(
        py: Python<'py>,
        l: Bound<'py, PyInt>,
        m: Bound<'py, PyInt>,
    ) -> Bound<'py, PyTuple> {
        let l = l.extract::<usize>().unwrap();
        let m = m.extract::<usize>().unwrap();
        let encoder_weights = generate::generate_weights(l * m, 1.0 / (m as f32).sqrt());
        let decoder_weights = encoder_weights.par_iter().map(|&x| x).collect::<Vec<bf16>>();
        PyTuple::new_bound(
            py,
            vec![
                PyArrayDyn::from_owned_array_bound(
                    py,
                    Array2::from_shape_vec(
                        [l, m],
                        encoder_weights.into_iter().map(|x| x.to_bits()).collect(),
                    )
                    .unwrap()
                    .into_dyn(),
                ),
                PyArrayDyn::from_owned_array_bound(
                    py,
                    Array2::from_shape_vec(
                        [l, m],
                        decoder_weights.into_iter().map(|x| x.to_bits()).collect(),
                    )
                    .unwrap()
                    .into_dyn(),
                ),
            ],
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "splatsplat")]
    fn splatsplat_py<'py>(
        py: Python<'py>,
        l: Bound<'py, PyInt>,
        m: Bound<'py, PyInt>,
        lr: Bound<'py, PyFloat>,
        b1: Bound<'py, PyFloat>,
        b2: Bound<'py, PyFloat>,
        eps: Bound<'py, PyFloat>,
        block_size: Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, AdamState>> {
        let l = l.extract::<usize>().unwrap();
        let m = m.extract::<usize>().unwrap();
        let lr = lr.extract::<f32>().unwrap();
        let b1 = b1.extract::<f32>().unwrap();
        let b2 = b2.extract::<f32>().unwrap();
        let eps = eps.extract::<f32>().unwrap();
        let block_size = block_size.extract::<usize>().unwrap();
        Bound::new(py, AdamState::new(lr, b1, b2, eps, l * m, block_size))
    }

    #[pyfn(m)]
    #[pyo3(name = "splatmat")]
    fn splatmat_py<'py>(
        py: Python<'py>,
        adam: Bound<'py, AdamState>,
        grads: PyReadonlyArrayDyn<'py, u16>,
        mut weights: PyReadwriteArrayDyn<'py, u16>,
    ) -> PyResult<()> {
        assert_std_layout!(grads);
        assert_std_layout!(weights);
        let grads_vec_bf16 = grads.as_slice().unwrap()
                .iter()
                .map(|&x| bf16::from_bits(x))
                .collect::<Vec<bf16>>();
        let grads_slice_bf16 = grads_vec_bf16.as_slice();
        let weights_slice = weights.as_slice_mut().unwrap();
        let weights_slice_bf16 = unsafe { std::mem::transmute::<&mut [u16], &mut [bf16]>(weights_slice) };
        adam.borrow_mut().update(
            grads_slice_bf16,
            weights_slice_bf16,
        );
        Ok(())
    }
    Ok(())
}
