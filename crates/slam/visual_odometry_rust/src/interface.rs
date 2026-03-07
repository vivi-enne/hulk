use ndarray::{ArrayView2, ArrayView3};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, ToPyArray, convert::IntoPyArray,
    ndarray::Array2,
};
use pyo3::prelude::*;

use coordinate_systems::Pixel;
use linear_algebra::{
    Point2,
    nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3},
};

#[derive(Debug)]
#[pyclass(frozen, get_all)]
pub struct Viewport {
    pub fov_x: f32,
    pub fov_y: f32,
}

#[derive(Debug, Clone, Copy)]
#[pyclass(frozen)]
#[repr(C)]
pub struct Isometry {
    pub isometry: Isometry3<f32>,
}

#[pymethods]
impl Isometry {
    #[new]
    pub fn from_python(
        translation: PyReadonlyArray1<f32>,
        rotvec: PyReadonlyArray1<f32>,
    ) -> PyResult<Self> {
        let [tx, ty, tz] = translation.as_slice()?.try_into()?;
        let [rx, ry, rz] = rotvec.as_slice()?.try_into()?;
        Ok(Self {
            isometry: Isometry3::from_parts(
                Translation3::new(tx, ty, tz),
                UnitQuaternion::from_scaled_axis(Vector3::new(rx, ry, rz)),
            ),
        })
    }

    #[staticmethod]
    pub fn zero() -> Self {
        Self {
            isometry: Isometry3::default(),
        }
    }

    #[doc(hidden)]
    pub fn _get_memory_address(&self) -> usize {
        self as *const Self as usize
    }
}

#[derive(Debug)]
#[pyclass(frozen, get_all)]
pub struct ExtractedFeatures {
    pub keypoints: Py<PyArray2<f32>>,
    pub descriptors: Py<PyArray2<f32>>,
    pub scores: Py<PyArray1<f32>>,
}

unsafe trait FlatF32Storage: Copy {
    const DIM: usize;

    fn vec_to_array2(mut vec: Vec<Self>) -> Array2<f32> {
        const { assert!(size_of::<Self>() == Self::DIM * size_of::<f32>()) }
        let n_rows = vec.len();
        let n_cols = Self::DIM;

        let length = n_rows * n_cols;
        let capacity = vec.capacity() * n_cols;

        let data = vec.as_mut_ptr() as *mut f32;
        std::mem::forget(vec);

        let flat_vector: Vec<f32> = unsafe { Vec::from_raw_parts(data, length, capacity) };

        Array2::<f32>::from_shape_vec((n_rows, n_cols), flat_vector)
            .expect("unsafe shenanigans gone wrong")
    }
}

unsafe impl FlatF32Storage for Point2<Pixel> {
    const DIM: usize = 2;
}
unsafe impl FlatF32Storage for [f32; 32] {
    const DIM: usize = 32;
}

impl ExtractedFeatures {
    pub fn new(
        keypoints: Vec<Point2<Pixel>>,
        descriptors: Vec<[f32; 32]>,
        scores: Vec<f32>,
    ) -> Self {
        const { assert!(size_of::<Point2<Pixel>>() == 2 * size_of::<f32>()) }
        const { assert!(size_of::<[f32; 32]>() == 32 * size_of::<f32>()) }

        Python::attach(|py| Self {
            keypoints: FlatF32Storage::vec_to_array2(keypoints)
                .into_pyarray(py)
                .unbind(),
            descriptors: FlatF32Storage::vec_to_array2(descriptors)
                .into_pyarray(py)
                .unbind(),
            scores: scores.into_pyarray(py).unbind(),
        })
    }
}

#[derive(Debug)]
#[pyclass(frozen, get_all)]
pub struct XFeatOutput {
    pub keypoints: Py<PyArray3<i64>>,
    pub scores: Py<PyArray2<f32>>,
    pub descriptors: Py<PyArray3<f32>>,
}

impl XFeatOutput {
    pub fn new(
        keypoints: ArrayView3<i64>,
        scores: ArrayView2<f32>,
        descriptors: ArrayView3<f32>,
    ) -> Self {
        Python::attach(|py| Self {
            keypoints: keypoints.to_pyarray(py).unbind(),
            scores: scores.to_pyarray(py).unbind(),
            descriptors: descriptors.to_pyarray(py).unbind(),
        })
    }
}

#[pymodule(name = "visual_odometry_rust")]
pub mod python_module {
    #[pymodule_export]
    use super::{ExtractedFeatures, Isometry, Viewport, XFeatOutput};
}
