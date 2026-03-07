use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray, PyReadonlyArray2};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    types::{PyAnyMethods, PyModule},
};
use thiserror::Error;

use crate::interface::{Isometry, XFeatOutput};

#[derive(Debug, Error)]
pub enum Matcher3DError {
    #[error("initialization failed: {0}")]
    InitializationFailed(#[source] PyErr),
    #[error("initialization failed: {0}")]
    StepFailed(#[source] PyErr),
}

pub struct Matcher3DPython {
    object: Py<PyAny>,
}

#[derive(Debug, Clone)]
pub struct MatcherOutput {
    pub isometry: Isometry,
    pub proposed_points: Array2<f32>,
    pub proposed_descriptors: Array2<f32>,
}

impl Matcher3DPython {
    pub fn initialize(
        left_calibration: Array2<f32>,
        right_calibration: Array2<f32>,
    ) -> Result<Self, Matcher3DError> {
        Python::attach::<_, PyResult<_>>(|py| {
            let vo_module = PyModule::import(py, "visual_odometry")?;
            let vo_class = vo_module.getattr("VisualOdometryMatcher")?;

            let object = vo_class.call1((
                left_calibration.into_pyarray(py),
                right_calibration.into_pyarray(py),
            ))?;

            Ok(Self {
                object: object.unbind(),
            })
        })
        .map_err(Matcher3DError::InitializationFailed)
    }

    pub fn step(&self, features: XFeatOutput) -> Result<MatcherOutput, Matcher3DError> {
        Python::attach::<_, PyResult<_>>(|py| {
            let output = self.object.bind(py).call_method1("step", (features,))?;
            let (isometry, proposed_points, proposed_descriptors): (
                Bound<'_, PyAny>,
                PyReadonlyArray2<f32>,
                PyReadonlyArray2<f32>,
            ) = output.extract()?;

            let address: usize = isometry.call_method0("_get_memory_address")?.extract()?;
            let isometry = unsafe { *(address as *const Isometry) };
            Ok(MatcherOutput {
                isometry,
                proposed_points: proposed_points.as_array().to_owned(),
                proposed_descriptors: proposed_descriptors.as_array().to_owned(),
            })
        })
        .map_err(Matcher3DError::StepFailed)
    }
}
