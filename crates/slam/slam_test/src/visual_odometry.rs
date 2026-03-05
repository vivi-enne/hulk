use color_eyre::eyre::{Context, Result};
use coordinate_systems::{NewSlamMap, PreviousSlamMap, SlamMap};
use linear_algebra::{IntoFramed, Isometry3, vector};
use nalgebra::{Quaternion, UnitQuaternion};
use ndarray::{Array2, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, convert::IntoPyArray};
use pyo3::{
    Py, PyAny, PyResult, Python,
    types::{PyAnyMethods, PyModule},
};

pub struct VisualOdometry {
    object: Py<PyAny>,
}

impl VisualOdometry {
    pub fn initialize(
        left_calibration: Array2<f32>,
        right_calibration: Array2<f32>,
    ) -> Result<Self> {
        Python::attach::<_, PyResult<_>>(|py| {
            let sys = PyModule::import(py, "sys")?;
            sys.getattr("path")?
                .call_method1("append", ("python_source",))?;
            let vo_module = PyModule::import(py, "visual_odometry")?;
            let vo_class = vo_module.getattr("VisualOdometry")?;

            let object = vo_class.call1((
                left_calibration.into_pyarray(py),
                right_calibration.into_pyarray(py),
                5,
                "cuda",
            ))?;

            Ok(Self {
                object: object.unbind(),
            })
        })
        .wrap_err("failed to initialize visual odometry")
    }

    pub fn step(
        &self,
        left_image: Array3<u8>,
        right_image: Array3<u8>,
    ) -> Result<Isometry3<PreviousSlamMap, SlamMap>> {
        Python::attach::<_, PyResult<_>>(|py| {
            let bound = self.object.bind(py);
            let output = bound.call_method1(
                "step",
                (left_image.into_pyarray(py), right_image.into_pyarray(py)),
            )?;

            let (translation, quaternion, _proposed_points, _proposed_descriptors): (
                PyReadonlyArray1<f32>,
                PyReadonlyArray1<f32>,
                PyReadonlyArray2<f32>,
                PyReadonlyArray2<f32>,
            ) = output.extract()?;

            let translation = translation.as_array();
            let quaternion = quaternion.as_array();

            let update = Isometry3::<PreviousSlamMap, SlamMap>::from_parts(
                vector![translation[0], translation[1], translation[2],],
                UnitQuaternion::from_quaternion(Quaternion::new(
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                    quaternion[3],
                ))
                .framed(),
            );

            Ok(update)
        })
        .wrap_err("failed to call .step(..)")
    }
}
