use std::time::SystemTime;

use factrs::{
    containers::Key,
    core::Values,
    linalg::{ForwardProp, Numeric, VectorX},
    traits::{Diff, Residual, Variable},
    variables::{MatrixLieGroup, SE23},
};
use nalgebra::{Const, Matrix3, Point2, Point3, Vector2};

use crate::{
    camera_intrinsics::CameraIntrinsics, sparse_gaussian_process::SE23SparseGaussianProcessSegment,
};

#[derive(Debug, Clone)]
pub struct LandmarkFactor {
    start_time: SystemTime,
    end_time: SystemTime,
    sharpness: f64,
    features: Vec<FeatureDetection>,
    gyro_noise: Matrix3<f64>,
    accel_noise: Matrix3<f64>,
}

#[derive(Debug, Clone)]
pub struct FeatureDetection {
    /// Time of the detection
    pub time: SystemTime,
    /// The detected feature in image space
    pub feature: Point2<f64>,
    /// Candidate 3d global correspondences
    pub candidates: Vec<Point3<f64>>,
}

impl Residual for LandmarkFactor {
    fn num_keys(&self) -> usize {
        3
    }

    fn dim_in(&self) -> usize {
        22
    }

    fn dim_out(&self) -> usize {
        self.features.len() * 2
    }

    fn residual(&self, values: &Values, keys: &[Key]) -> factrs::linalg::VectorX {
        let (start, end, intrinsics) = unwrap_values(values, keys);
        self.residuals_on_spline(start.clone(), end.clone(), intrinsics.clone())
    }

    fn residual_jacobian(
        &self,
        values: &Values,
        keys: &[Key],
    ) -> factrs::linalg::DiffResult<factrs::linalg::VectorX, factrs::linalg::MatrixX> {
        let (start, end, intrinsics) = unwrap_values(values, keys);
        ForwardProp::<Const<22>>::jacobian_3(
            |start, end, intrinsics| self.residuals_on_spline(start, end, intrinsics),
            start,
            end,
            intrinsics,
        )
    }
}

fn unwrap_values<'a>(
    values: &'a Values,
    keys: &[Key],
) -> (&'a SE23, &'a SE23, &'a CameraIntrinsics) {
    let [k1, k2, k3] = keys else {
        panic!("expected 3 keys")
    };
    let v1 = values.get_unchecked::<_, SE23>(*k1).unwrap();
    let v2 = values.get_unchecked::<_, SE23>(*k2).unwrap();
    let v3 = values.get_unchecked::<_, CameraIntrinsics>(*k3).unwrap();
    (v1, v2, v3)
}

impl LandmarkFactor {
    pub fn residuals_on_spline<T: Numeric>(
        &self,
        pose_start: SE23<T>,
        pose_end: SE23<T>,
        intrinsics: CameraIntrinsics<T>,
    ) -> VectorX<T> {
        let spline = SE23SparseGaussianProcessSegment::new(
            self.start_time,
            pose_start,
            self.end_time,
            pose_end,
            self.gyro_noise,
            self.accel_noise,
        );

        let mut residuals = VectorX::<T>::zeros(self.features.len());
        for (index, measurement) in self.features.iter().enumerate() {
            let state = spline.infer(measurement.time);
            let state_inverse = state.inverse();
            let detection = measurement.feature.cast::<T>();
            let best_residual = measurement
                .candidates
                .iter()
                .map(|global_point| {
                    let robot_point =
                        state_inverse.apply(global_point.cast::<T>().coords.as_view());
                    // TODO: Apply Camera Extrinsics here (T_CR * robot_point)
                    let projection = intrinsics.project(robot_point.as_view());
                    let error_vector = detection.coords - projection;
                    let squared_error = error_vector.norm_squared();
                    (error_vector, squared_error)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(error_vector, _)| error_vector)
                .unwrap_or_else(|| Vector2::zeros());

            residuals
                .fixed_view_mut::<2, 1>(index * 2, 0)
                .copy_from(&best_residual);
        }
        residuals
    }
}
