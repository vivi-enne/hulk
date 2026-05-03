use std::time::SystemTime;

use factrs::{
    containers::Key,
    core::Values,
    linalg::{ForwardProp, Numeric, VectorX},
    traits::{Diff, Residual},
    variables::{MatrixLieGroup, SE23},
};
use nalgebra::{Const, Matrix3, Point2, Point3};

use crate::{
    camera_intrinsics::CameraIntrinsics, sparse_gaussian_process::SE23SparseGaussianProcessSegment,
    symbols::State,
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
        self.features.len()
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
            let detection = measurement.feature.cast::<T>();
            let residual = measurement
                .candidates
                .iter()
                // Apply robot transform
                .map(|global| state.apply(global.cast::<T>().coords.as_view()))
                // TODO: Apply camera extrinsic transform
                .map(|robot| robot)
                // Transform to camera space
                .map(|camera| intrinsics.project(camera.as_view()))
                .map(|projection| (detection - projection).coords.norm())
                .softmin(self.sharpness);
            residuals[index] = residual;
        }
        residuals
    }
}

trait SoftminExt {
    type Output;

    /// Computes the softmin using log-sum-exp.
    /// `alpha` controls the sharpness where a larger values means the `softmin` operation is closer to a hard `min`.
    fn softmin(self, alpha: f64) -> Self::Output;
}

impl<Iter, T> SoftminExt for Iter
where
    Iter: IntoIterator<Item = T>,
    T: Numeric,
{
    type Output = T;

    fn softmin(self, alpha: f64) -> Self::Output {
        let v = self.into_iter().map(|s| (-s * alpha).exp()).sum::<T>();
        -v.ln() / alpha
    }
}
