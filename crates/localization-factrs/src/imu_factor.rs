use std::time::SystemTime;

use factrs::{
    core::Vector3,
    linalg::{ForwardProp, Matrix3, Numeric, VectorX},
    residuals::Residual2,
    traits::{Diff, Residual, Variable},
    variables::{MatrixLieGroup, SE23},
};
use nalgebra::{Const, dvector};

use crate::{
    measurements::ImuMeasurement, sparse_gaussian_process::SE23SparseGaussianProcessSegment,
};

#[derive(Debug, Clone)]
pub struct IntervalGaussianProcessImuFactor {
    measurements: Vec<ImuMeasurement>,
    gyroscope_noise: Matrix3<f64>,
    accelerometer_noise: Matrix3<f64>,
    gyroscope_information_root: Matrix3<f64>,
    accelerometer_information_root: Matrix3<f64>,
    gravity: Vector3<f64>,
    start_time: SystemTime,
    end_time: SystemTime,
}

impl Residual for IntervalGaussianProcessImuFactor {
    fn dim_in(&self) -> usize {
        2 * SE23::<f64>::DIM
    }

    fn dim_out(&self) -> usize {
        self.measurements.len() * 6
    }

    fn residual(&self, values: &factrs::core::Values, keys: &[factrs::containers::Key]) -> VectorX {
        let [k1, k2] = keys else {
            panic!("expected 2 keys")
        };
        let v1 = values.get_unchecked::<_, SE23>(*k1).unwrap();
        let v2 = values.get_unchecked::<_, SE23>(*k2).unwrap();
        self.residuals_on_spline(v1.clone(), v2.clone())
    }

    fn residual_jacobian(
        &self,
        values: &factrs::core::Values,
        keys: &[factrs::containers::Key],
    ) -> factrs::linalg::DiffResult<VectorX, factrs::linalg::MatrixX> {
        let [k1, k2] = keys else {
            panic!("expected 2 keys")
        };
        let v1 = values.get_unchecked::<_, SE23>(*k1).unwrap();
        let v2 = values.get_unchecked::<_, SE23>(*k2).unwrap();

        ForwardProp::<Const<18>>::jacobian_2(
            |start, end| self.residuals_on_spline(start, end),
            v1,
            v2,
        )
    }
}

impl Residual2 for IntervalGaussianProcessImuFactor {
    type V1 = SE23;
    type V2 = SE23;
    type DimIn = Const<18>;
    type DimOut = Const<1>;
    type Differ = ForwardProp<Const<18>>;

    fn residual2<T: Numeric>(&self, v1: SE23<T>, v2: SE23<T>) -> VectorX<T> {
        let residuals = self.residuals_on_spline(v1, v2);

        dvector![residuals.norm()]
    }
}

impl IntervalGaussianProcessImuFactor {
    pub fn new(
        measurements: Vec<ImuMeasurement>,
        gyroscope_noise: Matrix3<f64>,
        accelerometer_noise: Matrix3<f64>,
        gravity: Vector3<f64>,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Self {
        // The inverse of the lower Cholesky factor is required to whiten the residuals
        // such that the resulting error vectors have a covariance of the identity matrix.
        let gyroscope_information_root = gyroscope_noise
            .cholesky()
            .expect("gyroscope noise covariance must be positive definite")
            .l()
            .try_inverse()
            .expect("gyroscope lower triangular matrix must be invertible");

        let accelerometer_information_root = accelerometer_noise
            .cholesky()
            .expect("accelerometer noise covariance must be positive definite")
            .l()
            .try_inverse()
            .expect("accelerometer lower triangular matrix must be invertible");

        Self {
            measurements,
            gyroscope_noise,
            accelerometer_noise,
            gyroscope_information_root,
            accelerometer_information_root,
            gravity,
            start_time,
            end_time,
        }
    }

    fn residuals_on_spline<T: Numeric>(
        &self,
        pose_start: SE23<T>,
        pose_end: SE23<T>,
    ) -> VectorX<T> {
        let spline = SE23SparseGaussianProcessSegment::new(
            self.start_time,
            pose_start,
            self.end_time,
            pose_end,
            self.gyroscope_noise,
            self.accelerometer_noise,
        );

        let mut residual = VectorX::<T>::zeros(6 * self.measurements.len());
        for (i, measurement) in self.measurements.iter().enumerate() {
            let current_pose = spline.infer(measurement.time);
            let body_frame_gravity = current_pose
                .rot()
                .inverse()
                .apply(self.gravity.cast::<T>().as_view());

            let inferred_derivative = spline.infer_derivative(measurement.time);
            let predicted_gyro = inferred_derivative.fixed_view::<3, 1>(0, 0);
            let predicted_accel = inferred_derivative.fixed_view::<3, 1>(3, 0) + body_frame_gravity;

            let gyro_residual =
                predicted_gyro - measurement.state.angular_velocity.inner.cast::<T>();
            let accel_residual =
                predicted_accel - measurement.state.linear_acceleration.inner.cast::<T>();

            let whitened_gyroscope_error =
                self.gyroscope_information_root.cast::<T>() * gyro_residual;
            let whitened_accelerometer_error =
                self.accelerometer_information_root.cast::<T>() * accel_residual;

            residual
                .fixed_view_mut::<3, 1>(6 * i, 0)
                .copy_from(&whitened_gyroscope_error);
            residual
                .fixed_view_mut::<3, 1>(6 * i + 3, 0)
                .copy_from(&whitened_accelerometer_error);
        }

        residual
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use booster::ImuState;
    use factrs::core::SO3;
    use linear_algebra::IntoFramed;
    use nalgebra::{Vector3, vector};

    #[test]
    fn test_imu_factor_residual() {
        let now = SystemTime::now();
        let dt = Duration::from_secs_f64(1. / 500.);

        let mut measurements = Vec::new();
        for i in 0..500 {
            let time = now + Duration::from_secs_f64(i as f64 * dt.as_secs_f64());
            measurements.push(ImuMeasurement {
                time,
                state: ImuState {
                    roll_pitch_yaw: Vector3::zeros().framed(),
                    angular_velocity: Vector3::zeros().framed(),
                    linear_acceleration: Vector3::zeros().framed(),
                },
            });
        }
        let imu_factor = IntervalGaussianProcessImuFactor::new(
            measurements,
            Matrix3::identity(),
            Matrix3::identity(),
            Vector3::new(0., 0., 9.81),
            now,
            now + Duration::from_secs(1),
        );
        let start = SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), Vector3::zeros());
        let end = SE23::from_rot_vel_trans(
            SO3::identity(),
            vector![2.0, 0.0, 0.0],
            vector![1.0, 0.0, 0.0],
        );

        let residual = imu_factor.residuals_on_spline(start, end);
        dbg!(residual);
    }
}
