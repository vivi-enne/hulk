use std::time::SystemTime;

use factrs::{linalg::Numeric, traits::Variable, variables::SE23};
use nalgebra::{Matrix3, SMatrix, SVector};

pub struct SE23SparseGaussianProcessSegment<T: Numeric> {
    start_time: SystemTime,
    start_pose: SE23<T>,
    interpolation_factor: SVector<T, 9>,
    gyro_noise: Matrix3<f64>,
    accelerometer_noise: Matrix3<f64>,
}

impl<T: Numeric> SE23SparseGaussianProcessSegment<T> {
    pub fn new(
        start_time: SystemTime,
        start_pose: SE23<T>,
        end_time: SystemTime,
        end_pose: SE23<T>,
        gyro_noise: Matrix3<f64>,
        accelerometer_noise: Matrix3<f64>,
    ) -> Self {
        let duration = end_time
            .duration_since(start_time)
            .expect("invalid time order")
            .as_secs_f64();
        let covariance_between =
            Self::noise_covariance(&gyro_noise, &accelerometer_noise, duration).cast::<T>();

        let relative_algebra_error = start_pose.inverse().compose(&end_pose).log();
        let relative_algebra_error: SVector<T, 9> =
            SVector::from_column_slice(relative_algebra_error.as_slice());
        let interpolation_factor = covariance_between
            .cholesky()
            .expect("covariance must be positive definite")
            .solve(&relative_algebra_error);

        SE23SparseGaussianProcessSegment {
            start_time,
            start_pose,
            interpolation_factor,
            gyro_noise,
            accelerometer_noise,
        }
    }
    #[allow(non_snake_case)]
    pub fn infer(&self, tau: SystemTime) -> SE23<T> {
        let duration_prev = tau
            .duration_since(self.start_time)
            .expect("segment must start before timestamp");
        let Phi_prev = Self::transition_matrix(duration_prev.as_secs_f64()).cast::<T>();
        let Q_prev = Self::noise_covariance(
            &self.gyro_noise,
            &self.accelerometer_noise,
            duration_prev.as_secs_f64(),
        )
        .cast::<T>();

        let projected = Phi_prev.transpose() * self.interpolation_factor;
        let error = Q_prev * projected;

        self.start_pose.oplus_right(error.as_view())
    }

    #[allow(non_snake_case)]
    fn transition_matrix(dt: f64) -> SMatrix<f64, 9, 9> {
        let mut Phi = SMatrix::<f64, 9, 9>::identity();
        Phi.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&Matrix3::identity().scale(dt));
        Phi
    }

    #[allow(non_snake_case)]
    fn noise_covariance(
        gyro_noise: &Matrix3<f64>,
        accelerometer_noise: &Matrix3<f64>,
        dt: f64,
    ) -> SMatrix<f64, 9, 9> {
        let mut Q_d = SMatrix::<f64, 9, 9>::zeros();
        Q_d.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&gyro_noise.scale(dt));
        Q_d.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&accelerometer_noise.scale(dt));
        Q_d.fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&accelerometer_noise.scale(dt * dt / 2.));
        Q_d.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&accelerometer_noise.scale(dt * dt / 2.));
        Q_d.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&accelerometer_noise.scale(dt * dt * dt / 3.));
        Q_d
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use factrs::core::{SO3, Vector3};
    use nalgebra::vector;

    use super::*;

    #[test]
    fn test_infer() {
        let t_start = SystemTime::now();
        let gp = SE23SparseGaussianProcessSegment::new(
            t_start,
            SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), Vector3::zeros()),
            t_start + Duration::from_secs(1),
            SE23::from_rot_vel_trans(
                SO3::identity(),
                vector![2.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
            ),
            Matrix3::identity(),
            Matrix3::identity(),
        );

        dbg!(gp.infer(t_start + Duration::from_millis(500),));
    }
}
